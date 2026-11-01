"""Independently recompute results and check the actual executed participant roles."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.factorial import CONDITIONS, CONSTANTS, available_conditions, constant_prediction
from src.factorial_model import CELLS, STUDY_MODELS


@pytest.mark.parametrize("mode", ["factorial_smoke", "factorial"])
def test_executed_factorial_artifacts(mode):
    root = Path("results") / mode
    status_path = root / "status.json"
    if not status_path.exists() or json.loads(status_path.read_text())["status"] != "complete":
        pytest.skip(f"{mode} has not completed")
    protocol = json.loads((root/"protocol.json").read_text())
    config = protocol["config"]
    lock_path = root/"selection_locked.json"
    lock = json.loads(lock_path.read_text())
    seal = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert not protocol["independent_confirmatory_test"]
    assert not lock["outer_predictions_used"]
    assert len(lock["selections"]) == len(STUDY_MODELS)*len(config["seeds"])*3
    cohort = pd.read_csv(root/"tables/development_cohort.csv")
    assert not set(cohort.subject_id) & set(protocol["retired_test_subject_ids"])
    assert not cohort.waveform_sha256.duplicated().any()
    for fold in protocol["outer_folds"]:
        groups = fold["participant_ids"]
        assert not set(groups["train"]) & set(groups["validation"])
        assert not (set(groups["train"]) | set(groups["validation"])) & set(groups["outer"])
    for choice in lock["selections"]:
        fold, name, seed = choice["fold"], choice["model"], choice["seed"]
        groups = protocol["outer_folds"][fold]["participant_ids"]
        fit = root/"fits"/f"fold_{fold}"/name/f"seed_{seed}"
        outer = root/"outer"/f"fold_{fold}"/name/f"seed_{seed}"
        checkpoint_path = fit/"best.pt"
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == choice["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert set(checkpoint["provenance"]["participant_ids"]["train"]) == set(groups["train"])
        train = cohort[cohort.subject_id.isin(groups["train"])][["sbp_mmHg", "dbp_mmHg"]].to_numpy()
        np.testing.assert_allclose(checkpoint["scalers"]["y_mean"], train.mean(0), atol=1e-4, rtol=0)
        assert checkpoint["config"]["ranking_weight"] == 0
        assert checkpoint["config"]["training_augmentation"] == CELLS.get(name, (False, False))[1]
        history = pd.read_csv(fit/"history.csv")
        assert checkpoint["validation_score"] <= history.validation_clean_MAE.min()+.001
        validation = pd.read_csv(fit/"inner_validation_predictions.csv")
        assert set(validation.subject_id) == set(groups["validation"])
        if name in CELLS:
            scores = validation[["ecg_estimated_goodness", "ppg_estimated_goodness"]].min(axis=1)
            for coverage in config["selective_coverages"]:
                expected = 0 if coverage == 1 else scores.quantile(1-coverage)
                assert choice["quality_thresholds"][str(coverage)] == pytest.approx(expected)
        complete = json.loads((outer/"complete.json").read_text())
        assert complete["selection_lock_sha256"] == seal
        for condition in available_conditions(name):
            path = outer/f"predictions_{condition}.csv"
            assert path.stat().st_mtime_ns >= lock_path.stat().st_mtime_ns
            frame = pd.read_csv(path)
            assert set(frame.subject_id) == set(groups["outer"])
            if condition in ["ecg_missing", "ppg_missing"]:
                np.testing.assert_array_equal(frame[condition[:3]+"_fusion_weight"], 0)
    all_metrics = pd.read_csv(root/"tables/metrics_per_seed.csv")
    reference = cohort.set_index("record_id")
    for name in STUDY_MODELS+CONSTANTS:
        for seed in (config["seeds"] if name in STUDY_MODELS else [0]):
            for condition in (available_conditions(name) if name in STUDY_MODELS else CONDITIONS):
                frame = pd.read_csv(root/"predictions"/f"{name}_seed_{seed}_{condition}.csv")
                assert set(frame.record_id) == set(cohort.record_id) and not frame.record_id.duplicated().any()
                np.testing.assert_array_equal(frame[["sbp_mmHg", "dbp_mmHg"]], reference.loc[frame.record_id, ["sbp_mmHg", "dbp_mmHg"]])
                if name in CONSTANTS:
                    for fold in protocol["outer_folds"]:
                        train_ids = fold["participant_ids"]["train"]
                        actual = cohort[cohort.subject_id.isin(train_ids)][["sbp_mmHg", "dbp_mmHg"]].to_numpy()
                        records = frame[frame.outer_fold == fold["fold"]]
                        np.testing.assert_allclose(records[["sbp_predicted", "dbp_predicted"]], constant_prediction(actual, len(records), name))
                for target in ["SBP", "DBP"]:
                    error = frame[target.lower()+"_predicted"]-frame[target.lower()+"_mmHg"]
                    row = all_metrics[(all_metrics.model == name) & (all_metrics.seed == seed) & (all_metrics.condition == condition) & (all_metrics.target == target)].iloc[0]
                    assert row.MAE == pytest.approx(error.abs().mean())
                    assert row.RMSE == pytest.approx(np.sqrt((error**2).mean()))
                    assert row.SD_error == pytest.approx(error.std(ddof=1))
    decisions = pd.read_csv(root/"predictions/selective_clean_decisions.csv")
    np.testing.assert_array_equal(decisions.accepted, decisions.estimated_goodness_min >= decisions.validation_threshold)
    assert decisions[decisions.validation_target_coverage == 1].accepted.all()
    selective = pd.read_csv(root/"tables/selective_risk_coverage.csv")
    full = selective[selective.validation_target_coverage == 1]
    np.testing.assert_allclose(full.selective_MAE, full.all_records_MAE)
    np.testing.assert_allclose(full.random_rejection_MAE_mean, full.all_records_MAE)
    contrasts = pd.read_csv(root/"tables/factorial_contrasts.csv")
    assert len(contrasts[contrasts.primary]) == 4
    np.testing.assert_allclose(contrasts[contrasts.primary].family_interval_confidence, .9875)
    pooled = all_metrics.groupby(["model", "condition", "target"]).MAE.mean()
    for row in contrasts.itertuples():
        if row.condition not in CONDITIONS:
            continue
        expected = sum(weight*pooled.loc[(model, row.condition, row.target)] for model, weight in protocol["contrasts"][row.contrast].items())
        assert row.delta_MAE == pytest.approx(expected, abs=1e-12)
    assert len(list((root/"figures").glob("*.png"))) >= 11
    assert len(list((root/"figures").glob("*.pdf"))) >= 11
