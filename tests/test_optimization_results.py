"""Checks on numerical artifacts, participant roles, and locked selection timing."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.mark.parametrize("mode", ["optimization_smoke", "optimization"])
def test_nested_executed_results(mode):
    root = Path("results") / mode
    if not (root / "status.json").exists() or json.loads((root / "status.json").read_text())["status"] != "complete":
        pytest.skip(f"{mode} has not completed")
    protocol = json.loads((root / "protocol.json").read_text())
    config = protocol["config"]
    lock_path = root / "selection_locked.json"
    lock = json.loads(lock_path.read_text())
    seal = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert lock["source_sha256"] == protocol["source_sha256"]
    assert not set(protocol["retired_test_subject_ids"]) & set(protocol["development_subject_ids"])
    cohort = pd.read_csv(root / "tables/development_cohort.csv")
    ids = set(cohort.record_id)
    trials = pd.read_csv(root / "tables/inner_selection_trials.csv")
    for choice in lock["selections"]:
        candidates = trials[trials.fold == choice["fold"]]
        assert choice["selection_score"] == pytest.approx(candidates.selection_score.min())
    summary = pd.read_csv(root / "tables/oof_metrics_per_seed.csv")
    for fold in protocol["outer_folds"]:
        groups = fold["participant_ids"]
        assert not (set(groups["train"]) | set(groups["validation"])) & set(groups["outer"])
        expected_train = cohort[cohort.subject_id.isin(groups["train"])][["sbp_mmHg", "dbp_mmHg"]].to_numpy(np.float64)
        for model in config["models"]:
            for seed in config["seeds"]:
                directory = root / "outer" / f"fold_{fold['fold']}" / model / f"seed_{seed}"
                complete = json.loads((directory / "complete.json").read_text())
                assert complete["selection_lock_sha256"] == seal
                checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
                assert set(checkpoint["provenance"]["participant_ids"]["train"]) == set(groups["train"])
                np.testing.assert_allclose(checkpoint["scalers"]["y_mean"], expected_train.mean(0), atol=1e-4, rtol=0)
                np.testing.assert_allclose(checkpoint["scalers"]["y_std"], expected_train.std(0), atol=1e-4, rtol=0)
                predicted = pd.read_csv(directory / "predictions_clean.csv")
                assert set(predicted.subject_id) == set(groups["outer"])
                assert (directory / "predictions_clean.csv").stat().st_mtime_ns >= lock_path.stat().st_mtime_ns
                if model != "legacy_proposed":
                    for condition, missing in [("ecg_missing", "ecg"), ("ppg_missing", "ppg")]:
                        degraded = pd.read_csv(directory / f"predictions_{condition}.csv")
                        np.testing.assert_array_equal(degraded[missing+"_fusion_weight"], 0)
    for model in config["models"]:
        for seed in config["seeds"]:
            frame = pd.read_csv(root / "predictions" / f"{model}_seed_{seed}_clean.csv")
            assert set(frame.record_id) == ids and not frame.record_id.duplicated().any()
            for target in ["sbp", "dbp"]:
                row = summary[(summary.model == model) & (summary.seed == seed) & (summary.condition == "clean") & (summary.target == target.upper())].iloc[0]
                error = frame[target+"_predicted"]-frame[target+"_mmHg"]
                assert row.MAE == pytest.approx(error.abs().mean())
                assert row.RMSE == pytest.approx(np.sqrt((error**2).mean()))
                assert row.SD_error == pytest.approx(error.std(ddof=1))
    final = torch.load(root / "final/optimized_quality/best.pt", map_location="cpu", weights_only=True)
    assert final["fixed_epoch_development_refit"]
    assert set(final["provenance"]["subject_ids"]) == set(protocol["development_subject_ids"])
    assert final["validation_score"] is None
