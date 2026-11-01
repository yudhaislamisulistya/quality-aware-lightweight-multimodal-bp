"""Independent checks of artifacts from actually completed suites."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_executed_artifacts_and_leakage(mode):
    root = Path("results") / mode
    if not (root / "status.json").exists() or json.loads((root / "status.json").read_text())["status"] != "complete":
        pytest.skip(f"{mode} suite has not finished")
    config = json.loads((root / "resolved_config.json").read_text())
    split = json.loads((root / "participant_splits.json").read_text())["participant_ids"]
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not set(split[left]) & set(split[right])
    cohort = pd.read_csv(root / "tables/cohort.csv")
    train = cohort[cohort.subject_id.isin(split["train"])][["sbp_mmHg", "dbp_mmHg"]].to_numpy(np.float64)
    expected_ids = set(cohort[cohort.subject_id.isin(split["test"])].record_id)
    assert cohort.waveform_sha256.nunique() == len(cohort)
    assert len(cohort) == 3742
    per_seed = pd.read_csv(root / "tables/per_seed_results.csv")
    assert len(per_seed) == len(config["models"])*len(config["seeds"])
    reference_order = None
    for model in config["models"]:
        for seed in config["seeds"]:
            directory = root / "checkpoints" / model / f"seed_{seed}"
            assert (directory / "complete.json").exists()
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            # Saved fitting is float32; allow sub-0.0001 mmHg reduction roundoff.
            np.testing.assert_allclose(checkpoint["scalers"]["y_mean"], train.mean(0), rtol=0, atol=1e-4)
            np.testing.assert_allclose(checkpoint["scalers"]["y_std"], train.std(0), rtol=0, atol=1e-4)
            predicted = pd.read_csv(directory / "predictions.csv")
            assert set(predicted.record_id) == expected_ids
            assert set(predicted.subject_id) == set(split["test"])
            order = predicted.record_id.tolist()
            if reference_order is None:
                reference_order = order
            assert order == reference_order
            metrics = json.loads((directory / "metrics.json").read_text())
            for c, target in enumerate(["sbp", "dbp"]):
                error = predicted[target+"_predicted"].to_numpy() - predicted[target+"_mmHg"].to_numpy()
                np.testing.assert_allclose(metrics[c]["MAE"], np.abs(error).mean(), rtol=1e-6)
                np.testing.assert_allclose(metrics[c]["RMSE"], np.sqrt(np.mean(error**2)), rtol=1e-6)
                np.testing.assert_allclose(metrics[c]["ME"], error.mean(), atol=1e-8)
                np.testing.assert_allclose(metrics[c]["SD_error"], error.std(ddof=1), rtol=1e-6)
            efficiency = json.loads((directory / "efficiency.json").read_text())
            assert efficiency["parameters"] > 0 and efficiency["MACs_per_record"] > 0
            assert efficiency["model_only"]["batch_1"]["median_latency_ms"] > 0
            assert efficiency["measurement_iterations"] == config["latency_repetitions"]
            history = pd.read_csv(directory / "history.csv")
            assert checkpoint["validation_mae_mmHg"] <= history.validation_mae_mmHg.min() + config["min_delta_mmHg"]
    summary = pd.read_csv(root / "tables/baseline_comparison.csv").set_index("model")
    for name, runs in per_seed.groupby("model"):
        for col in ["SBP_MAE", "DBP_MAE", "SBP_RMSE", "DBP_RMSE"]:
            np.testing.assert_allclose(summary.loc[name, col], runs[col].mean(), rtol=1e-6)
    assert len(list((root / "figures").glob("*.png"))) >= 14
    assert len(list((root / "figures").glob("*.pdf"))) >= 14
