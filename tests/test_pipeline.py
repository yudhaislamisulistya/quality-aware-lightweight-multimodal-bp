import numpy as np
import pandas as pd
import pytest
import torch
import wfdb
from torch import nn

from src.ablation import degrade
from src.dataset_loader import participant_split, read_waveform
from src.efficiency import count_macs
from src.evaluate import cluster_bootstrap, paired_comparisons, regression_metrics
from src.fusion import quality_weights
from src.models import BPModel, MODEL_NAMES
from src.preprocessing import DEFAULT_PREPROCESSING, fit_scalers, preprocess


def test_subject_isolation_and_quick_does_not_touch_full_test():
    cohort = pd.DataFrame({"subject_id": np.repeat([f"S{i:03}" for i in range(100)], 4)})
    full, ids = participant_split(cohort, 2026)
    quick, qids = participant_split(cohort, 2026, [24, 8, 8])
    assert not (set(ids["train"]) & set(ids["test"]))
    assert set(sum(qids.values(), [])) <= set(ids["train"])
    assert not set(quick["train"]) & set(quick["test"])
    assert sum(map(len, full.values())) == len(cohort)


def test_scalers_use_only_training_records():
    sqi = np.asarray([np.zeros((2, 7)), np.ones((2, 7)), np.full((2, 7), 1e8)])
    y = np.asarray([[100, 60], [120, 80], [1e8, 1e8]])
    scalers = fit_scalers(sqi, y, np.asarray([0, 1]))
    np.testing.assert_allclose(scalers["y_mean"], [110, 70])
    np.testing.assert_allclose(scalers["sqi_mean"], .5)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_forward_backward_without_manual_quality(name):
    torch.set_num_threads(2)
    model = BPModel(name)
    x, sqi, automatic = torch.randn(2, 2, 3000), torch.randn(2, 2, 7), torch.rand(2, 2)
    predicted, logits, weights = model(x, sqi, automatic)
    assert predicted.shape == (2, 2)
    assert logits.shape == (2, 2)
    assert torch.isfinite(predicted).all()
    torch.testing.assert_close(weights.sum(1), torch.ones(2))
    predicted.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_quality_weights_and_ablated_gates():
    goodness = torch.tensor([[0., 1.]])
    weights = quality_weights(goodness)
    assert weights[0, 1] > weights[0, 0] > 0
    model = BPModel("proposed_no_gate").eval()
    with torch.inference_mode():
        _, _, gates = model(torch.randn(2, 2, 3000), torch.randn(2, 2, 7), torch.rand(2, 2))
    torch.testing.assert_close(gates, torch.full((2, 2), .5))


def test_metrics_and_participant_bootstrap():
    actual = np.asarray([[100., 60.], [120., 80.], [130., 90.]])
    predicted = actual + np.asarray([2., -3.])
    metrics = regression_metrics(actual, predicted)
    assert metrics[0]["MAE"] == metrics[0]["RMSE"] == metrics[0]["ME"] == 2
    assert metrics[1]["ME"] == -3
    assert metrics[1]["SD_error"] == 0
    ci = cluster_bootstrap(actual, predicted, ["A", "A", "B"], 50, 17)
    assert ci[0]["lower_95"] == ci[0]["upper_95"] == 2


def test_cluster_bootstrap_keeps_record_weights(monkeypatch):
    class FixedDraw:
        def integers(self, high, size):
            return np.asarray([[0, 1]])
    monkeypatch.setattr(np.random, "default_rng", lambda seed: FixedDraw())
    actual = np.zeros((3, 2))
    predicted = np.asarray([[1., 1.], [1., 1.], [10., 10.]])
    ci = cluster_bootstrap(actual, predicted, ["A", "A", "B"], 1, 17)
    assert ci[0]["lower_95"] == ci[0]["upper_95"] == 4


def test_paired_comparison_averages_losses_not_predictions():
    def frame(offset):
        return pd.DataFrame({"subject_id": ["A", "A", "B"], "sbp_mmHg": [100., 110., 120.],
                             "dbp_mmHg": [60., 70., 80.], "sbp_predicted": np.asarray([100., 110., 120.])+offset,
                             "dbp_predicted": np.asarray([60., 70., 80.])+offset})
    results = paired_comparisons({"proposed": [frame(2), frame(-2)], "baseline": [frame(3), frame(-3)]}, 50, 17)
    np.testing.assert_allclose(results.delta_MAE, -1)
    np.testing.assert_allclose(results.lower_95, -1)
    np.testing.assert_allclose(results.upper_95, -1)


def test_preprocessing_and_controlled_noise():
    t = np.arange(12000)/500
    raw = np.stack([np.sin(2*np.pi*8*t), np.sin(2*np.pi*1.2*t)], axis=1)
    x, sqi, automatic = preprocess(raw, DEFAULT_PREPROCESSING)
    assert x.shape == (2, 3000) and sqi.shape == (2, 7)
    np.testing.assert_allclose(x.mean(1), 0, atol=1e-5)
    np.testing.assert_allclose(x.std(1), 1, atol=1e-5)
    assert ((automatic >= 0) & (automatic <= 1)).all()
    noisy = degrade(raw, "ecg_noise_10dB", 17)
    measured_snr = 20*np.log10(raw[:, 0].std()/(noisy[:, 0]-raw[:, 0]).std())
    assert measured_snr == pytest.approx(10)
    np.testing.assert_array_equal(raw[:, 1], noisy[:, 1])
    missing = degrade(raw, "ppg_missing", 17)
    _, missing_sqi, missing_auto = preprocess(missing, DEFAULT_PREPROCESSING)
    assert missing_auto[1] == 0 and missing_sqi[1, 3] == 1
    with pytest.raises(ValueError):
        preprocess(raw[:10], DEFAULT_PREPROCESSING)


def test_wfdb_digital_checksums_and_path_boundary(tmp_path):
    t = np.arange(12000)/500
    digital = (np.stack([np.sin(2*np.pi*8*t), np.sin(2*np.pi*1.2*t)], axis=1)*1000).astype(np.int32)
    wfdb.wrsamp("record", fs=500, units=["NU", "NU"], sig_name=["ECG", "PPG"],
                d_signal=digital, fmt=["32", "32"], adc_gain=[1., 1.], baseline=[0, 0], write_dir=str(tmp_path))
    np.testing.assert_array_equal(read_waveform(tmp_path, "record"), digital)
    with pytest.raises(ValueError, match="escapes"):
        read_waveform(tmp_path, "../outside")
    path = tmp_path / "record.dat"
    payload = bytearray(path.read_bytes())
    payload[0] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        read_waveform(tmp_path, "record")


def test_macs_include_groups_correctly():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(2, 2, 3, groups=2)
        def forward(self, x):
            return self.conv(x)
    assert count_macs(Tiny(), [torch.zeros(1, 2, 10)]) == 2*8*3
