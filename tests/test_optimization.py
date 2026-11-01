import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.efficiency import count_macs
from src.optimization import corrupt, nested_subject_splits
from src.optimized_model import OPTIMIZED_MODELS, RobustBPModel
from src.preprocessing import scale_sqi
from src.train import load_model


def test_nested_subject_splits_have_complete_disjoint_outer_coverage():
    frame = pd.DataFrame({"subject_id": np.repeat([f"S{i:03}" for i in range(90)], 4)})
    config = {"outer_folds": 3, "protocol_seed": 290731, "inner_validation_fraction": .2}
    folds = nested_subject_splits(frame, config)
    seen = []
    for fold in folds:
        groups = fold["participant_ids"]
        assert not set(groups["train"]) & set(groups["validation"])
        assert not (set(groups["train"]) | set(groups["validation"])) & set(groups["outer"])
        assert len(groups["outer"]) == 30
        seen.extend(groups["outer"])
    assert len(seen) == len(set(seen)) == 90


def test_bounded_sqi_does_not_extrapolate_and_preserves_legacy_behavior():
    sqi = np.asarray([[[0, .1, .2, 1, np.log1p(10000), .7, 1]]*2], dtype=np.float32)
    scaled = scale_sqi(sqi, {"sqi_mode": "bounded_v2"})
    assert np.isfinite(scaled).all() and scaled.min() >= 0 and scaled.max() <= 1
    assert scaled[0, 0, 3] == 1
    legacy = scale_sqi(sqi, {"sqi_mean": np.zeros((2, 7)).tolist(), "sqi_std": np.ones((2, 7)).tolist()})
    np.testing.assert_array_equal(legacy, sqi)


@pytest.mark.parametrize("name", OPTIMIZED_MODELS)
def test_optimized_models_missing_sensor_and_backward(name):
    torch.set_num_threads(2)
    model = RobustBPModel(name)
    x, sqi, automatic = torch.randn(2, 2, 3000), torch.rand(2, 2, 7), torch.rand(2, 2)
    x[:, 1] = 0
    predicted, logits, gates = model(x, sqi, automatic)
    assert predicted.shape == (2, 2) and torch.isfinite(predicted).all()
    torch.testing.assert_close(gates, torch.tensor([[1., 0.], [1., 0.]]))
    if model.supervised_quality:
        assert (logits[:, 1].sigmoid() < 1e-6).all()
    predicted.square().mean().backward()
    with pytest.raises(ValueError, match="Both modalities"):
        model(torch.zeros_like(x), sqi, automatic)


def test_bp_gradient_cannot_redefine_quality_and_quality_training_still_works():
    model = RobustBPModel("optimized_quality")
    x, sqi, automatic = torch.randn(3, 2, 3000), torch.rand(3, 2, 7), torch.rand(3, 2)
    p, q, gates = model(x, sqi, automatic)
    p.sum().backward()
    assert all(parameter.grad is None for parameter in model.quality.parameters())
    assert any(parameter.grad is not None for parameter in model.encoders.parameters())
    model.zero_grad(set_to_none=True)
    _, logits, _ = model(x, sqi, automatic)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
    assert any(parameter.grad is not None for parameter in model.quality.parameters())
    assert all(parameter.grad is None for parameter in model.encoders.parameters())
    torch.testing.assert_close(gates.sum(1), torch.ones(3))


def test_matched_no_gate_is_independent_of_quality_inputs():
    model = RobustBPModel("optimized_no_gate").eval()
    x = torch.randn(2, 2, 3000)
    with torch.inference_mode():
        a, _, gates = model(x, torch.zeros(2, 2, 7), torch.zeros(2, 2))
        b, _, _ = model(x, torch.ones(2, 2, 7), torch.ones(2, 2))
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(gates, torch.full((2, 2), .5))


def test_training_noise_snr_is_controlled_and_new_stress_changes_only_one_modality():
    t = np.arange(12000)/500
    raw = np.stack([np.sin(2*np.pi*8*t), np.sin(2*np.pi*1.2*t)], axis=1)
    original = raw.copy()
    perturbed = corrupt(raw, "ppg_noise_10dB", 731, snr=7.)
    measured = 20*np.log10(raw[:, 1].std() / (perturbed[:, 1]-raw[:, 1]).std())
    assert measured == pytest.approx(7.)
    for condition, unchanged in [("ecg_baseline_drift", 1), ("ppg_motion", 0), ("ppg_clipping", 0)]:
        altered = corrupt(raw, condition, 731)
        np.testing.assert_array_equal(raw[:, unchanged], altered[:, unchanged])
        assert np.isfinite(altered).all()
    np.testing.assert_array_equal(raw, original)


def test_new_checkpoint_loader_and_parameter_budget(tmp_path):
    model = RobustBPModel("optimized_quality").eval()
    path = tmp_path / "new.pt"
    torch.save({"architecture_version": "quality_v2", "model_name": model.name, "state_dict": model.state_dict()}, path)
    restored, _ = load_model(path)
    args = [torch.randn(1, 2, 3000), torch.rand(1, 2, 7), torch.rand(1, 2)]
    with torch.inference_mode():
        torch.testing.assert_close(model(*args)[0], restored(*args)[0])
    assert sum(p.numel() for p in model.parameters()) < 100000
    assert count_macs(model, args) > 0
