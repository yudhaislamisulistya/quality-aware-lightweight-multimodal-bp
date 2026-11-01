import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.efficiency import count_macs
from src.factorial import CONDITIONS, constant_prediction, contrast_table, demographic_strata, quality_threshold
from src.factorial_model import CELLS, STUDY_MODELS, FactorialBPModel
from src.optimization import fit_new_model
from src.train import load_model


def test_factorial_cells_have_identical_initial_state_and_budget():
    torch.set_num_threads(2)
    states = []
    for name in CELLS:
        torch.manual_seed(17)
        model = FactorialBPModel(name)
        states.append(model.state_dict())
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        for key in state:
            torch.testing.assert_close(state[key], states[0][key], rtol=0, atol=0)
    x, sqi, auto = torch.randn(2, 2, 3000), torch.rand(2, 2, 7), torch.rand(2, 2)
    for name in CELLS:
        model = FactorialBPModel(name)
        x[:, 1] = 0
        p, _, gates = model(x, sqi, auto)
        torch.testing.assert_close(gates, torch.tensor([[1., 0.], [1., 0.]]))
        p.sum().backward()
        assert all(p.grad is None for p in model.quality.parameters())


@pytest.mark.parametrize("name,channel", [("ecg_only_clean", 0), ("ppg_only_clean", 1)])
def test_unimodal_does_not_use_other_sensor_or_its_sqi(name, channel):
    model = FactorialBPModel(name).eval()
    x, sqi, auto = torch.randn(2, 2, 3000), torch.rand(2, 2, 7), torch.rand(2, 2)
    with torch.inference_mode():
        expected, _, _ = model(x, sqi, auto)
        x[:, 1-channel] = 100
        sqi[:, 1-channel] = 100
        auto[:] = 0
        observed, _, _ = model(x, sqi, auto)
    torch.testing.assert_close(expected, observed, rtol=0, atol=0)
    x[:, channel] = 0
    with pytest.raises(ValueError, match="abstain"):
        model(x, sqi, auto)


def test_dense_control_has_more_macs_and_checkpoint_loader_handles_every_model(tmp_path):
    inputs = [torch.randn(1, 2, 3000), torch.rand(1, 2, 7), torch.rand(1, 2)]
    separable = FactorialBPModel("A_uniform_clean").eval()
    dense = FactorialBPModel("dense_uniform_clean").eval()
    assert count_macs(separable, inputs) < count_macs(dense, inputs)
    assert sum(p.numel() for p in separable.parameters()) < sum(p.numel() for p in dense.parameters())
    for name in STUDY_MODELS:
        model = FactorialBPModel(name).eval()
        path = tmp_path / f"{name}.pt"
        torch.save({"architecture_version": "factorial_v1", "model_name": name, "state_dict": model.state_dict()}, path)
        loaded, _ = load_model(path)
        with torch.inference_mode():
            torch.testing.assert_close(model(*inputs)[0], loaded(*inputs)[0], rtol=0, atol=0)


def test_constants_are_training_only_and_selective_threshold_handles_ties():
    targets = np.array([[100., 60.], [120., 80.], [200., 90.]])
    np.testing.assert_allclose(constant_prediction(targets, 2, "training_median"), [[120, 80], [120, 80]])
    np.testing.assert_allclose(constant_prediction(targets, 1, "training_mean"), [[140, 230/3]])
    assert quality_threshold([.1, .3, .7, .9], .5) == pytest.approx(.5)
    threshold = quality_threshold([.5]*4, .5)
    assert np.mean(np.array([.5]*4) >= threshold) == 1  # Inclusive ties need actual, not nominal coverage.
    assert quality_threshold([.1, .2], 1.) == 0
    with pytest.raises(ValueError):
        quality_threshold([.1], 0.)


def test_demographic_strata_respect_top_coded_age_and_unavailable_sex():
    frame = pd.DataFrame({"age_years": ["90+", "35", None], "sex_code": [2, 0, 1],
                          "sbp_mmHg": [150, 120, 90], "dbp_mmHg": [95, 80, 60],
                          "ecg_quality_label": [0, 0, 1], "ppg_quality_label": [1, 0, 2]})
    masks = demographic_strata(frame)
    assert masks["age_>=60"].tolist() == [True, False, False]
    assert masks["sex_unavailable"].tolist() == [True, False, False]
    assert masks["age_missing"].tolist() == [False, False, True]


def test_factorial_interaction_and_seed_loss_averaging():
    frame = pd.DataFrame({"record_id": ["a", "b", "c", "d"], "subject_id": ["s1", "s1", "s2", "s2"],
                          "sbp_mmHg": [100.]*4, "dbp_mmHg": [70.]*4})
    frames = {}
    for name, delta in zip(CELLS, [10., 9., 7., 5.], strict=True):
        f = frame.copy()
        f["sbp_predicted"], f["dbp_predicted"] = 100+delta, 70+delta
        frames[name] = [f, f.copy()]
    result = contrast_table({c: frames for c in CONDITIONS}, {"protocol_seed": 17, "bootstrap_repetitions": 100,
                            "primary_condition": "clean", "primary_contrasts": ["B_minus_A", "D_minus_C"], "primary_targets": ["SBP", "DBP"]})
    primary = result[result.primary]
    assert len(primary) == 4
    assert set(primary.family_interval_confidence) == {.9875}
    assert set(primary[primary.contrast == "D_minus_C"].delta_MAE) == {-2.}
    interaction = result[result.contrast == "interaction_D_minus_C_minus_B_plus_A"]
    np.testing.assert_array_equal(interaction.delta_MAE, -1.)


def test_clean_training_never_uses_corrupted_inputs_and_fits_scalers_on_training(tmp_path):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({"record_id": ["a", "b", "c", "d"], "subject_id": ["s1", "s1", "s2", "s2"],
                          "sbp_mmHg": [100., 130., 200., 210.], "dbp_mmHg": [60., 80., 100., 110.],
                          "ecg_quality_label": [0, 1, 2, 3], "ppg_quality_label": [0, 0, 1, 2]})
    clean = {"x": rng.normal(size=(4, 2, 3000)).astype(np.float32), "sqi": rng.uniform(size=(4, 2, 7)).astype(np.float32), "automatic": np.ones((4, 2), np.float32)}
    bad = {k: np.full((5, *v.shape), np.nan, dtype=np.float32) for k, v in clean.items()}
    config = json.loads(open("configs/factorial.json").read())
    config.update({"threads": 2, "device": "cpu", "epochs": 1, "training_augmentation": False})
    path = fit_new_model("A_uniform_clean", 17, frame, clean, {"training": bad, "evaluation": bad},
                         {"train": np.array([0, 1]), "validation": np.array([2, 3])}, config, tmp_path, {})
    _, checkpoint = load_model(path)
    assert checkpoint["scalers"]["y_mean"] == [115., 70.]
    assert np.isfinite(checkpoint["validation_score"])
    history = pd.read_csv(tmp_path / "history.csv")
    assert history.train_ranking.iloc[0] == 0
    assert history.validation_corrupted_MAE.isna().all()
