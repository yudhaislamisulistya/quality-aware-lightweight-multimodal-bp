import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.reliability import crossfit_partitions, fit_risk_head, shuffle_goodness
from src.risk_model import POLICIES, RiskFusion, fusion_weights, predict_components
from src.train import load_model


def scalers():
    return [{"y_mean": [120., 80.], "y_std": [20., 10.], "sqi_mode": "bounded_v2"} for _ in range(2)]


def test_target_specific_weights_and_exact_missing_sensor_mask():
    q = torch.tensor([[.8, .2], [.5, .5]])
    risk = torch.tensor([[[2., 8.], [8., 2.]], [[3., 5.], [6., 2.]]])
    available = torch.tensor([[True, True], [True, False]])
    w = fusion_weights("adaptive_risk_fusion", q, risk, torch.ones(2, 2), available)
    torch.testing.assert_close(w[0], torch.tensor([[.8, .2], [.2, .8]]))
    torch.testing.assert_close(w[1], torch.tensor([[1., 1.], [0., 0.]]))
    torch.testing.assert_close(w.sum(1), torch.ones(2, 2))
    with pytest.raises(ValueError, match="abstain"):
        fusion_weights("uniform_fusion", q, risk, torch.ones(2, 2), torch.zeros_like(available))


def test_absolute_risk_is_distinct_from_normalized_weights():
    available, q = torch.ones(1, 2, dtype=torch.bool), torch.full((1, 2), .5)
    low, high = torch.ones(1, 2, 2)*2, torch.ones(1, 2, 2)*20
    w_low = fusion_weights("adaptive_risk_fusion", q, low, torch.ones(2, 2), available)
    w_high = fusion_weights("adaptive_risk_fusion", q, high, torch.ones(2, 2), available)
    torch.testing.assert_close(w_low, w_high)
    assert ((w_high*high).sum(1) > (w_low*low).sum(1)).all()


def test_crossfitting_excludes_held_out_subject_from_teacher_fit_and_selection():
    frame = pd.DataFrame({"subject_id": np.repeat([f"s{i:03}" for i in range(90)], 3)})
    config = {"crossfit_folds": 3, "crossfit_seed": 820391, "inner_validation_fraction": .2}
    folds = crossfit_partitions(frame, config, 0)
    seen = []
    for fold in folds:
        ids = fold["participant_ids"]
        assert not set(ids["outer"]) & (set(ids["train"]) | set(ids["validation"]))
        assert not set(ids["train"]) & set(ids["validation"])
        seen.extend(ids["outer"])
    assert len(seen) == len(set(seen)) == 90


def test_shuffled_gate_is_reproducible_preserves_joint_vectors_and_availability():
    q = np.array([[.1,.2], [.3,.4], [.5,.6], [.7,0], [.9,0]], np.float32)
    available = np.array([[1,1], [1,1], [1,1], [1,0], [1,0]], bool)
    original = q.copy()
    a, b = shuffle_goodness(q, available, 531), shuffle_goodness(q, available, 531)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(q, original)
    assert sorted(map(tuple, a[:3])) == sorted(map(tuple, q[:3]))
    assert sorted(map(tuple, a[3:])) == sorted(map(tuple, q[3:]))
    np.testing.assert_array_equal(a[~available], 0)


def test_missing_risk_targets_are_excluded_from_head_fitting():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(6, 12, 24)).astype(np.float32)
    available = np.ones((6, 12, 2), bool)
    available[4, :, 0], available[5, :, 1] = False, False
    observed = np.abs(rng.normal(10, 2, size=(6, 12, 2, 2))).astype(np.float32)
    observed[~available] = np.nan
    data = {"features": features, "available": available, "observed_absolute_error": observed}
    config = {"ridge_alpha": 10., "risk_floor_mmHg": 1., "risk_clean_sample_weight": .5}
    a, b = RiskFusion(scalers()), RiskFusion(scalers())
    info = fit_risk_head(a, data, config)
    replaced = observed.copy()
    replaced[~available] = np.inf
    fit_risk_head(b, {**data, "observed_absolute_error": replaced}, config)
    torch.testing.assert_close(a.risk_head.weight, b.risk_head.weight, rtol=0, atol=0)
    assert info["parameter_search"] is False
    assert len(info["feature_names"]) == 24
    assert np.all(np.asarray(info["feature_scale"]) > 0)


def test_bp_only_ablation_excludes_explicit_quality_information():
    model = RiskFusion(scalers(), policy="bp_only_risk_fusion").eval()
    rng = np.random.default_rng(17)
    data = {"features": rng.normal(size=(6, 12, 24)).astype(np.float32), "available": np.ones((6,12,2), bool),
            "observed_absolute_error": rng.uniform(2,20,size=(6,12,2,2)).astype(np.float32)}
    fit_risk_head(model,data,{"ridge_alpha":10.,"risk_floor_mmHg":1.,"risk_clean_sample_weight":.5},feature_indices=range(18,24))
    torch.testing.assert_close(model.risk_head.weight[:,:18],torch.zeros(4,18),rtol=0,atol=0)
    x, a = torch.randn(2,2,3000), torch.rand(2,2)
    with torch.inference_mode():
        p1,_,w1=model(x,torch.zeros(2,2,7),a)
        p2,_,w2=model(x,torch.ones(2,2,7),torch.zeros_like(a))
    torch.testing.assert_close(p1,p2,rtol=0,atol=0)
    torch.testing.assert_close(w1,w2,rtol=0,atol=0)


def test_risk_checkpoint_loading_and_convex_prediction_envelope(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = RiskFusion(scalers()).eval()
    x, sqi, auto = torch.randn(3, 2, 3000), torch.rand(3, 2, 7), torch.rand(3, 2)
    with torch.no_grad():
        model.risk_head.bias.copy_(torch.tensor([5., 10., 10., 5.]))
    for policy in POLICIES:
        model.policy = policy
        with torch.inference_mode():
            details = model.components(x, sqi, auto)
        p, experts = details["prediction"], details["expert_prediction"]
        assert (p >= experts.amin(1)-1e-5).all() and (p <= experts.amax(1)+1e-5).all()
        path = tmp_path/f"{policy}.pt"
        torch.save({"architecture_version": "risk_v1", "model_name": policy, "expert_scalers": scalers(),
                    "state_dict": model.state_dict(), "config": {"risk_floor_mmHg": 1.}}, path)
        restored, _ = load_model(path)
        with torch.inference_mode():
            torch.testing.assert_close(model(x,sqi,auto)[0], restored(x,sqi,auto)[0], rtol=0, atol=0)
    x[:, 1] = 0
    with torch.inference_mode():
        details = model.components(x,sqi,auto)
    torch.testing.assert_close(details["weights"][:, 1], torch.zeros(3,2))
    torch.testing.assert_close(details["prediction"], details["expert_prediction"][:, 0])
    with pytest.raises(ValueError, match="abstain"):
        model(torch.zeros_like(x),sqi,auto)


def test_risk_cli_inference_accepts_waveforms_without_manual_labels(tmp_path, monkeypatch):
    import main
    model = RiskFusion(scalers()).eval()
    checkpoint = {"architecture_version": "risk_v1", "model_name": "adaptive_risk_fusion", "expert_scalers": scalers(),
                  "state_dict": model.state_dict(), "config": {"risk_floor_mmHg": 1., "threads": 2},
                  "preprocessing": {}, "scalers": {"sqi_mode": "bounded_v2", "y_mean": [0.,0.], "y_std": [1.,1.]}}
    path = tmp_path/"risk.pt"
    torch.save(checkpoint,path)
    x, sqi, auto = np.ones((2,3000),np.float32), np.zeros((2,7),np.float32), np.zeros(2,np.float32)
    monkeypatch.setattr(main,"read_waveform",lambda *_: None)
    monkeypatch.setattr(main,"preprocess",lambda *_: (x,sqi,auto))
    result = main.inference(path,tmp_path,"record")
    assert result["manual_quality_used"] is False
    assert set(result["fusion_weights_by_target"]) == {"SBP","DBP"}
    assert np.isfinite([result["SBP_mmHg"],result["DBP_mmHg"]]).all()
