"""Frozen ECG/PPG experts with quality or cross-fitted absolute-error weighting."""
import numpy as np
import torch
from torch import nn

from .factorial_model import FactorialBPModel
from .preprocessing import scale_sqi

POLICIES = ["uniform_fusion", "quality_fusion", "static_risk_fusion", "bp_only_risk_fusion", "adaptive_risk_fusion"]
FEATURE_NAMES = ([f"{m}_sqi_{j}" for m in ["ecg", "ppg"] for j in range(7)]
                 + ["ecg_automatic", "ppg_automatic", "ecg_estimated_goodness", "ppg_estimated_goodness"]
                 + [f"{m}_{t}_prediction" for m in ["ecg", "ppg"] for t in ["sbp", "dbp"]]
                 + ["ecg_available", "ppg_available"])


def fusion_weights(policy, goodness, risks, static_risks, available):
    if not available.any(1).all():
        raise ValueError("Both modalities are absent; abstain from BP estimation")
    if policy == "uniform_fusion":
        scores = torch.ones_like(risks)
    elif policy == "quality_fusion":
        scores = goodness.clamp(min=.05).unsqueeze(-1).expand_as(risks)
    elif policy == "static_risk_fusion":
        scores = static_risks.reciprocal().expand_as(risks)
    elif policy in {"adaptive_risk_fusion", "bp_only_risk_fusion"}:
        scores = risks.reciprocal()
    else:
        raise ValueError(policy)
    scores = scores * available.unsqueeze(-1)
    return scores / scores.sum(1, keepdim=True)


class RiskFusion(nn.Module):
    """Outputs mmHg directly; risk estimates are errors in mmHg, not quality probabilities."""
    def __init__(self, expert_scalers, policy="adaptive_risk_fusion", risk_floor=1.):
        super().__init__()
        if policy not in POLICIES or risk_floor <= 0:
            raise ValueError("Invalid fusion policy or numerical risk floor")
        if any(s.get("sqi_mode") != "bounded_v2" for s in expert_scalers):
            raise ValueError("This study requires bounded SQIs for both reference experts")
        self.experts = nn.ModuleList([FactorialBPModel("ecg_only_clean"), FactorialBPModel("ppg_only_clean")])
        self.register_buffer("expert_mean", torch.tensor([s["y_mean"] for s in expert_scalers], dtype=torch.float32))
        self.register_buffer("expert_std", torch.tensor([s["y_std"] for s in expert_scalers], dtype=torch.float32))
        self.register_buffer("feature_mean", torch.zeros(len(FEATURE_NAMES)))
        self.register_buffer("feature_scale", torch.ones(len(FEATURE_NAMES)))
        self.register_buffer("static_risks", torch.ones(2, 2)*risk_floor)
        self.risk_head = nn.Linear(len(FEATURE_NAMES), 4)
        nn.init.zeros_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)
        self.policy, self.risk_floor, self.supervised_quality = policy, risk_floor, True
        self.requires_grad_(False)

    def components(self, x, sqi, automatic):
        available = x.abs().amax(-1) > 1e-6
        if not available.any(1).all():
            raise ValueError("Both modalities are absent; abstain from BP estimation")
        predictions, logits = [], []
        for c, expert in enumerate(self.experts):
            z = expert.encoders[0](x[:, c:c+1])
            p = expert.regressor(z)*self.expert_std[c] + self.expert_mean[c]
            predictions.append(torch.where(available[:, c:c+1], p, self.expert_mean[c]))
            logits.append(expert.quality[0](z, sqi[:, c]).masked_fill(~available[:, c], -20.))
        expert_prediction, quality_logits = torch.stack(predictions, 1), torch.stack(logits, 1)
        goodness = quality_logits.sigmoid()
        features = torch.cat([sqi.flatten(1), automatic, goodness, expert_prediction.flatten(1), available.to(x.dtype)], dim=1)
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite waveform-derived reliability features")
        risks = self.risk_head((features-self.feature_mean)/self.feature_scale).reshape(-1, 2, 2).clamp(min=self.risk_floor)
        weights = fusion_weights(self.policy, goodness, risks, self.static_risks, available)
        return {"prediction": (weights*expert_prediction).sum(1), "goodness": goodness, "quality_logits": quality_logits,
                "weights": weights, "risk": risks, "expert_prediction": expert_prediction, "features": features,
                "available": available, "risk_proxy": (weights*risks).sum(1)}

    def forward(self, x, sqi, automatic):
        result = self.components(x, sqi, automatic)
        return result["prediction"], result["quality_logits"], result["weights"]


def predict_components(model, arrays, scalers, device="cpu", batch_size=64):
    sqi = scale_sqi(arrays["sqi"], scalers)
    collected = {}
    with torch.inference_mode():
        for start in range(0, len(arrays["x"]), batch_size):
            inputs = [torch.as_tensor(v[start:start+batch_size], dtype=torch.float32, device=device)
                      for v in [arrays["x"], sqi, arrays["automatic"]]]
            result = model.components(*inputs)
            for key, value in result.items():
                collected.setdefault(key, []).append(value.cpu().numpy())
    return {key: np.concatenate(value) for key, value in collected.items()}
