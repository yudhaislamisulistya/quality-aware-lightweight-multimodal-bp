"""Small residual encoders and independently supervised waveform quality gates."""
import torch
from torch import nn

from .fusion import QualityEstimator
from .signal_quality import SQI_NAMES

OPTIMIZED_MODELS = ["optimized_standard", "optimized_no_gate", "optimized_quality_no_rank", "optimized_quality"]


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, kernel, dilation=1):
        super().__init__()
        self.main = nn.Sequential(nn.Conv1d(cin, cin, kernel, stride=2, padding=dilation*(kernel//2), dilation=dilation, groups=cin, bias=False),
                                  nn.Conv1d(cin, cout, 1, bias=False), nn.GroupNorm(4, cout), nn.SiLU())
        self.skip = nn.Conv1d(cin, cout, 1, stride=2, bias=False)

    def forward(self, x):
        return self.main(x) + self.skip(x)


class ResidualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.Sequential(nn.Conv1d(1, 8, 15, padding=7, bias=False), nn.GroupNorm(2, 8), nn.SiLU(),
                                      nn.AvgPool1d(3), ResidualBlock(8, 16, 9), ResidualBlock(16, 32, 7), ResidualBlock(32, 32, 5, dilation=2))
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.embedding = nn.Sequential(nn.Linear(32*11, 32), nn.LayerNorm(32), nn.SiLU())

    def forward(self, x):
        z = self.temporal(x)
        statistics = torch.cat([self.pool(z).flatten(1), z.mean(-1), z.amax(-1), z.std(-1, correction=0)], dim=1)
        return self.embedding(statistics)


class RobustBPModel(nn.Module):
    def __init__(self, name="optimized_quality"):
        super().__init__()
        if name not in OPTIMIZED_MODELS:
            raise ValueError(name)
        self.name = name
        self.supervised_quality = name != "optimized_standard"
        self.encoders = nn.ModuleList([ResidualEncoder(), ResidualEncoder()])
        if self.supervised_quality:
            self.quality = nn.ModuleList([QualityEstimator(32, len(SQI_NAMES)), QualityEstimator(32, len(SQI_NAMES))])
        self.regressor = nn.Sequential(nn.Linear(128, 32), nn.SiLU(), nn.Dropout(.15), nn.Linear(32, 2))

    def forward(self, x, sqi, automatic):
        available = x.abs().amax(-1) > 1e-6
        if not available.any(1).all():
            raise ValueError("Both modalities are absent; abstain from BP estimation")
        features = [encoder(x[:, c:c+1]) for c, encoder in enumerate(self.encoders)]
        logits = x.new_zeros((len(x), 2))
        if self.supervised_quality:
            # BP loss cannot reward a quality score that contradicts its independent supervision.
            logits = torch.stack([q(z.detach(), sqi[:, c]) for c, (q, z) in enumerate(zip(self.quality, features, strict=True))], dim=1)
            logits = logits.masked_fill(~available, -20.)
        if self.name in {"optimized_standard", "optimized_no_gate"}:
            goodness = available.to(x.dtype)
        else:
            goodness = logits.sigmoid().detach().clamp(min=.05) * available
        weights = goodness / goodness.sum(1, keepdim=True)
        ecg, ppg = [z * (2 * weights[:, c:c+1]) for c, z in enumerate(features)]
        fused = torch.cat([ecg, ppg, ecg * ppg, (ecg - ppg).abs()], dim=1)
        return self.regressor(fused), logits, weights
