import torch
from torch import nn

from .fusion import QualityEstimator, quality_weights
from .signal_quality import SQI_NAMES

MODEL_NAMES = ["ecg_only", "ppg_only", "standard_fusion", "quality_fusion",
               "proposed_no_gate", "proposed", "proposed_dense", "cnn_bigru"]


def block(in_channels, out_channels, kernel, stride, separable):
    if separable:
        layers = [nn.Conv1d(in_channels, in_channels, kernel, stride, kernel//2, groups=in_channels, bias=False),
                  nn.Conv1d(in_channels, out_channels, 1, bias=False)]
    else:
        layers = [nn.Conv1d(in_channels, out_channels, kernel, stride, kernel//2, bias=False)]
    return nn.Sequential(*layers, nn.BatchNorm1d(out_channels), nn.ReLU())


class Encoder(nn.Module):
    def __init__(self, separable=True):
        super().__init__()
        self.layers = nn.Sequential(block(1, 8, 15, 5, False), block(8, 16, 9, 3, separable),
                                    block(16, 32, 7, 2, separable), nn.AdaptiveAvgPool1d(8),
                                    nn.Flatten(), nn.Linear(32*8, 32), nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class BPModel(nn.Module):
    """Inference accepts only waveforms, waveform SQIs, and automatic goodness."""
    def __init__(self, name):
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {name}")
        self.name = name
        self.supervised_quality = name in {"proposed_no_gate", "proposed", "proposed_dense"}
        if name == "cnn_bigru":
            self.temporal = nn.Sequential(block(2, 16, 15, 5, False), block(16, 32, 9, 3, False), block(32, 64, 7, 2, False))
            self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
            self.regressor = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 2))
        else:
            modalities = [0] if name == "ecg_only" else ([1] if name == "ppg_only" else [0, 1])
            self.modalities = modalities
            self.encoders = nn.ModuleList([Encoder(separable=name != "proposed_dense") for _ in modalities])
            if self.supervised_quality:
                self.quality = nn.ModuleList([QualityEstimator(32, len(SQI_NAMES)) for _ in modalities])
            self.regressor = nn.Sequential(nn.Linear(32 * len(modalities), 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 2))

    def forward(self, x, sqi, automatic):
        batch = x.shape[0]
        logits = x.new_zeros((batch, 2))
        weights = x.new_full((batch, 2), 0.5)
        if self.name == "cnn_bigru":
            z = self.temporal(x).transpose(1, 2)
            _, h = self.gru(z)
            return self.regressor(torch.cat([h[-2], h[-1]], dim=1)), logits, weights
        features = [encoder(x[:, c:c+1]) for encoder, c in zip(self.encoders, self.modalities, strict=True)]
        if self.supervised_quality:
            logits = torch.stack([quality(feature, sqi[:, c]) for quality, feature, c in zip(self.quality, features, self.modalities, strict=True)], dim=1)
            if self.name != "proposed_no_gate":
                weights = quality_weights(logits.sigmoid())
        elif self.name == "quality_fusion":
            weights = quality_weights(automatic)
        if len(features) == 2:
            # Factor two keeps the equal-weight fusion scale identical to concatenation.
            features = [z * (2 * weights[:, c:c+1]) for c, z in enumerate(features)]
        else:
            weights = x.new_zeros((batch, 2))
            weights[:, self.modalities[0]] = 1
        return self.regressor(torch.cat(features, dim=1)), logits, weights
