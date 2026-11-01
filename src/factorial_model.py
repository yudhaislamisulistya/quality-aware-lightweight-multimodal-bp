"""Matched quality-gating controls and single-sensor controls for the factorial study."""
import torch
from torch import nn

from .optimized_model import RobustBPModel

CELLS = {
    "A_uniform_clean": (False, False),
    "B_gated_clean": (True, False),
    "C_uniform_augmented": (False, True),
    "D_gated_augmented": (True, True),
}
UNIMODAL = {"ecg_only_clean": 0, "ppg_only_clean": 1}
STUDY_MODELS = list(CELLS) + list(UNIMODAL) + ["dense_uniform_clean"]


class FactorialBPModel(RobustBPModel):
    def __init__(self, name):
        if name not in STUDY_MODELS:
            raise ValueError(name)
        gated = CELLS.get(name, (False, False))[0]
        super().__init__("optimized_quality" if gated else "optimized_no_gate")
        self.study_name = name
        self.modalities = [UNIMODAL[name]] if name in UNIMODAL else [0, 1]
        if name == "dense_uniform_clean":
            for encoder in self.encoders:
                for block in encoder.temporal[4:]:
                    conv = block.main[0]
                    block.main[0] = nn.Conv1d(conv.in_channels, conv.out_channels, conv.kernel_size,
                                              stride=conv.stride, padding=conv.padding, dilation=conv.dilation, bias=False)
        if name in UNIMODAL:
            c = self.modalities[0]
            self.encoders = nn.ModuleList([self.encoders[c]])
            self.quality = nn.ModuleList([self.quality[c]])
            self.regressor = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Dropout(.15), nn.Linear(32, 2))

    def forward(self, x, sqi, automatic):
        if len(self.modalities) == 2:
            return super().forward(x, sqi, automatic)
        c = self.modalities[0]
        if not (x[:, c].abs().amax(-1) > 1e-6).all():
            raise ValueError("Required unimodal sensor is absent; abstain")
        feature = self.encoders[0](x[:, c:c+1])
        logits = x.new_full((len(x), 2), -20.)
        logits[:, c] = self.quality[0](feature.detach(), sqi[:, c])
        weights = x.new_zeros((len(x), 2))
        weights[:, c] = 1.
        return self.regressor(feature), logits, weights
