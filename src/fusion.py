import torch
from torch import nn


class QualityEstimator(nn.Module):
    def __init__(self, features, indicators):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(features + indicators, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, feature, sqi):
        return self.layers(torch.cat([feature, sqi], dim=1)).squeeze(1)


def quality_weights(goodness):
    goodness = goodness.clamp(min=0.05)
    return goodness / goodness.sum(dim=1, keepdim=True)
