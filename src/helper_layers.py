import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LeakyHardSigmoid(nn.Module):
    def __init__(self, slope: float = 0.01, **kwargs):
        super().__init__()
        self.slope = slope

    def forward(self, x):
        return F.leaky_relu(1.0 - F.leaky_relu(1 - x, self.slope), self.slope)


class CriterionLayerSpread(nn.Module):
    def __init__(
        self,
        criteria_nr,
        hidden_nr,
        input_range=(0, 1),
        criterion_layer_spread_normalize_bias=False,
        **kwargs
    ):
        input_range = (-input_range[0], -input_range[1])
        self.max_bias = max(input_range)
        self.min_bias = min(input_range)
        super().__init__()
        self.normalize_bias = criterion_layer_spread_normalize_bias
        self.bias = nn.Parameter(torch.FloatTensor(hidden_nr, criteria_nr))
        self.weight = nn.Parameter(torch.FloatTensor(hidden_nr, criteria_nr))
        self.reset_parameters()

    def b(self):
        if self.normalize_bias:
            return torch.clamp(self.bias, self.min_bias, self.max_bias)
        else:
            return self.bias

    def reset_parameters(self):
        nn.init.uniform_(self.weight, 1, 10.0)
        nn.init.uniform_(self.bias, self.min_bias, self.max_bias)

    def w(self):
        return torch.clamp(self.weight, 0.0)

    def forward(self, x):
        return (x + self.b()) * self.w()


class CriterionLayerCombine(nn.Module):
    def __init__(
        self, criteria_nr, hidden_nr, criterionLayerCombine_min_weight=0.0, **kwargs
    ):
        super().__init__()
        self.criterionLayerCombine_min_weight = criterionLayerCombine_min_weight
        self.weight = nn.Parameter(torch.FloatTensor(hidden_nr, criteria_nr))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, 0.2, 1.0)
        self.weight.data = self.weight.data / torch.sum(self.weight.data)

    def w(self):
        return torch.clamp(self.weight, self.criterionLayerCombine_min_weight)

    def forward(self, x):
        return (x * self.w()).sum(1)


class SumLayer(nn.Linear):
    def __init__(self, criteria_nr, sum_layer_weight_std=0, **kwargs):
        super().__init__(criteria_nr, 1, bias=False)
        self.weight = nn.Parameter(
            (torch.zeros(1, criteria_nr).uniform_(-1, 1) * sum_layer_weight_std + 1)
            / criteria_nr
        )

    def w(self):
        return torch.clamp(self.weight, 0.01)

    def forward(self, x):
        return (x * self.w()).sum(1).view(-1, 1)


class ThresholdLayer(nn.Module):
    def __init__(self, threshold=None, requires_grad=True):
        super().__init__()
        if threshold is None:
            self.threshold = nn.Parameter(
                torch.FloatTensor(1).uniform_(0.1, 0.9),
                requires_grad=requires_grad)
        else:
            self.treshold = nn.Parameter(
                torch.FloatTensor([threshold]), requires_grad=requires_grad
            )

    def forward(self, x):
        return x - self.threshold


class NormLayer(nn.Module):
    def __init__(self, method, criteria_nr, ideal_alternative: np.ndarray,
                 antiideal_alternative: np.ndarray):
        super().__init__()
        self.method = method
        self.criteria_nr = criteria_nr
        self.thresholdLayer = ThresholdLayer()
        self.ideal_alternative = torch.FloatTensor(ideal_alternative) \
            .view(1, 1, self.criteria_nr)
        self.antiideal_alternative = torch.FloatTensor(antiideal_alternative) \
            .view(1, 1, self.criteria_nr)

    def forward(self, x, *args):
        self.out = self.method(x)
        self.best = self.ideal_alternative.to(self.out.device)
        self.worst = self.antiideal_alternative.to(self.out.device)
        self.best = self.method(self.best)
        self.worst = self.method(self.worst)
        self.out = (self.out - self.worst) / (self.best - self.worst)
        return self.thresholdLayer(self.out)
