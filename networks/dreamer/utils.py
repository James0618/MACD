import torch
import torch.nn as nn
from torch.nn import functional as F


def build_model(in_dim, out_dim, layers, hidden, activation, normalize=lambda x: x, norm=nn.Identity,
                if_last_norm=True):
    model = [normalize(nn.Linear(in_dim, hidden))]
    model += [norm(hidden)]
    model += [activation()]
    for i in range(layers - 1):
        model += [normalize(nn.Linear(hidden, hidden))]
        model += [norm(hidden)]
        model += [activation()]
    model += [normalize(nn.Linear(hidden, out_dim))]
    if if_last_norm:
        model += [norm(out_dim)]

    return nn.Sequential(*model)


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
        discrete_number=255
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=discrete_number).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = (torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1).to(x.device)
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        ).to(x.device)
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, torch.ones(1).to(x.device), torch.abs(self.buckets[below] - x)).to(x.device)
        dist_to_above = torch.where(equal, torch.ones(1).to(x.device), torch.abs(self.buckets[above] - x)).to(x.device)
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)

        return (target * log_pred).sum(-1)


class RewardEMA(object):
    """running mean and std"""
    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]

        return offset.detach(), scale.detach()
