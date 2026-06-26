import math
from typing import (
    List,
    Optional,
)

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from wedsak.registry import registry


@registry.loss.register("NCE")
class NormalizedCrossEntropy(_Loss):
    def __init__(self, num_classes, scale=1.0, weight: Optional[List[float]] = None):
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

        if weight is not None:
            self.weight = torch.Tensor(weight)
        else:
            self.weight = weight

        self.name = "NCE"
        self.scale = scale

    def get_params(self):
        return {
            "scale": self.scale,
            "num_classes": self.num_classes,
            "name": self.name,
        }

    def forward(self, pred, labels, **kwargs):
        if self.weight is not None:
            weight = self.weight.to(pred.device)
            pred = F.log_softmax(pred, dim=1) * weight
        else:
            pred = F.log_softmax(pred, dim=1)
        # label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()

        nce = -1 * torch.sum(labels * pred, dim=1) / (-pred.sum(dim=1))  # FIXME
        return self.scale * nce.mean()


@registry.loss.register("RCE")
class ReverseCrossEntropy(_Loss):
    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels, **kwargs):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        # label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()

        label_one_hot = torch.clamp(labels, min=1e-4, max=1.0)  # FIXME
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        return self.scale * rce.mean()


@registry.loss.register("NCEandRCE")
class NCEandRCE(_Loss):
    def __init__(self, alpha, beta, num_classes, weight: Optional[List[float]] = None):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(
            scale=alpha, num_classes=num_classes, weight=weight
        )
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)
        self.name = "NCEandRCE"
        self.alpha = alpha
        self.beta = beta

    def get_params(self):
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "num_classes": self.num_classes,
            "name": self.name,
        }

    def forward(self, pred, labels, **kwargs):
        return self.nce(pred, labels) + self.rce(pred, labels)


@registry.loss("NormalizedGeneralizedCrossEntropy")
class NormalizedGeneralizedCrossEntropy(_Loss):
    def __init__(self, num_classes, scale=1.0, q=0.7):
        super(NormalizedGeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels, **kwargs):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        # label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        numerators = 1.0 - torch.pow(torch.sum(labels * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        ngce = numerators / denominators
        return self.scale * ngce.mean()


@registry.loss("MeanAbsoluteError")
class MeanAbsoluteError(_Loss):
    def __init__(self, num_classes, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels, **kwargs):
        pred = F.softmax(pred, dim=1)
        # label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        mae = 1.0 - torch.sum(labels * pred, dim=1)
        return self.scale * mae.mean()


@registry.loss("NGCEandMAE")
class NGCEandMAE(_Loss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)
        self.name = "NGCEandMAE"
        self.alpha = alpha
        self.beta = beta
        self.q = q

    def get_params(self):
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "q": self.q,
            "num_classes": self.num_classes,
            "name": self.name,
        }

    def forward(self, pred, labels, **kwargs):
        return self.ngce(pred, labels) + self.mae(pred, labels)


@registry.loss("ANL_CE")
class ANL_CE(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        alpha: float,
        beta: float,
        delta: float,
        lamb: Optional[float] = None,
        min_prob: float = 1e-7,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.min_prob = min_prob
        self.lamb = lamb
        self.A = -torch.tensor(min_prob).log()

    def forward(self, pred, labels, model, **kwargs):
        loss_nce = self.nce(pred, labels)
        loss_nnce = self.nnce(pred, labels)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        if self.lamb is not None:
            entropy = self.entropy_reg(pred)
        else:
            entropy = 0.0
        loss = (
            self.alpha * loss_nce
            + self.beta * loss_nnce
            + self.delta * l1_norm
            + entropy
        )
        return loss

    def nce(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        if labels.shape[-1] > 1:
            label_one_hot = labels
        else:
            label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)

        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return nce.mean()

    def nnce(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=self.min_prob, max=1 - self.min_prob)
        pred = self.A + pred.log()  # - log(1e-7) - (- log(p(k|x)))
        if labels.shape[-1] > 1:
            label_one_hot = labels
        else:
            label_one_hot = F.one_hot(labels, self.num_classes).to(pred.device)
        nnce = 1 - (label_one_hot * pred).sum(dim=1) / pred.sum(dim=1)
        return nnce.mean()

    def entropy_reg(self, pred):
        prob = F.softmax(pred, dim=1).clamp(min=self.min_prob, max=1 - self.min_prob)
        prob_class = prob.sum(dim=0).view(-1) / prob.sum()
        prob_class = prob_class.clamp(min=self.min_prob, max=1 - self.min_prob)
        entropy = math.log(self.num_classes) + (prob_class * prob_class.log()).sum()
        return entropy
