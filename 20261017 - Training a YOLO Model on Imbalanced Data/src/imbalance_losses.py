from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import DetectionModel as BaseDetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8DetectionLoss


# --------------------------------------------------------------------------- #
# REFERENCE ONLY -- Ultralytics' own focal loss.
#
# class FocalLoss(nn.Module):
#     """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""
#
#     def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
#         """Initialize FocalLoss class with focusing and balancing parameters."""
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = torch.tensor(alpha)
#
#     def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
#         """Calculate focal loss with modulating factors for class imbalance."""
#         loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
#         pred_prob = pred.sigmoid()  # prob from logits
#         p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
#         modulating_factor = (1.0 - p_t) ** self.gamma
#         loss *= modulating_factor
#         if (self.alpha > 0).any():
#             self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
#             alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
#             loss *= alpha_factor
#         return loss.mean(1).sum()
# --------------------------------------------------------------------------- #


class ElementwiseFocalLoss(nn.Module):
    """
    The maths is identical to Ultralytics' ``FocalLoss`` above; only the final
    reduction is dropped so the output keeps the ``(bs, num_anchors, nc)`` shape
    that ``v8DetectionLoss`` sums and broadcasts ``class_weights`` over.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # p_t = probability the model assigns to the TRUE class
        p = pred.sigmoid()
        p_t = target * p + (1 - target) * (1 - p)
        loss = bce * (1.0 - p_t) ** self.gamma

        if self.alpha > 0:
            loss = loss * (target * self.alpha + (1 - target) * (1 - self.alpha))
        return loss


class FocalDetectionLoss(v8DetectionLoss):
    def __init__(self, model: nn.Module, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__(model)
        self.bce = ElementwiseFocalLoss(gamma=gamma, alpha=alpha)


class FocalDetectionModel(BaseDetectionModel):
    def __init__(self, cfg=None, ch=3, nc=None, verbose=True, loss_kwargs=None):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        self._loss_kwargs = loss_kwargs or {}

    def init_criterion(self):
        return FocalDetectionLoss(self, **self._loss_kwargs)


def make_focal_trainer(gamma: float = 2.0, alpha: float = 0.25):
    from ultralytics.models.yolo.detect import DetectionTrainer

    loss_kwargs = dict(gamma=gamma, alpha=alpha)

    class FocalTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                FocalDetectionModel(
                    cfg, nc=self.data["nc"], ch=self.data["channels"],
                    verbose=verbose and RANK == -1, loss_kwargs=loss_kwargs,
                )
            )
            if weights:
                model.load(weights)
            model.args = self.args
            return model

    return FocalTrainer
