import torch
import torch.nn.functional as F
import torch.nn as nn

from ..utils import LossOutput

class RPNLoss(nn.Module):
    def __init__(
            self, 
        ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
            self, 
            positive_anchor_offsets: torch.Tensor,
            objectness: torch.Tensor,
            objectness_targets: torch.Tensor,
            target_offsets: torch.Tensor
        ):
        # objectness loss
        if len(objectness) == 0:
            cls_loss = 0.0
        else:
            pred = torch.where(objectness == 0, 1e-7, objectness)
            cls_loss = self.bce(pred, objectness_targets)

        # regression loss
        if len(positive_anchor_offsets) == 0:
            regr_loss = 0.0
        else:
            regr_loss = F.smooth_l1_loss(positive_anchor_offsets, target_offsets, reduction='mean', beta=0.1111111111111)

        return LossOutput(
            total_loss=cls_loss + regr_loss, 
            classification_loss=cls_loss, 
            regression_loss=regr_loss
        )
