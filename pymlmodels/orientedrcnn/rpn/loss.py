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
            positive_objectness: torch.Tensor,
            negative_objectness: torch.Tensor,
            target_offsets: torch.Tensor
        ):
        device = positive_anchor_offsets.device
        n_pos = len(positive_objectness)
        n_neg = len(negative_objectness)

        # objectness loss
        cls_targets = torch.zeros(len(positive_objectness) + len(negative_objectness), device=device)
        cls_targets[:n_pos] = 1.0
        pred = torch.cat((positive_objectness, negative_objectness), dim=0)
        pred = torch.where(pred == 0, 1e-7, pred)
        cls_loss = self.bce(pred, cls_targets)

        # regression loss
        if n_pos == 0:
            regr_loss = 0
        else:
            regr_loss = F.smooth_l1_loss(positive_anchor_offsets, target_offsets, reduction='mean', beta=0.1111111111111)

        return LossOutput(
            total_loss=cls_loss + regr_loss, 
            classification_loss=cls_loss, 
            regression_loss=regr_loss
        )
