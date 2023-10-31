import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import LossOutput

class HeadLoss(nn.Module):
    def __init__(
            self, 
        ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
            self, 
            positive_boxes: torch.Tensor,
            pred_cls: torch.Tensor,
            target_boxes: torch.Tensor,
            target_cls: torch.Tensor
        ):
        # classification loss
        pred_cls = torch.where(pred_cls == 0, 1e-7, pred_cls)
        cls_loss = self.ce(pred_cls, target_cls)            

        # regression loss
        if len(positive_boxes) == 0:
            regr_loss = 0
        else:
            regr_loss = F.smooth_l1_loss(positive_boxes, target_boxes, reduction='mean', beta=0.1111111111111)

        return LossOutput(
            total_loss=cls_loss + regr_loss, 
            classification_loss=cls_loss, 
            regression_loss=regr_loss
        )
