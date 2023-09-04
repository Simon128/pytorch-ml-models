import torch
import torch.nn as nn
from typing import OrderedDict

from .rpn_loss import RPNLoss
from .head_loss import HeadLoss
from ..data_formats import RPNOutput, Annotation

class OrientedRCNNLoss(nn.Module):
    def __init__(self, fpn_strides: list[int], n_samples = 256) -> None:
        super().__init__()
        self.rpn_loss = RPNLoss(fpn_strides, n_samples)
        self.head_loss = HeadLoss()

    def forward(
            self, 
            prediction: OrderedDict[str, RPNOutput], 
            anchors: OrderedDict[str, torch.Tensor],
            annotations: Annotation
        ):
        rpn_loss = self.rpn_loss(prediction, anchors, annotations)
        head_loss = self.head_loss()
        return rpn_loss, head_loss
