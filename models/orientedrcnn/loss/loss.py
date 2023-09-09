import torch.nn as nn

from .rpn_loss import RPNLoss
from .head_loss import HeadLoss
from ..utils import OrientedRCNNOutput, Annotation

class OrientedRCNNLoss(nn.Module):
    def __init__(self, fpn_strides: list[int], n_samples = 256) -> None:
        super().__init__()
        self.rpn_loss = RPNLoss(fpn_strides, n_samples)
        self.head_loss = HeadLoss(n_samples)

    def forward(
            self, 
            prediction: OrientedRCNNOutput, 
            annotations: Annotation
        ):
        rpn_loss = self.rpn_loss(prediction.rpn_output, prediction.anchors, annotations)
        head_loss = self.head_loss(prediction.head_output, annotations)
        return rpn_loss, head_loss
