from dataclasses import dataclass
from typing import OrderedDict
import torch

@dataclass
class RPNOutput:
    anchor_offsets: torch.Tensor
    objectness_scores: torch.Tensor

@dataclass
class HeadOutput:
    classification: torch.Tensor
    boxes: torch.Tensor
    rois: torch.Tensor

@dataclass
class OrientedRCNNOutput:
    rpn_output: OrderedDict[str, RPNOutput]
    anchors: OrderedDict[str, torch.Tensor]
    backbone_output: OrderedDict[str, torch.Tensor]
    head_output: HeadOutput

@dataclass
class LossOutput:
    total_loss: torch.Tensor
    classification_loss: torch.Tensor
    regression_loss: torch.Tensor

    def detach(self):
        return LossOutput(
            self.total_loss.detach(), 
            self.classification_loss.detach(), 
            self.regression_loss.detach()
        )

    def clone(self):
        return LossOutput(
            self.total_loss.clone(), 
            self.classification_loss.clone(), 
            self.regression_loss.clone()
        )

    def __add__(self, o: 'LossOutput'):
        return LossOutput(
            total_loss=self.total_loss + o.total_loss,
            classification_loss=self.classification_loss + o.classification_loss,
            regression_loss=self.regression_loss + o.regression_loss
        )

    def __str__(self):
        return \
            "-----------------------------\n" \
            + f"total_loss: {self.total_loss}\n" \
            + f"regression_loss: {self.regression_loss}\n" \
            + f"classification_loss: {self.classification_loss}" \

@dataclass
class Annotation:
    boxes: list[torch.Tensor]
    classifications: list[torch.Tensor]
