from dataclasses import dataclass
from typing import OrderedDict
import torch
from torch.utils.tensorboard.writer import SummaryWriter

@dataclass
class RPNOutput:
    anchor_offsets: torch.Tensor
    objectness_scores: torch.Tensor

@dataclass
class HeadOutput:
    classification: list[torch.Tensor]
    boxes: torch.Tensor | list[torch.Tensor]
    rois: list[torch.Tensor]
    strides: list[torch.Tensor] | torch.Tensor

@dataclass
class OrientedRCNNOutput:
    rpn_output: OrderedDict[str, RPNOutput]
    anchors: OrderedDict[str, torch.Tensor]
    backbone_output: OrderedDict[str, torch.Tensor]
    head_output: HeadOutput

@dataclass
class LossOutput:
    total_loss: float | torch.Tensor
    classification_loss: float | torch.Tensor
    regression_loss: float | torch.Tensor

    def __add__(self, o: 'LossOutput'):
        return LossOutput(
            total_loss=self.total_loss + o.total_loss,
            classification_loss=self.classification_loss + o.classification_loss,
            regression_loss=self.regression_loss + o.regression_loss
        )

    def to_writer(self, writer: SummaryWriter, pre_tag: str, step: int):
        writer.add_scalar(f"{pre_tag}/total_loss", self.total_loss, step)
        writer.add_scalar(f"{pre_tag}/classification_loss", self.classification_loss, step)
        writer.add_scalar(f"{pre_tag}/regression_loss", self.regression_loss, step)

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
