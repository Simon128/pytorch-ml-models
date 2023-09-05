from typing import OrderedDict
import torch
from torchvision.ops import box_iou
import torch.nn.functional as F
import torch.nn as nn

from .utils import relevant_samples_mask
from ..data_formats import LossOutput, RPNOutput, Annotation
from ..encoder import encode, Encodings

class RPNLoss(nn.Module):
    def __init__(self, fpn_strides: list[int], n_samples = 256) -> None:
        super().__init__()
        self.fpn_strides = fpn_strides
        self.n_samples = n_samples

    def forward(
            self, 
            prediction: OrderedDict[str, RPNOutput], 
            anchors: OrderedDict[str, torch.Tensor],
            annotations: Annotation
        ):
        aggregated_loss = None

        for (key, rpn_out), stride in zip(prediction.items(), self.fpn_strides):
            loss = self.forward_single(
                rpn_out, 
                anchors[key], 
                annotations,
                stride
            )
            if aggregated_loss is None:
                aggregated_loss = loss
            else:
                aggregated_loss.classification_loss += loss.classification_loss
                aggregated_loss.regression_loss += loss.regression_loss
                aggregated_loss.total_loss += loss.total_loss

        return aggregated_loss

    def rpn_anchor_iou(self, anchors: torch.Tensor, target_boxes: torch.Tensor):
        hbb_anchors = encode(anchors, Encodings.HBB_CENTERED, Encodings.HBB_CORNERS)
        hbb_target_boxes = encode(target_boxes, Encodings.VERTICES, Encodings.HBB_CORNERS)
        return box_iou(hbb_anchors, hbb_target_boxes)

    def forward_single(
            self, 
            prediction: RPNOutput, 
            anchors: torch.Tensor, 
            annotation: Annotation,
            stride: float = 1.0
        ):
        anchor_offsets = prediction.anchor_offsets 
        objectness_scores = prediction.objectness_scores
        anchors = anchors
        targets = annotation.boxes

        cls_losses = []
        regr_losses = []

        for i in range(len(targets)):
            iou = self.rpn_anchor_iou(anchors[i], targets[i])
            sample_mask, positives_idx, _ = relevant_samples_mask(iou, self.n_samples)

            # objectness loss
            cls_target = torch.zeros_like(objectness_scores[i])
            cls_target[positives_idx[0]] = 1.0

            cls_loss = F.binary_cross_entropy_with_logits(
                objectness_scores[i][sample_mask], 
                cls_target[sample_mask], 
                reduction='sum'
            )
            cls_losses.append(cls_loss)

            # regression loss
            if len(positives_idx[0]) <= 0:
                continue
            relevant_gt = targets[i][positives_idx[1]]
            relevant_pred = anchor_offsets[i][positives_idx[0]]
            relevant_anchor = anchors[i][positives_idx[0]]
            relevant_targets = encode(relevant_gt / stride, Encodings.VERTICES, Encodings.ANCHOR_OFFSET, relevant_anchor)
            regr_loss = F.smooth_l1_loss(relevant_pred, relevant_targets, reduction='sum')
            regr_losses.append(regr_loss)

        regression_loss = torch.mean(torch.stack(regr_losses)) if len(regr_losses) > 0 else torch.tensor(0).to(anchors.device)
        classification_loss = torch.mean(torch.stack(cls_losses))
        loss = classification_loss + regression_loss 
        return LossOutput(
            total_loss=loss, 
            classification_loss=classification_loss, 
            regression_loss=regression_loss
        )
