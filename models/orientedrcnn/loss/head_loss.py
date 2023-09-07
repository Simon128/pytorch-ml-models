import torch
import torch.nn as nn
import torch.nn.functional as F
from ..rotated_iou import pairwise_iou_rotated

from ..encoder import encode, Encodings
from .utils import relevant_samples_mask
from ..data_formats import HeadOutput, Annotation, LossOutput

class HeadLoss(nn.Module):
    def __init__(self, n_samples: int = 256) -> None:
        super().__init__()
        self.n_samples = n_samples

    def forward(
            self,
            predictions: HeadOutput,
            annotations: Annotation
        ):
        classification_targets = annotations.classifications
        regression_targets = annotations.boxes
        box_pred = predictions.boxes
        class_pred = predictions.classification
        rois = predictions.rois

        batch_size = len(classification_targets)
        cls_losses = []
        regr_losses = []

        for b in range(batch_size):
            box_targets = encode(regression_targets[b], Encodings.VERTICES, Encodings.ORIENTED_CV2_FORMAT)
            #hbb_pred = encode(rois[b], Encodings.ORIENTED_CV2_FORMAT, Encodings.HBB_CORNERS)
            #hbb_targets = encode(regression_targets[b], Encodings.VERTICES, Encodings.HBB_CORNERS)

            iou = pairwise_iou_rotated(rois[b], box_targets)
            sample_mask, positives_idx, _ = relevant_samples_mask(iou, self.n_samples)

            background_cls = class_pred[b].shape[-1] - 1
            cls_target = torch.full([len(class_pred[b])], background_cls).to(class_pred[b].device)
            if len(positives_idx[0]) > 0:
                cls_target[positives_idx[0]] = classification_targets[b][positives_idx[1]]

            cls_loss = F.cross_entropy(class_pred[b][sample_mask], cls_target[sample_mask].to(torch.long), reduction='mean')
            cls_losses.append(cls_loss.reshape(1))

            if len(positives_idx[0]) <= 0:
                continue

            relevant_gt = box_targets[positives_idx[1]]
            relevant_pred = box_pred[b][positives_idx[0]]
            regr_loss = F.smooth_l1_loss(relevant_pred, relevant_gt, reduction='mean')
            regr_losses.append(regr_loss.reshape(1))

        if len(cls_losses) > 0:
            classification_loss = torch.mean(torch.cat(cls_losses))
        else:
            classification_loss = 0.0
        if len(regr_losses) > 0:
            regression_loss = torch.mean(torch.cat(regr_losses))
        else:
            regression_loss = 0.0

        if isinstance(classification_loss, torch.Tensor) and torch.isnan(classification_loss).any():
            classification_loss = 0.0
        if isinstance(regression_loss, torch.Tensor) and torch.isnan(regression_loss).any():
            regression_loss = 0.0

        return LossOutput(
            total_loss=classification_loss + regression_loss,
            classification_loss=classification_loss,
            regression_loss=regression_loss
        )
