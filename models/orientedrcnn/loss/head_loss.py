import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..ops import pairwise_iou_rotated
from ..utils import encode, Encodings, HeadOutput, Annotation, LossOutput
from .sampler import BalancedSampler

class HeadLoss(nn.Module):
    def __init__(
            self, 
            n_samples: int = 256,
            pos_fraction: float = 0.5,
            neg_thr: float = 0.5,
            pos_thr: float = 0.5,
            sample_max_inbetween_as_pos: bool = False
        ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.ce = nn.CrossEntropyLoss()
        self.sampler = BalancedSampler(
            n_samples=n_samples,
            pos_fraction=pos_fraction,
            neg_thr=neg_thr,
            pos_thr=pos_thr,
            sample_max_inbetween_as_pos=sample_max_inbetween_as_pos
        )

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
            box_targets[..., -1] = -1 * box_targets[..., -1] * 180 / math.pi
            # try to decode here to real targets maybe???
            # or try midpoints offset as targets????
            iou = pairwise_iou_rotated(rois[b], box_targets)
            pos_indices, neg_indices = self.sampler(iou)
            background_cls = class_pred[b].shape[-1] - 1

            # classification loss
            cls_targets = torch.full((len(pos_indices[0]) + len(neg_indices[0]),), background_cls, device=iou.device)
            cls_targets[:len(pos_indices[0])] = classification_targets[b][pos_indices[1]]
            pred = torch.cat((class_pred[b][pos_indices[0]], class_pred[b][neg_indices[0]]), dim=0)
            pred = torch.where(pred == 0, 1e-7, pred)
            cls_loss = self.ce(pred, cls_targets.to(torch.long))
            cls_losses.append(cls_loss.reshape(1))

            if len(pos_indices[0]) <= 0:
                continue

            relevant_gt = box_targets[pos_indices[1]]
            relevant_pred = box_pred[b][pos_indices[0]]
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
