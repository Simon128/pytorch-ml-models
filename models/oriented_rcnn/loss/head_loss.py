import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou

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

        batch_size = len(classification_targets)
        cls_losses = []
        regr_losses = []

        for b in range(batch_size):
            box_targets = encode(regression_targets[b], Encodings.VERTICES, Encodings.ORIENTED_CV2_FORMAT)
            hbb_box_pred = encode(box_pred[b], Encodings.ORIENTED_CV2_FORMAT, Encodings.HBB_CORNERS)
            hbb_targets = encode(regression_targets[b], Encodings.VERTICES, Encodings.HBB_CORNERS)

            iou = box_iou(hbb_box_pred, hbb_targets)
            sample_mask, positives_idx, _ = relevant_samples_mask(iou, self.n_samples)

            background_cls = class_pred.shape[0] - 1
            cls_target = torch.full([len(class_pred[b])], background_cls).to(class_pred.device)
            if len(positives_idx[0]) > 0:
                cls_target[positives_idx[0]] = classification_targets[b][positives_idx[1]]

            cls_loss = F.cross_entropy(class_pred[b][sample_mask], cls_target[sample_mask].to(torch.long), reduction='sum')
            cls_losses.append(cls_loss)

            if len(positives_idx[0]) <= 0:
                continue

            relevant_gt = box_targets[positives_idx[1]]
            relevant_pred = box_pred[b][positives_idx[0]]
            regr_loss = F.smooth_l1_loss(relevant_pred, relevant_gt, reduction='sum')
            regr_losses.append(regr_loss)

        classification_loss = torch.tensor(cls_losses).sum() / batch_size
        regression_loss = torch.tensor(regr_losses).sum() / batch_size

        return LossOutput(
            total_loss=classification_loss + regression_loss,
            classification_loss=classification_loss,
            regression_loss=regression_loss
        )
