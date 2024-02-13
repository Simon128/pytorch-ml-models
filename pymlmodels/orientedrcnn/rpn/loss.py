import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import box_iou

from ..utils import LossOutput, BalancedSampler, encode, Encodings

class RPNLoss(nn.Module):
    def __init__(
            self, 
            sampler: BalancedSampler
        ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.sampler = sampler

    def rpn_anchor_iou(self, anchors: torch.Tensor, target_boxes: torch.Tensor):
        hbb_anchors = encode(anchors, Encodings.HBB_CENTERED, Encodings.HBB_CORNERS)
        hbb_target_boxes = encode(target_boxes, Encodings.VERTICES, Encodings.HBB_CORNERS)
        return box_iou(hbb_anchors, hbb_target_boxes)

    def forward(
            self, 
            anchors: torch.Tensor,
            target_boxes: torch.Tensor,
            anchor_offsets_predictions: torch.Tensor,
            objectness_predictions: torch.Tensor
        ):
        hbb_anchors = encode(anchors, Encodings.HBB_CENTERED, Encodings.HBB_CORNERS)

        cls_loss = 0.0
        regr_loss = 0.0

        for b in range(anchors.shape[0]):
            hbb_target_boxes = encode(target_boxes[b], Encodings.VERTICES, Encodings.HBB_CORNERS)
            iou = box_iou(hbb_anchors[b], hbb_target_boxes)
            positives_idx, negative_idx = self.sampler(iou)
            n_pos = len(positives_idx[0])
            n_neg = len(negative_idx)
            all_pred_idx = torch.cat((positives_idx[0], negative_idx))
            sampled_obj_pred = objectness_predictions[b][all_pred_idx]
            sampled_obj_pred = torch.where(sampled_obj_pred == 0, 1e-7, sampled_obj_pred)
            target_objectness = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)]).to(iou.device)
            cls_loss = cls_loss + self.bce(sampled_obj_pred, target_objectness)

            if n_pos > 0:
                sampled_anchor_offsets = anchor_offsets_predictions[b][positives_idx[0]]
                target_offsets = encode(
                    target_boxes[b][positives_idx[1]], 
                    Encodings.VERTICES, Encodings.ANCHOR_OFFSET, 
                    anchors[b][positives_idx[0]]
                )
                regr_loss = regr_loss + F.smooth_l1_loss(
                    sampled_anchor_offsets, 
                    target_offsets,
                    reduction="mean",
                    beta=0.1111111111111
                )

        return LossOutput(
            total_loss=cls_loss + regr_loss, 
            classification_loss=cls_loss, 
            regression_loss=regr_loss
        )
