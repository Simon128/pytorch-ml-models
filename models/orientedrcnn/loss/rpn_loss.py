from typing import OrderedDict
import torch
from torchvision.ops import box_iou
import torch.nn.functional as F
import torch.nn as nn

from ..utils import LossOutput, RPNOutput, Annotation, encode, Encodings
from .sampler import BalancedSampler

class RPNLoss(nn.Module):
    def __init__(
            self, 
            fpn_strides: list[int], 
            n_samples = 256,
            pos_fraction: float = 0.5, 
            neg_thr: float = 0.3,
            pos_thr: float = 0.7,
            sample_max_inbetween_as_pos: bool = True
        ) -> None:
        super().__init__()
        self.fpn_strides = fpn_strides
        self.n_samples = n_samples
        self.bce = nn.BCEWithLogitsLoss()
        self.sampler = BalancedSampler(
            n_samples=n_samples,
            pos_fraction=pos_fraction,
            neg_thr=neg_thr,
            pos_thr=pos_thr,
            sample_max_inbetween_as_pos=sample_max_inbetween_as_pos
        )

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
                aggregated_loss.classification_loss = aggregated_loss.classification_loss + loss.classification_loss
                aggregated_loss.regression_loss = aggregated_loss.regression_loss + loss.regression_loss 
                aggregated_loss.total_loss = aggregated_loss.total_loss + loss.total_loss

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
            iou = self.rpn_anchor_iou(anchors[i] * stride, targets[i])
            pos_indices, neg_indices = self.sampler(iou)
            # debug
            #import cv2
            #import numpy as np
            #image = cv2.imread("/home/simon/unibw/pytorch-ml-models/models/orientedrcnn/example/image.tif") # type:ignore
            #vert_anchors = encode(anchors[i], Encodings.HBB_CENTERED, Encodings.HBB_VERTICES) * stride
            #pos_anchors = vert_anchors[pos_indices[0]]
            #neg_anchors = vert_anchors[neg_indices[0]]
            #pos_img = image.copy()
            #neg_img = image.copy()
            #for a in pos_anchors:
            #    anchor_pts = a.cpu().detach().numpy().astype(np.int32)
            #    pos_img = cv2.polylines(pos_img, [anchor_pts], True, (0, 255, 0), 2) # type: ignore
            #for a in neg_anchors:
            #    anchor_pts = a.cpu().detach().numpy().astype(np.int32)
            #    neg_img = cv2.polylines(neg_img, [anchor_pts], True, (0, 255, 0), 2) # type: ignore
            #cv2.imwrite(f"pos_anchors_{stride}.png", pos_img) # type: ignore
            #cv2.imwrite(f"neg_anchors_{stride}.png", neg_img) # type: ignore

            # objectness loss
            cls_targets = torch.zeros(len(pos_indices[0]) + len(neg_indices[0]), device=iou.device)
            cls_targets[:len(pos_indices[0])] = 1.0
            pred = torch.cat((objectness_scores[i][pos_indices[0]], objectness_scores[i][neg_indices[0]]), dim=0)
            pred = torch.where(pred == 0, 1e-7, pred)
            cls_loss = self.bce(pred, cls_targets)
            cls_losses.append(cls_loss)

            # regression loss
            if len(pos_indices[0]) == 0:
                continue

            relevant_gt = targets[i][pos_indices[1]]
            relevant_pred = anchor_offsets[i][pos_indices[0]]
            relevant_anchor = anchors[i][pos_indices[0]]
            relevant_targets = encode(relevant_gt / stride, Encodings.VERTICES, Encodings.ANCHOR_OFFSET, relevant_anchor)
            regr_loss = F.smooth_l1_loss(relevant_pred, relevant_targets, reduction='mean', beta=0.1111111111111)
            regr_losses.append(regr_loss)

        regression_loss = torch.mean(torch.stack(regr_losses)) if len(regr_losses) > 0 else torch.tensor(0).to(anchors.device)
        classification_loss = torch.mean(torch.stack(cls_losses))
        loss = classification_loss + regression_loss 
        return LossOutput(
            total_loss=loss, 
            classification_loss=classification_loss, 
            regression_loss=regression_loss
        )
