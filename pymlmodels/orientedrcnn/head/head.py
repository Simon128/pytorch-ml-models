import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple
import numpy as np

from ..utils import HeadOutput, normalize, encode, Encodings, RPNOutput, Annotation
from .roi_align_rotated import RoIAlignRotatedWrapper
from ..ops import nms_rotated
from ..ops import pairwise_iou_rotated
from ..utils.sampler import BalancedSampler
from .loss import HeadLoss

class OrientedRCNNHead(nn.Module):
    def __init__(
            self, 
            in_channels: int = 256, 
            fpn_strides: list = [4, 8, 16, 32, 64],
            out_channels: int = 1024,
            num_classes: int = 10,
            roi_align_size: Tuple[int, int] = (7,7),
            roi_align_sampling_ratio: int = 2,
            inject_annotation: bool = False,
            sampler: BalancedSampler = BalancedSampler(
                n_samples=512,
                pos_fraction=0.25,
                neg_thr=0.5,
                pos_thr=0.5,
                sample_max_inbetween_as_pos=False
            ),
            loss: HeadLoss = HeadLoss()
        ):
        super().__init__()
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.inject_annotation = inject_annotation
        self.sampler = sampler
        self.loss = loss

        self.roi_align_rotated = RoIAlignRotatedWrapper(
            roi_align_size, 
            spatial_scale = 1, 
            sampling_ratio = roi_align_sampling_ratio,
            fpn_strides=self.fpn_strides[:-1]
        )
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(in_channels * roi_align_size[0] * roi_align_size[1], out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )
        # +1 for background class
        self.classification = nn.Linear(out_channels, self.num_classes + 1)
        # note: we predict x, y, w, h, theta instead of the midpoint offset thingy
        # as shown in figure 2 of the paper
        self.regression = nn.Linear(out_channels, 5)

    def sample(
            self, 
            proposals: torch.Tensor, 
            objectness_scores: torch.Tensor,
            ground_truth_boxes: list[torch.Tensor],
            ground_truth_cls: list[torch.Tensor],
            stride: float
        ):
        device = proposals[0].device
        sampled_proposals = []
        sampled_objectness = []
        pos_masks = []
        sampled_gt_boxes = []
        sampled_gt_cls = []

        for b in range(len(proposals)):
            if self.inject_annotation:
                rois = torch.cat([proposals[b], ground_truth_boxes[b] / stride])
                rois_obj = torch.cat([objectness_scores[b], torch.ones((len(ground_truth_boxes[b],))).float().to(device)])
            else:
                rois = proposals[b]
                rois_obj = objectness_scores[b]

            box_targets = encode(ground_truth_boxes[b] / stride, Encodings.VERTICES, Encodings.THETA_FORMAT_TL_RT)
            iou_rois = encode(rois, Encodings.VERTICES, Encodings.THETA_FORMAT_TL_RT)
            iou = pairwise_iou_rotated(iou_rois, box_targets)
            pos_indices, neg_indices = self.sampler(iou)
            n_pos = len(pos_indices[0])
            n_neg = len(neg_indices[0])
            background_cls = self.num_classes 
            rand_permutation = torch.randperm(n_pos + n_neg, device=iou.device)
            pos_mask = torch.zeros((n_pos + n_neg,), device=rois.device)
            pos_mask[:n_pos] = 1
            pos_mask = pos_mask[rand_permutation].to(torch.bool)
            pos_masks.append(pos_mask)
            sampled_proposals.append(torch.cat((rois[pos_indices[0]], rois[neg_indices[0]]))[rand_permutation])
            sampled_objectness.append(torch.cat((rois_obj[pos_indices[0]], rois_obj[neg_indices[0]]))[rand_permutation])
            # we just fill the samples gt boxes up, later the correct ones will be selected via pos_masks
            sampled_gt_boxes.append(torch.cat((ground_truth_boxes[b][pos_indices[1]], ground_truth_boxes[b][neg_indices[1]]))[rand_permutation])
            sampled_gt_cls.append(torch.cat((ground_truth_cls[b][pos_indices[1]], torch.full((n_neg,), background_cls).to(device)))[rand_permutation])

        return torch.stack(sampled_proposals), torch.stack(sampled_objectness), sampled_gt_boxes, sampled_gt_cls, pos_masks

    def flatten_levels(
            self, 
            aligned_feat: OrderedDict, 
            proposals: OrderedDict, 
            objectness: OrderedDict,
            gt_boxes: OrderedDict | None = None,
            gt_cls: OrderedDict | None = None,
            pos_masks: OrderedDict | None = None
        ):
        num_batches = list(aligned_feat.values())[0].shape[0]
        flat_features = []
        flat_boxes = []
        flat_strides = []
        flat_scores = []
        flat_gt_boxes = []
        flat_gt_cls = []
        flat_pos_masks = []

        for b in range(num_batches):
            merged_scores = []
            merged_features = []
            merged_proposals = []
            merged_strides = []
            merged_gt_boxes = []
            merged_gt_cls = []
            merged_pos_masks = []

            for s_idx, k in enumerate(aligned_feat.keys()):
                merged_features.append(aligned_feat[k][b])
                merged_scores.append(objectness[k][b])
                merged_proposals.append(proposals[k][b])
                merged_strides.append(torch.full_like(objectness[k][b], self.fpn_strides[s_idx]))
                if gt_boxes is not None:
                    merged_gt_boxes.append(gt_boxes[k][b])
                if gt_cls is not None:
                    merged_gt_cls.append(gt_cls[k][b])
                if pos_masks is not None:
                    merged_pos_masks.append(pos_masks[k][b])

            merged_scores = torch.cat(merged_scores)
            merged_features = torch.cat(merged_features)
            merged_proposals = torch.cat(merged_proposals)
            merged_strides = torch.cat(merged_strides)
            if gt_boxes is not None:
                merged_gt_boxes = torch.cat(merged_gt_boxes)
            if gt_cls is not None:
                merged_gt_cls = torch.cat(merged_gt_cls)
            if pos_masks is not None:
                merged_pos_masks = torch.cat(merged_pos_masks)

            topk_k = min(1000, len(merged_scores))
            keep = torch.topk(merged_scores, k=topk_k).indices
            flat_features.append(merged_features[keep])
            flat_boxes.append(merged_proposals[keep])
            flat_scores.append(merged_scores[keep])
            flat_strides.append(merged_strides[keep])
            if gt_boxes is not None:
                flat_gt_boxes.append(merged_gt_boxes[keep])
            if gt_cls is not None:
                flat_gt_cls.append(merged_gt_cls[keep])
            if pos_masks is not None:
                flat_pos_masks.append(merged_pos_masks[keep])

        return torch.stack(flat_features), torch.stack(flat_boxes), torch.stack(flat_scores), torch.stack(flat_strides), flat_gt_boxes, flat_gt_cls, flat_pos_masks

    def forward(
            self, 
            proposals: OrderedDict[str, RPNOutput], 
            fpn_feat: OrderedDict, 
            ground_truth: Annotation 
        ):
        filtered_feat = OrderedDict()
        filtered_proposals = OrderedDict()
        filtered_objectness = OrderedDict()
        filtered_gt_boxes = OrderedDict()
        filtered_gt_cls = OrderedDict()
        filtered_pos_masks = OrderedDict()

        for s_idx, k in enumerate(fpn_feat.keys()):
            if k == "pool":
                continue
            else:
                filtered_feat[k] = fpn_feat[k]
                if self.training:
                    s_prop, s_obj, s_gt_boxes, s_gt_cls, masks = self.sample(
                        proposals[k].region_proposals.clone().detach(), 
                        proposals[k].objectness_scores.clone().detach(),
                        ground_truth.boxes,
                        ground_truth.classifications,
                        self.fpn_strides[s_idx]
                    )
                    filtered_proposals[k] = s_prop
                    filtered_objectness[k] = s_obj
                    filtered_gt_boxes[k] = s_gt_boxes
                    filtered_gt_cls[k] = s_gt_cls
                    filtered_pos_masks[k] = masks
                else:
                    filtered_proposals[k] = proposals[k].region_proposals.clone().detach()
                    filtered_objectness[k] = proposals[k].objectness_scores.clone().detach()

        aligned_feat = self.roi_align_rotated(filtered_feat, filtered_proposals)
        (flat_features, flat_proposals, flat_scores, 
         flat_strides, flat_gt_boxes, flat_gt_cls, flat_pos_masks) = self.flatten_levels(
            aligned_feat, filtered_proposals, filtered_objectness, filtered_gt_boxes, filtered_gt_cls, filtered_pos_masks
        )
        post_fc = self.fc(flat_features)
        classification = self.classification(post_fc)
        regression = self.regression(post_fc)

        # see https://arxiv.org/pdf/1311.2524.pdf
        clamp_v = np.abs(np.log(16/1000))
        flat_proposals = encode(flat_proposals, Encodings.VERTICES, Encodings.THETA_FORMAT_TL_RT)
        boxes_x = flat_proposals[..., 2] * regression[..., 0] + flat_proposals[..., 0]
        boxes_y = flat_proposals[..., 3] * regression[..., 1] + flat_proposals[..., 1]
        boxes_w = flat_proposals[..., 2] * torch.exp(torch.clamp(regression[..., 2], max=clamp_v, min=-clamp_v))
        boxes_h = flat_proposals[..., 3] * torch.exp(torch.clamp(regression[..., 3], max=clamp_v, min=-clamp_v))
        boxes_a = flat_proposals[..., 4] + regression[..., 4]
        boxes = torch.stack((boxes_x, boxes_y, boxes_w, boxes_h, boxes_a), dim=-1)
        boxes[..., :-1] = boxes[..., :-1] * flat_strides.unsqueeze(-1)

        if not self.training:
            post_class_nms_classification = []
            post_class_nms_rois = []
            post_class_nms_boxes = []
            for b in range(classification.shape[0]):
                keep = []
                for c in range(self.num_classes):
                    thr_mask = classification[b, ..., c] > 0.05
                    thr_cls = classification[b, thr_mask]
                    thr_boxes = boxes[b, thr_mask]
                    if len(thr_boxes) == 0:
                        keep.append(torch.empty(0, dtype=torch.int64).to(boxes.device))
                        continue
                    keep_nms = nms_rotated(thr_boxes, thr_cls[..., c], 0.1) # type: ignore
                    keep.append(thr_mask.nonzero().squeeze(-1)[keep_nms])

                keep = torch.cat(keep, dim=0)
                post_class_nms_classification.append(classification[b, keep])
                post_class_nms_rois.append(flat_proposals[b, keep])
                post_class_nms_boxes.append(boxes[b, keep])

            classification = post_class_nms_classification
            boxes = post_class_nms_boxes
            rois = post_class_nms_rois
            loss = None
        else:
            rois = flat_proposals
            loss = None

            for b in range(len(boxes)):
                positive_boxes = boxes[b][flat_pos_masks[b]]
                target_boxes = flat_gt_boxes[b][flat_pos_masks[b]]
                target_boxes = encode(target_boxes, Encodings.VERTICES, Encodings.THETA_FORMAT_TL_RT)
                if loss is None:
                    loss = self.loss(positive_boxes, classification[b], target_boxes, flat_gt_cls[b])
                else:
                    loss = loss + self.loss(positive_boxes, classification[b], target_boxes, flat_gt_cls[b])

        return HeadOutput(
            classification=classification,
            boxes=boxes,
            rois=rois,
            strides=flat_strides,
            loss=loss
        )

    def forward_old(
            self, 
            proposals: OrderedDict, 
            fpn_feat: OrderedDict, 
            anchors: OrderedDict, 
            ground_truth: list[torch.Tensor] | torch.Tensor | None = None,
            reduce_injected_samples: int = 0
        ):
        filtered_feat = {}
        filtered_proposals = {}
        for k in fpn_feat.keys():
            if k == "pool":
                continue
            else:
                filtered_feat[k] = fpn_feat[k]
                filtered_proposals[k] = proposals[k]
        x = self.roi_align_rotated(filtered_feat, filtered_proposals, anchors, ground_truth, reduce_injected_samples)
        post_fc = self.fc(x["features"])
        classification = self.classification(post_fc)
        regression = self.regression(post_fc)
        # bring regression results to reasonable mean and std
        #regression = normalize(
        #    regression, 
        #    target_mean=[0.0] * 5,
        #    target_std=[1.0] * 5,
        #    dim=-2
        #)
        rois = x["boxes"].clone()
        rois[..., :-1] = rois[..., :-1] * x["strides"].unsqueeze(-1)
        clamp_v = np.abs(np.log(16/1000))

        # see https://arxiv.org/pdf/1311.2524.pdf
        boxes_x = x["boxes"][..., 2] * regression[..., 0] + x["boxes"][..., 0]
        boxes_y = x["boxes"][..., 3] * regression[..., 1] + x["boxes"][..., 1]
        #boxes_x = regression[..., 0] * x["boxes"][..., 2] * torch.cos(x["boxes"][..., 4]) - regression[..., 1] * x["boxes"][..., 3] * torch.sin(x["boxes"][..., 4]) + x["boxes"][..., 0]
        #boxes_y = regression[..., 0] * x["boxes"][..., 2] * torch.cos(x["boxes"][..., 4]) + regression[..., 1] * x["boxes"][..., 3] * torch.sin(x["boxes"][..., 4]) + x["boxes"][..., 1]
        boxes_w = x["boxes"][..., 2] * torch.exp(torch.clamp(regression[..., 2], max=clamp_v, min=-clamp_v))
        boxes_h = x["boxes"][..., 3] * torch.exp(torch.clamp(regression[..., 3], max=clamp_v, min=-clamp_v))
        boxes_a = x["boxes"][..., 4] + regression[..., 4]
        boxes = torch.stack((boxes_x, boxes_y, boxes_w, boxes_h, boxes_a), dim=-1)
        boxes[..., :-1] = boxes[..., :-1] * x["strides"].unsqueeze(-1)

        # see section 3.3 of the paper
        if not self.training:
            post_class_nms_classification = []
            post_class_nms_rois = []
            post_class_nms_boxes = []
            for b in range(classification.shape[0]):
                keep = []
                for c in range(self.num_classes):
                    thr_mask = classification[b, ..., c] > 0.05
                    thr_cls = classification[b, thr_mask]
                    thr_boxes = boxes[b, thr_mask]
                    if len(thr_boxes) == 0:
                        keep.append(torch.empty(0, dtype=torch.int64).to(boxes.device))
                        continue
                    keep_nms = nms_rotated(thr_boxes, thr_cls[..., c], 0.1) # type: ignore
                    keep.append(thr_mask.nonzero().squeeze(-1)[keep_nms])

                keep = torch.cat(keep, dim=0)
                post_class_nms_classification.append(classification[b, keep])
                post_class_nms_rois.append(rois[b, keep])
                post_class_nms_boxes.append(boxes[b, keep])

            classification = post_class_nms_classification
            boxes = post_class_nms_boxes
            rois = post_class_nms_rois

        return HeadOutput(
            classification=classification,
            boxes=boxes,
            rois=rois,
            strides = x["strides"]
        )
