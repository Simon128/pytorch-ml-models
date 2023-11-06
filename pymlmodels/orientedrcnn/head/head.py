import time
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple
import numpy as np
import torch.nn.functional as F

from ..utils import HeadOutput, encode, Encodings, RPNOutput, Annotation
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
            proposals: OrderedDict[str, RPNOutput],
            ground_truth_boxes: list[torch.Tensor],
            ground_truth_cls: list[torch.Tensor]
        ):
        n_batches = len(list(proposals.values())[0].region_proposals)
        background_cls = self.num_classes 
        device = ground_truth_boxes[0].device
        sampled_proposals = OrderedDict()
        sampled_objectness = OrderedDict()
        pos_masks = OrderedDict()
        sampled_gt_boxes = OrderedDict()
        sampled_gt_cls = OrderedDict()
        
        for b in range(n_batches):
            merged_levels = []
            merged_iou = []

            for s_idx, k in enumerate(proposals.keys()):
                regions = encode(proposals[k].region_proposals[b], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
                targets = encode(ground_truth_boxes[b] / self.fpn_strides[s_idx], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
                merged_iou.append(pairwise_iou_rotated(regions, targets))
                merged_levels.append(len(merged_iou[-1]))
                
            merged_iou = torch.cat(merged_iou)
            pos_indices, neg_indices = self.sampler(merged_iou)

            _min = 0
            for nl, k in zip(merged_levels,  proposals.keys()):
                rel_pos_pro = pos_indices[0][(pos_indices[0] < _min+nl) & (pos_indices[0] >= _min)] - _min
                rel_neg_pro = neg_indices[0][(neg_indices[0] < _min+nl) & (neg_indices[0] >= _min)] - _min
                rel_pos_gt = pos_indices[1][(pos_indices[0] < _min+nl) & (pos_indices[0] >= _min)]
                n_pos = len(rel_pos_pro)
                n_neg = len(rel_neg_pro)
                
                rand_permutation = torch.randperm(n_pos + n_neg, device=merged_iou.device)
                rand_proposals = torch.cat((
                    proposals[k].region_proposals[b][rel_pos_pro],
                    proposals[k].region_proposals[b][rel_neg_pro]
                ))[rand_permutation]
                rand_objectness = torch.cat((
                    proposals[k].objectness_scores[b][rel_pos_pro],
                    proposals[k].objectness_scores[b][rel_neg_pro]
                ))[rand_permutation]

                sampled_proposals.setdefault(k, [])
                sampled_proposals[k].append(rand_proposals)
                sampled_objectness.setdefault(k, [])
                sampled_objectness[k].append(rand_objectness)
                pos_mask = torch.zeros((n_pos + n_neg,), device=device, dtype=torch.bool)
                pos_mask[:n_pos] = True
                pos_mask = pos_mask[rand_permutation]
                pos_masks.setdefault(k, [])
                pos_masks[k].append(pos_mask)

                gt_boxes = torch.zeros([n_pos + n_neg] + list(ground_truth_boxes[b].shape[1:]), device=device) #type: ignore
                gt_boxes[:n_pos] = ground_truth_boxes[b][rel_pos_gt]
                sampled_gt_boxes.setdefault(k, [])
                sampled_gt_boxes[k].append(gt_boxes[rand_permutation])
                gt_cls = torch.full((n_pos + n_neg,), background_cls, device=device) #type: ignore
                gt_cls[:n_pos] = ground_truth_cls[b][rel_pos_gt]
                gt_cls = gt_cls[rand_permutation]
                sampled_gt_cls.setdefault(k, [])
                sampled_gt_cls[k].append(gt_cls)

                _min += nl

        return sampled_proposals, sampled_objectness, sampled_gt_boxes, sampled_gt_cls, pos_masks

    def _inject_annotations(self, proposals: OrderedDict[str, RPNOutput], ground_truth: Annotation):
        device = ground_truth.boxes[0].device
        for s_idx, k in enumerate(proposals.keys()):
            stride = self.fpn_strides[s_idx]
            for b in range(len(proposals[k].region_proposals)):
                rois = torch.cat([proposals[k].region_proposals[b], ground_truth.boxes[b] / stride])
                rois_obj = torch.cat([
                    proposals[k].objectness_scores[b], 
                    torch.ones((len(ground_truth.boxes[b],))).float().to(device)
                ])
                proposals[k].region_proposals[b] = rois
                proposals[k].objectness_scores[b] = rois_obj

    def flatten_dict(
            self,
            tensor_dict: OrderedDict[str, list[torch.Tensor]], 
            free_memory: bool = True,
            strides: list | None = None
        ):
        """
        tensor_dict values: list len == num batches
        """
        num_batches = len(list(tensor_dict.values())[0])
        device = list(tensor_dict.values())[0][0].device
        flat = [None] * num_batches
        if strides is not None:
            flat_strides = [None] * num_batches

        for s_idx, k in enumerate(list(tensor_dict.keys())):
            for b in range(num_batches):
                if flat[b] is None:
                    flat[b] = tensor_dict[k][b]
                else:
                    flat[b] = torch.cat((flat[b], tensor_dict[k][b]))

                if strides is not None and flat_strides[b] is None:
                    flat_strides[b] = torch.full((tensor_dict[k][b].shape[0],), strides[s_idx]).to(device)
                elif strides is not None:
                    s = torch.full((tensor_dict[k][b].shape[0],), strides[s_idx]).to(device)
                    flat_strides[b] = torch.cat((flat_strides[b], s))

            # free up mem
            if free_memory:
                del tensor_dict[k]

        if strides:
            return flat, flat_strides
        else:
            return flat

    def forward(
            self, 
            proposals: OrderedDict[str, RPNOutput], 
            fpn_feat: OrderedDict, 
            ground_truth: Annotation | None = None
        ):
        if self.training and self.inject_annotation:
            assert ground_truth is not None, "ground truth cannot be None during training"
            self._inject_annotations(proposals, ground_truth)

        filtered_proposals = OrderedDict()
        filtered_objectness = OrderedDict()
        filtered_gt_boxes = OrderedDict()
        filtered_gt_cls = OrderedDict()
        filtered_pos_masks = OrderedDict()

        if ground_truth is not None:
            filtered_proposals, filtered_objectness, filtered_gt_boxes, filtered_gt_cls, filtered_pos_masks = self.sample(
                proposals,
                ground_truth.boxes,
                ground_truth.classifications
            )
        else:
            for k in fpn_feat.keys():
                filtered_proposals[k] = proposals[k].region_proposals
                filtered_objectness[k] = proposals[k].objectness_scores

        aligned_feat = self.roi_align_rotated(fpn_feat, filtered_proposals)

        flat_proposals, flat_strides = self.flatten_dict(filtered_proposals, strides=self.fpn_strides)
        flat_features = self.flatten_dict(aligned_feat)
        if ground_truth is not None:
            flat_gt_boxes = self.flatten_dict(filtered_gt_boxes)
            flat_gt_cls = self.flatten_dict(filtered_gt_cls)
            flat_pos_masks = self.flatten_dict(filtered_pos_masks)
            
        # preventing torch stack because it copies data
        # -> large max mem allocation
        try:
            post_fc = [self.fc(ff.unsqueeze(0)) for ff in flat_features]
        except:
            for ff in flat_features:
                print(ff.shape)
            raise ValueError()
        classification = [self.classification(pf).squeeze(0) for pf in post_fc]
        regression = [self.regression(pf).squeeze(0) for pf in post_fc]

        # see https://arxiv.org/pdf/1311.2524.pdf
        clamp_v = np.abs(np.log(16/1000))
        boxes = []
        for b in range(len(regression)):
            efp = encode(flat_proposals[b], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
            boxes_x = efp[..., 2] * regression[b][..., 0] + efp[..., 0]
            boxes_y = efp[..., 3] * regression[b][..., 1] + efp[..., 1]
            boxes_w = efp[..., 2] * torch.exp(torch.clamp(regression[b][..., 2], max=clamp_v, min=-clamp_v))
            boxes_h = efp[..., 3] * torch.exp(torch.clamp(regression[b][..., 3], max=clamp_v, min=-clamp_v))
            boxes_a = efp[..., 4] + regression[b][..., 4]
            boxes.append(torch.stack((boxes_x, boxes_y, boxes_w, boxes_h, boxes_a), dim=-1))
            boxes[-1][..., :-1] = boxes[-1][..., :-1] * flat_strides[b].unsqueeze(-1)
        loss = None

        if ground_truth is not None:
            loss = None

            for b in range(len(boxes)):
                positive_boxes = regression[b][flat_pos_masks[b]]
                target_boxes = flat_gt_boxes[b][flat_pos_masks[b]] / flat_strides[b][flat_pos_masks[b]].unsqueeze(-1).unsqueeze(-1)
                target_boxes = encode(target_boxes, Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
                fp = encode(flat_proposals[b][flat_pos_masks[b]], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
                rel_target_dx = (target_boxes[..., 0] - fp[..., 0]) / fp[..., 2]
                rel_target_dy = (target_boxes[..., 1] - fp[..., 1]) / fp[..., 3]
                rel_target_dw = torch.log((target_boxes[..., 2] / fp[..., 2]))
                rel_target_dh = torch.log((target_boxes[..., 3] / fp[..., 3]))
                rel_target_da = target_boxes[..., 4] - fp[..., 4]
                rel_targets = torch.stack((rel_target_dx, rel_target_dy, rel_target_dw, rel_target_dh, rel_target_da), dim=-1)
                if loss is None:
                    loss = self.loss(positive_boxes, classification[b], rel_targets, flat_gt_cls[b])
                else:
                    loss = loss + self.loss(positive_boxes, classification[b], rel_targets, flat_gt_cls[b])
            loss = loss / len(boxes)

        if not self.training:
            post_class_nms_classification = []
            post_class_nms_rois = []
            post_class_nms_boxes = []
            for b in range(len(classification)):
                keep = []
                for c in range(self.num_classes):
                    thr_mask = classification[b][..., c] > 0.05
                    thr_cls = classification[b][thr_mask]
                    thr_boxes = boxes[b][thr_mask]
                    if len(thr_boxes) == 0:
                        keep.append(torch.empty(0, dtype=torch.int64).to(boxes[b].device))
                        continue
                    keep_nms = nms_rotated(thr_boxes, thr_cls[..., c], 0.1) # type: ignore
                    keep.append(thr_mask.nonzero().squeeze(-1)[keep_nms])

                keep = torch.cat(keep, dim=0)
                post_class_nms_classification.append(classification[b][keep])
                post_class_nms_rois.append(flat_proposals[b][keep])
                post_class_nms_boxes.append(boxes[b][keep])

            classification = post_class_nms_classification
            boxes = post_class_nms_boxes
            
        return HeadOutput(
            classification=classification,
            boxes=boxes,
            loss=loss
        )

