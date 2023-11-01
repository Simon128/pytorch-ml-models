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

    def sample_old(
            self, 
            proposals: list[torch.Tensor], 
            objectness_scores: list[torch.Tensor],
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
            pos_indices, neg_indices = self.sampler(iou, not self.training)
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

        return sampled_proposals, sampled_objectness, sampled_gt_boxes, sampled_gt_cls, pos_masks

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
                regions = encode(proposals[k].region_proposals[b], Encodings.VERTICES, Encodings.THETA_FORMAT_TL_RT)
                targets = encode(ground_truth_boxes[b] / self.fpn_strides[s_idx], Encodings.VERTICES, Encodings.THETA_FORMAT_TL_RT)
                merged_iou.append(pairwise_iou_rotated(regions, targets))
                merged_levels.append(len(merged_iou[-1]))
                
            merged_iou = torch.cat(merged_iou)
            pos_indices, neg_indices = self.sampler(merged_iou, not self.training)

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

    def flatten_levels(
            self, 
            aligned_feat: OrderedDict, 
            proposals: OrderedDict, 
            objectness: OrderedDict,
            gt_boxes: OrderedDict | None = None,
            gt_cls: OrderedDict | None = None,
            pos_masks: OrderedDict | None = None
        ):
        num_batches = len(list(aligned_feat.values())[0])
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

        not_padded = []
        max_n = max([ff.shape[0] for ff in flat_features])
        device = flat_features[0].device

        for b in range(len(flat_features)):
            n = flat_features[b].shape[0]
            if n == max_n: 
                not_padded.append(torch.ones((max_n,)).to(torch.bool).to(device))
            else:
                flat_features[b] = torch.cat((
                    flat_features[b], 
                    torch.zeros([max_n - n] + list(flat_features[b].shape[1:])).to(device)
                ))
                flat_boxes[b] = torch.cat((
                    flat_boxes[b], 
                    torch.zeros([max_n - n] + list(flat_boxes[b].shape[1:])).to(device)
                ))
                flat_scores[b] = torch.cat((
                    flat_scores[b], 
                    torch.zeros([max_n - n] + list(flat_scores[b].shape[1:])).to(device)
                ))
                flat_strides[b] = torch.cat((
                    flat_strides[b], 
                    torch.zeros([max_n - n] + list(flat_strides[b].shape[1:])).to(device)
                ))
                not_padded.append(torch.cat((
                    torch.ones((n,)).to(torch.bool).to(device),
                    torch.zeros((max_n - n,)).to(torch.bool).to(device)
                )))

        return torch.stack(flat_features), torch.stack(flat_boxes), torch.stack(flat_scores), torch.stack(flat_strides), not_padded, flat_gt_boxes, flat_gt_cls, flat_pos_masks

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
        (flat_features, flat_proposals, flat_scores, 
         flat_strides, not_padded, flat_gt_boxes, flat_gt_cls, flat_pos_masks) = self.flatten_levels(
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
        loss = None

        if ground_truth is not None:
            rois = flat_proposals.clone().detach()
            loss = None

            for b in range(len(boxes)):
                positive_boxes = regression[b][not_padded[b]][flat_pos_masks[b]]
                target_boxes = flat_gt_boxes[b][flat_pos_masks[b]] / flat_strides[b][not_padded[b]][flat_pos_masks[b]].unsqueeze(-1).unsqueeze(-1)
                target_boxes = encode(target_boxes, Encodings.VERTICES, Encodings.THETA_FORMAT_TL_RT)
                fp = flat_proposals[b][not_padded[b]][flat_pos_masks[b]]
                rel_target_dx = (target_boxes[..., 0] - fp[..., 0]) / fp[..., 2]
                rel_target_dy = (target_boxes[..., 1] - fp[..., 1]) / fp[..., 3]
                rel_target_dw = torch.log((target_boxes[..., 2] / fp[..., 2]))
                rel_target_dh = torch.log((target_boxes[..., 3] / fp[..., 3]))
                rel_target_da = target_boxes[..., 4] - fp[..., 4]
                rel_targets = torch.stack((rel_target_dx, rel_target_dy, rel_target_dw, rel_target_dh, rel_target_da), dim=-1)
                if loss is None:
                    loss = self.loss(positive_boxes, classification[b][not_padded[b]], rel_targets, flat_gt_cls[b])
                else:
                    loss = loss + self.loss(positive_boxes, classification[b][not_padded[b]], rel_targets, flat_gt_cls[b])
            loss = loss / len(boxes)

        if not self.training:
            post_class_nms_classification = []
            post_class_nms_rois = []
            post_class_nms_boxes = []
            for b in range(classification.shape[0]):
                keep = []
                for c in range(self.num_classes):
                    thr_mask = classification[b, not_padded[b], ..., c] > 0.05
                    thr_cls = classification[b, not_padded[b]][thr_mask]
                    thr_boxes = boxes[b, not_padded[b]][thr_mask]
                    if len(thr_boxes) == 0:
                        keep.append(torch.empty(0, dtype=torch.int64).to(boxes.device))
                        continue
                    keep_nms = nms_rotated(thr_boxes, thr_cls[..., c], 0.1) # type: ignore
                    keep.append(thr_mask.nonzero().squeeze(-1)[keep_nms])

                keep = torch.cat(keep, dim=0)
                post_class_nms_classification.append(classification[b, not_padded[b]][keep])
                post_class_nms_rois.append(flat_proposals[b, not_padded[b]][keep])
                post_class_nms_boxes.append(boxes[b, not_padded[b]][keep])

            classification = post_class_nms_classification
            boxes = post_class_nms_boxes
            rois = post_class_nms_rois
            
        return HeadOutput(
            classification=classification,
            boxes=boxes,
            rois=rois,
            strides=flat_strides,
            loss=loss
        )

