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
            proposals: list[torch.Tensor],
            strides: list[torch.Tensor],
            ground_truth_boxes: list[torch.Tensor],
            ground_truth_cls: list[torch.Tensor]
        ):
        n_batches = len(proposals)
        background_cls = self.num_classes 
        device = ground_truth_boxes[0].device

        sampled_indices = []
        positives = []
        sampled_ground_truth_boxes = []
        sampled_ground_truth_cls = []
        
        for b in range(n_batches):
            regions = encode(proposals[b] * strides[b][:, None, None], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
            targets = encode(ground_truth_boxes[b], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
            iou = pairwise_iou_rotated(regions, targets)

            pos_indices, neg_indices = self.sampler(iou)
            n_pos, n_neg = len(pos_indices[0]), len(neg_indices[0])
            positives.append(n_pos)

            sampled_indices.append(torch.cat((pos_indices[0], neg_indices[0])))
            sampled_ground_truth_boxes.append(targets[pos_indices[1]])
            gt_cls = torch.full((n_pos + n_neg,), background_cls, device=device) #type: ignore
            gt_cls[:n_pos] = ground_truth_cls[b][pos_indices[1]]
            sampled_ground_truth_cls.append(gt_cls)

        return sampled_indices, positives, sampled_ground_truth_boxes, sampled_ground_truth_cls

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

        region_proposals = OrderedDict()
        for k in proposals.keys():
            region_proposals[k] = proposals[k].region_proposals
        
        aligned_feat = self.roi_align_rotated(fpn_feat, region_proposals)
        flat_proposals, flat_strides = self.flatten_dict(region_proposals, strides=self.fpn_strides)
        flat_features = self.flatten_dict(aligned_feat)
            
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
            sampled_indices, positives, sampled_ground_truth_boxes, sampled_ground_truth_cls = self.sample(
                flat_proposals,
                flat_strides,
                ground_truth.boxes,
                ground_truth.classifications
            )

            for b in range(len(boxes)):
                pos_idx = sampled_indices[b][:positives[b]]
                positive_boxes = regression[b][pos_idx]
                target_boxes = sampled_ground_truth_boxes[b]
                fp = encode(flat_proposals[b][pos_idx] * flat_strides[b][pos_idx][:, None, None], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
                rel_target_dx = (target_boxes[..., 0] - fp[..., 0]) / fp[..., 2]
                rel_target_dy = (target_boxes[..., 1] - fp[..., 1]) / fp[..., 3]
                rel_target_dw = torch.log((target_boxes[..., 2] / fp[..., 2]))
                rel_target_dh = torch.log((target_boxes[..., 3] / fp[..., 3]))
                rel_target_da = target_boxes[..., 4] - fp[..., 4]
                rel_targets = torch.stack((rel_target_dx, rel_target_dy, rel_target_dw, rel_target_dh, rel_target_da), dim=-1)
                if loss is None:
                    loss = self.loss(positive_boxes, classification[b][sampled_indices[b]], rel_targets, sampled_ground_truth_cls[b])
                else:
                    loss = loss + self.loss(positive_boxes, classification[b][sampled_indices[b]], rel_targets, sampled_ground_truth_cls[b])
                
            if torch.dist.is_initialized() and torch.dist.get_world_size() > 1:
                # prevent unused parameters (which crashes DDP)
                # is there a better way?
                loss = loss + torch.sum(regression) * 0
            loss = loss / len(boxes)

        softmax_class = []
        post_class_nms_classification = []
        post_class_nms_rois = []
        post_class_nms_boxes = []
        for b in range(len(classification)):
            keep = []
            softmax_class = torch.softmax(classification[b], dim=-1)
            for c in range(self.num_classes):
                thr_mask = softmax_class[..., c] > 0.05
                thr_cls = softmax_class[thr_mask]
                thr_boxes = boxes[b][thr_mask]
                if len(thr_boxes) == 0:
                    keep.append(torch.empty(0, dtype=torch.int64).to(boxes[b].device))
                    continue
                keep_nms = nms_rotated(thr_boxes, thr_cls[..., c], 0.1) # type: ignore
                keep.append(thr_mask.nonzero().squeeze(-1)[keep_nms])

            keep = torch.cat(keep, dim=0)
            post_class_nms_classification.append(softmax_class[keep])
            post_class_nms_rois.append(flat_proposals[b][keep])
            post_class_nms_boxes.append(boxes[b][keep])

        classification = post_class_nms_classification
        boxes = post_class_nms_boxes
            
        return HeadOutput(
            classification=classification,
            boxes=boxes,
            loss=loss
        )

