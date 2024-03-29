import time
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple
import math
import numpy as np
import torch.nn.functional as F
import torch.distributed as torchdist

from ..utils import HeadOutput, encode, Encodings, RPNOutput, Annotation, LossOutput
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
            nn.Flatten(start_dim=1),
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
            n_pos, n_neg = len(pos_indices[0]), len(neg_indices)
            positives.append(n_pos)

            all_indices = torch.cat((pos_indices[0], neg_indices))
            sampled_indices.append(all_indices)

            sampled_ground_truth_boxes.append(targets[pos_indices[1]])
            gt_cls = torch.full((n_pos + n_neg,), background_cls, device=device) #type: ignore
            gt_cls[:n_pos] = ground_truth_cls[b][pos_indices[1]]
            sampled_ground_truth_cls.append(gt_cls)

        return {
            "indices": sampled_indices, 
            "num_pos": positives, 
            "gt_boxes": sampled_ground_truth_boxes, 
            "gt_cls": sampled_ground_truth_cls
        }

    def _inject_annotations(self, proposals: RPNOutput, ground_truth: Annotation):
        device = ground_truth.boxes[0].device
        for k in proposals.region_proposals.keys():
            stride = self.fpn_strides[k]
            for b in range(len(proposals.region_proposals[k])):
                rois = torch.cat([proposals.region_proposals[k][b], ground_truth.boxes[b] / stride])
                rois_obj = torch.cat([
                    proposals.objectness_scores[k][b], 
                    torch.ones((len(ground_truth.boxes[b],))).float().to(device)
                ])
                #rois = torch.cat([ground_truth.boxes[b] / stride])
                #rois_obj = torch.cat([
                #    torch.ones((len(ground_truth.boxes[b],))).float().to(device)
                #])
                proposals.region_proposals[k][b] = rois
                proposals.objectness_scores[k][b] = rois_obj

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
        flat_keys = [None] * num_batches
        flat_strides = [None] * num_batches

        for k in list(tensor_dict.keys()):
            for b in range(num_batches):
                new_flat_keys = torch.full((len(tensor_dict[k][b]),), fill_value=int(k), device=device)
                if flat[b] is None:
                    flat[b] = tensor_dict[k][b] #type: ignore
                    flat_keys[b] = new_flat_keys #type: ignore
                else:
                    flat[b] = torch.cat((flat[b], tensor_dict[k][b])) #type: ignore
                    flat_keys[b] =  torch.cat((flat_keys[b], new_flat_keys)) #type: ignore

                if strides is not None:
                    s = torch.full((tensor_dict[k][b].shape[0],), strides[int(k)], device=device)
                    if flat_strides[b] is None:
                        flat_strides[b] = s # type:ignore
                    else:
                        flat_strides[b] = torch.cat((flat_strides[b], s)) #type: ignore

        if strides:
            return flat, flat_strides, flat_keys
        else:
            return flat, flat_keys

    def _compute_predictions(self, regression, flat_proposals, flat_strides):
        # see https://arxiv.org/pdf/1311.2524.pdf
        clamp_v = np.abs(np.log(16/1000))
        boxes = []
        encoded_proposals = []
        for b in range(len(regression)):
            efp = encode(flat_proposals[b] * flat_strides[b][:, None, None], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
            boxes_x = efp[..., 2] * regression[b][..., 0] + efp[..., 0]
            boxes_y = efp[..., 3] * regression[b][..., 1] + efp[..., 1]
            boxes_w = efp[..., 2] * torch.exp(torch.clamp(regression[b][..., 2], max=clamp_v, min=-clamp_v))
            boxes_h = efp[..., 3] * torch.exp(torch.clamp(regression[b][..., 3], max=clamp_v, min=-clamp_v))
            boxes_a = efp[..., 4] + regression[b][..., 4]
            boxes_a = norm_angle(boxes_a)
            boxes.append(torch.stack((boxes_x, boxes_y, boxes_w, boxes_h, boxes_a), dim=-1))
            encoded_proposals.append(efp)

        return boxes, encoded_proposals

    def _compute_loss(self, regression, classification, theta_propsals, sample_results):
        loss = LossOutput(0, 0, 0)
        positives = sample_results["num_pos"]
        sampled_indices = sample_results["indices"]
        sampled_ground_truth_boxes = sample_results["gt_boxes"]
        sampled_ground_truth_cls = sample_results["gt_cls"]

        for b in range(len(regression)):
            if self.training:
                positive_boxes = regression[b][:positives[b]]
                target_boxes = sampled_ground_truth_boxes[b] 
                fp = theta_propsals[b][:positives[b]]
            else:
                pos_idx = sampled_indices[b][:positives[b]]
                positive_boxes = regression[b][pos_idx]
                target_boxes = sampled_ground_truth_boxes[b] 
                fp = theta_propsals[b][pos_idx]
            rel_target_dx = (target_boxes[..., 0] - fp[..., 0]) / fp[..., 2]
            rel_target_dy = (target_boxes[..., 1] - fp[..., 1]) / fp[..., 3]
            rel_target_dw = torch.log((target_boxes[..., 2] / fp[..., 2]))
            rel_target_dh = torch.log((target_boxes[..., 3] / fp[..., 3]))
            rel_target_da = target_boxes[..., 4] - fp[..., 4]
            rel_target_da = norm_angle(rel_target_da)
            rel_targets = torch.stack((rel_target_dx, rel_target_dy, rel_target_dw, rel_target_dh, rel_target_da), dim=-1)
            if self.training: 
                loss = loss + self.loss(positive_boxes, classification[b], rel_targets, sampled_ground_truth_cls[b])
            else:
                loss = loss + self.loss(positive_boxes, classification[b][sampled_indices[b]], rel_targets, sampled_ground_truth_cls[b])

            if torchdist.is_initialized() and torchdist.get_world_size() > 1:
                # prevent unused parameters (which crashes DDP)
                # is there a better way?
                loss.total_loss = loss.total_loss + torch.sum(regression[b].flatten()) * 0
            
        return loss / len(regression)

    def _remove_background_predictions(self, predictions, classification):
        # remove prediction where background class is argmax
        # and encode boxes as vertices
        for b in range(len(classification)):
            mask = (torch.argmax(classification[b], dim=-1) == self.num_classes) == False
            classification[b] = classification[b][mask]
            predictions[b] = predictions[b][mask]

        return predictions, classification

    def _post_forward_nms(self, predictions, classification):
        softmax_class = []
        post_class_nms_classification = []
        post_class_nms_boxes = []
        device = predictions[0].device
        for b in range(len(classification)):
            keep = []
            softmax_class = torch.softmax(classification[b], dim=-1)
            for c in range(self.num_classes): 
                thr_mask = softmax_class[..., c] > 0.05
                thr_cls = softmax_class[thr_mask]
                thr_boxes = predictions[b][thr_mask]
                if len(thr_boxes) == 0:
                    keep.append(torch.empty(0, dtype=torch.int64, device=device))
                    continue
                keep_nms = nms_rotated(thr_boxes, thr_cls[..., c], 0.1) # type: ignore
                keep.append(thr_mask.nonzero().squeeze(-1)[keep_nms])

            keep = torch.cat(keep, dim=0).unique()
            # remove background class
            post_class_nms_classification.append(softmax_class[keep][..., :self.num_classes])
            post_class_nms_boxes.append(predictions[b][keep])

        return post_class_nms_boxes, post_class_nms_classification

    def forward(
            self, 
            proposals: RPNOutput, 
            fpn_feat: OrderedDict, 
            ground_truth: Annotation | None = None
        ):
        if self.training and self.inject_annotation:
            self._inject_annotations(proposals, ground_truth) # type:ignore

        # detaching proposals is key to prevent gradient 
        # propagation from head to RPN specific layers, resulting in 
        # RPN and head "combating" each other 
        region_proposals = OrderedDict()
        for k in proposals.region_proposals.keys():
            region_proposals[k] = [p.detach() for p in proposals.region_proposals[k]]

        flat_proposals, flat_strides, flat_keys = self.flatten_dict(region_proposals, strides=self.fpn_strides) # type:ignore

        if ground_truth is not None:
            sample_results = self.sample(flat_proposals, flat_strides, ground_truth.boxes, ground_truth.classifications)
            if self.training:
                sampled_indices = sample_results["indices"]
                for b in range(len(flat_proposals)):
                    flat_proposals[b] = flat_proposals[b][sampled_indices[b]] # type:ignore
                    flat_strides[b] = flat_strides[b][sampled_indices[b]] # type:ignore
                    flat_keys[b] = flat_keys[b][sampled_indices[b]] # type:ignore

        aligned_feat = self.roi_align_rotated(fpn_feat, flat_proposals, flat_keys)
        post_fc = [self.fc(ff) for ff in aligned_feat]
        classification = [self.classification(pf) for pf in post_fc]
        regression = [self.regression(pf) for pf in post_fc]

        predictions, theta_proposals = self._compute_predictions(regression, flat_proposals, flat_strides)

        if ground_truth is not None:
            loss = self._compute_loss(regression, classification, theta_proposals, sample_results)
        else:
            loss = LossOutput(0, 0, 0)

        predictions, classification = self._remove_background_predictions(predictions, classification)
        boxes, classification = self._post_forward_nms(predictions, classification)
            
        return HeadOutput(
            classification=classification,
            boxes=[encode(b, Encodings.THETA_FORMAT_BL_RB, Encodings.VERTICES) for b in boxes],
            loss=loss
        )
def norm_angle(angle):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).

    Returns:
        angle (ndarray): shape(n, ).
    """
    return (angle + np.pi / 2) % np.pi - np.pi / 2
