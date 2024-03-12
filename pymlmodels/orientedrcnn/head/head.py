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

        return sampled_indices, positives, sampled_ground_truth_boxes, sampled_ground_truth_cls

    def _inject_annotations(self, proposals: OrderedDict[str, RPNOutput], ground_truth: Annotation):
        device = ground_truth.boxes[0].device
        for k in proposals.keys():
            stride = self.fpn_strides[k]
            for b in range(len(proposals[k].region_proposals)):
                rois = torch.cat([proposals[k].region_proposals[b], ground_truth.boxes[b] / stride])
                rois_obj = torch.cat([
                    proposals[k].objectness_scores[b], 
                    torch.ones((len(ground_truth.boxes[b],))).float().to(device)
                ])
                #rois = torch.cat([ground_truth.boxes[b] / stride])
                #rois_obj = torch.cat([
                #    torch.ones((len(ground_truth.boxes[b],))).float().to(device)
                #])
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

    def forward(
            self, 
            proposals: OrderedDict[str, RPNOutput], 
            fpn_feat: OrderedDict, 
            ground_truth: Annotation | None = None
        ):
        if self.training and self.inject_annotation:
            self._inject_annotations(proposals, ground_truth) # type:ignore

        # detaching proposals is key to prevent gradient 
        # propagation from head to RPN, resulting in 
        # RPN and head "combating" each other
        region_proposals = OrderedDict()
        for k in proposals.keys():
            region_proposals[k] = [p.detach() for p in proposals[k].region_proposals]

        flat_proposals, flat_strides, flat_keys = self.flatten_dict(region_proposals, strides=self.fpn_strides) # type:ignore


        if ground_truth is not None:
            sampled_indices, positives, sampled_ground_truth_boxes, sampled_ground_truth_cls = self.sample(
                flat_proposals, # type:ignore
                flat_strides, # type:ignore
                ground_truth.boxes,
                ground_truth.classifications
            )
            if self.training:
                for b in range(len(flat_proposals)):
                    flat_proposals[b] = flat_proposals[b][sampled_indices[b]] # type:ignore
                    flat_strides[b] = flat_strides[b][sampled_indices[b]] # type:ignore
                    flat_keys[b] = flat_keys[b][sampled_indices[b]] # type:ignore


        # debug
        #import cv2
        #import numpy as np
        #image = cv2.imread("/home/simon/unibw/pytorch-ml-models/pymlmodels/orientedrcnn/example/image.tif")
        #t_img = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(0).to("cuda") 
        ##for p, k in zip(flat_proposals[0], flat_keys[0]):
        ##    pts = p.detach().clone().cpu().numpy().copy() * self.fpn_strides[k]
        ##    image = cv2.polylines(image, np.array([pts], dtype=np.int32), True, (255,0,0), 2)


        #test_input = {k: t_img.float() / 255 for k in fpn_feat.keys()}
        #test_roi = self.roi_align_rotated(
        #    test_input, 
        #    torch.stack([fp * self.fpn_strides[k] for fp, k in zip(flat_proposals[0], flat_keys[0])]).unsqueeze(0).float(),
        #    [torch.zeros_like(flat_keys[0])]
        #)

        #for idx in range(len(test_roi[0])):
        #    pts = flat_proposals[0][idx].detach().clone().cpu().numpy().copy() * self.fpn_strides[flat_keys[0][idx]]
        #    image_temp = cv2.polylines(image.copy(), np.array([pts], dtype=np.int32), True, (255,0,0), 2)
        #    roi_img = test_roi[0][idx].detach().clone().cpu().permute((1, 2, 0)).numpy()
        #    cv2.imshow("image", image_temp)
        #    cv2.imshow("roi", roi_img)
        #    while (1):
        #        if cv2.waitKey(20) & 0xFF == 27:
        #            break
        #    cv2.destroyAllWindows()


        aligned_feat = self.roi_align_rotated(fpn_feat, flat_proposals, flat_keys)
        post_fc = [self.fc(ff) for ff in aligned_feat]
        classification = [self.classification(pf) for pf in post_fc]
        regression = [self.regression(pf) for pf in post_fc]

        # see https://arxiv.org/pdf/1311.2524.pdf
        clamp_v = np.abs(np.log(16/1000))
        boxes = []
        for b in range(len(regression)):
            efp = encode(flat_proposals[b], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
            # we resize the proposals here instead of after
            # the changes, to prevent small width/height changes to have a
            # large impact (1% * stride of 64 = 64% change)
            efp[..., :-1] = efp[..., :-1] * flat_strides[b].unsqueeze(-1)
            boxes_x = efp[..., 2] * regression[b][..., 0] + efp[..., 0]
            boxes_y = efp[..., 3] * regression[b][..., 1] + efp[..., 1]
            boxes_w = efp[..., 2] * torch.exp(torch.clamp(regression[b][..., 2], max=clamp_v, min=-clamp_v))
            boxes_h = efp[..., 3] * torch.exp(torch.clamp(regression[b][..., 3], max=clamp_v, min=-clamp_v))
            boxes_a = efp[..., 4] + regression[b][..., 4]
            boxes.append(torch.stack((boxes_x, boxes_y, boxes_w, boxes_h, boxes_a), dim=-1))
        loss = LossOutput(0, 0, 0)

        if ground_truth is not None:
            for b in range(len(boxes)):
                if self.training:
                    fp = encode(flat_proposals[b][:positives[b]], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
                    fp[..., :-1] = fp[..., :-1] * flat_strides[b][:positives[b]].unsqueeze(-1)
                    positive_boxes = regression[b][:positives[b]]
                    target_boxes = sampled_ground_truth_boxes[b] 
                else:
                    pos_idx = sampled_indices[b][:positives[b]]
                    positive_boxes = regression[b][pos_idx]
                    target_boxes = sampled_ground_truth_boxes[b] 
                    fp = encode(flat_proposals[b][pos_idx], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
                    fp[..., :-1] = fp[..., :-1] * flat_strides[b][pos_idx].unsqueeze(-1)
                rel_target_dx = (target_boxes[..., 0] - fp[..., 0]) / fp[..., 2]
                rel_target_dy = (target_boxes[..., 1] - fp[..., 1]) / fp[..., 3]
                rel_target_dw = torch.log((target_boxes[..., 2] / fp[..., 2]))
                rel_target_dh = torch.log((target_boxes[..., 3] / fp[..., 3]))
                rel_target_da = target_boxes[..., 4] - fp[..., 4]
                rel_targets = torch.stack((rel_target_dx, rel_target_dy, rel_target_dw, rel_target_dh, rel_target_da), dim=-1)
                if self.training: 
                    loss = loss + self.loss(positive_boxes, classification[b], rel_targets, sampled_ground_truth_cls[b])
                else:
                    loss = loss + self.loss(positive_boxes, classification[b][sampled_indices[b]], rel_targets, sampled_ground_truth_cls[b])

                if torchdist.is_initialized() and torchdist.get_world_size() > 1:
                        # prevent unused parameters (which crashes DDP)
                        # is there a better way?
                        loss.total_loss = loss.total_loss + torch.sum(regression[b].flatten()) * 0
                
            loss = loss / len(boxes)

        # remove prediction where background class is argmax
        # and encode boxes as vertices
        for b in range(len(classification)):
            mask = (torch.argmax(classification[b], dim=-1) == self.num_classes) == False
            classification[b] = classification[b][mask]
            boxes[b] = boxes[b][mask]

        softmax_class = []
        post_class_nms_classification = []
        post_class_nms_boxes = []
        for b in range(len(classification)):
            keep = []
            softmax_class = torch.softmax(classification[b], dim=-1)
            for c in range(self.num_classes): 
                thr_mask = softmax_class[..., c] > 0.05
                thr_cls = softmax_class[thr_mask]
                thr_boxes = boxes[b][thr_mask]
                if len(thr_boxes) == 0:
                    keep.append(torch.empty(0, dtype=torch.int64, device=boxes[b].device))
                    continue
                keep_nms = nms_rotated(thr_boxes, thr_cls[..., c], 0.1) # type: ignore
                keep.append(thr_mask.nonzero().squeeze(-1)[keep_nms])

            keep = torch.cat(keep, dim=0).unique()
            post_class_nms_classification.append(softmax_class[keep])
            post_class_nms_boxes.append(boxes[b][keep])

        classification = post_class_nms_classification
        boxes = post_class_nms_boxes
            
        return HeadOutput(
            classification=classification,
            boxes=boxes,
            loss=loss
        )

