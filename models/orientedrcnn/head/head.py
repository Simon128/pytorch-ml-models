import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple

from ..utils import HeadOutput, normalize, encode, Encodings
from .roi_align_rotated import RoIAlignRotatedWrapper
from ..ops import nms_rotated

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
            n_injected_samples: int = 1000
        ):
        super().__init__()
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides

        self.roi_align_rotated = RoIAlignRotatedWrapper(
            roi_align_size, 
            spatial_scale = 1, 
            sampling_ratio = roi_align_sampling_ratio,
            fpn_strides=self.fpn_strides[:-1],
            inject_annotation=inject_annotation,
            n_injected_samples=n_injected_samples
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

    def forward(
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
        regression = normalize(
            regression, 
            target_mean=[0.0] * 5,
            target_std=[0.1, 0.1, 0.2, 0.2, 0.1],
            dim=-2
        )
        rois = x["boxes"].clone()
        rois[..., :-1] = rois[..., :-1] * x["strides"][..., :-1]

        # see https://arxiv.org/pdf/1311.2524.pdf
        boxes_x = x["boxes"][..., 2] * regression[..., 0] + x["boxes"][..., 0]
        boxes_y = x["boxes"][..., 3] * regression[..., 1] + x["boxes"][..., 1]
        boxes_w = x["boxes"][..., 2] * torch.exp(regression[..., 2])
        boxes_h = x["boxes"][..., 3] * torch.exp(regression[..., 3])
        # not sure what to do with angles
        boxes_a = x["boxes"][..., 4] * torch.exp(regression[..., 4])
        boxes = torch.stack((boxes_x, boxes_y, boxes_w, boxes_h, boxes_a), dim=-1)
        boxes[..., :-1] = boxes[..., :-1] * x["strides"][..., :-1]

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
