import torch
import torch.nn as nn
from collections import OrderedDict

from ..data_formats import HeadOutput

from .roi_align_rotated import RoIAlignRotatedWrapper
from ..nms_rotated import nms_rotated

class OrientedRCNNHead(nn.Module):
    def __init__(self, cfg: dict = {}):
        super().__init__()
        if cfg is None:
            cfg = {}

        in_channels = cfg.get("in_channels", 256)
        self.fpn_strides = cfg.get("fpn_strides", [4, 8, 16, 32, 64])
        out_channels = cfg.get("out_channels", 1024)
        self.num_classes = cfg.get("num_classes", 10)
        roi_align_size = cfg.get("roi_align_size", (7,7))
        roi_align_sampling_ratio = cfg.get("roi_align_sampling_ratio", 4)

        self.roi_align_rotated = RoIAlignRotatedWrapper(
            roi_align_size, 
            spatial_scale = 1, 
            sampling_ratio = roi_align_sampling_ratio,
            fpn_strides=self.fpn_strides
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
        self.regression = nn.Sequential(
            nn.Linear(out_channels, 5)
        )

    def forward(self, proposals: OrderedDict, fpn_feat: OrderedDict, anchors: OrderedDict):
        x = self.roi_align_rotated(fpn_feat, proposals, anchors)
        post_fc = self.fc(x["features"])
        classification = self.classification(post_fc)
        regression = self.regression(post_fc)
        # this needs to be changed
        boxes = x["boxes"] + regression
        rois = x["boxes"]

        if not self.training:
            # see section 3.3 of the paper
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
                    keep_nms = nms_rotated(thr_boxes, thr_cls[..., c], 0.1)
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
            rois=rois
        )
