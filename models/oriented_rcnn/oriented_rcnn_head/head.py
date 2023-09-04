import torch
import torch.nn as nn
from collections import OrderedDict

from ..data_formats import HeadOutput

from .roi_align_rotated import RoIAlignRotatedWrapper

# todo:
# https://mmcv.readthedocs.io/en/latest/api/generated/mmcv.ops.RoIAlignRotated.html?highlight=rotated%20roi%20align#mmcv.ops.RoIAlignRotated

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
        roi_align_sampling_ratio = cfg.get("roi_align_sampling_ratio", 2)

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
        boxes = regression + x["boxes"]
        return HeadOutput(
            classification=classification,
            boxes=boxes
        )
