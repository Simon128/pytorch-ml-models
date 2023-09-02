import torch
import torch.nn as nn
from collections import OrderedDict

from .roi_align_rotated import RoIAlignRotatedWrapper

# todo:
# https://mmcv.readthedocs.io/en/latest/api/generated/mmcv.ops.RoIAlignRotated.html?highlight=rotated%20roi%20align#mmcv.ops.RoIAlignRotated

class OrientedRCNNHead(nn.Module):
    def __init__(self, cfg: dict = {}):
        super().__init__()
        if cfg is None:
            cfg = {}

        roi_align_size = cfg.get("roi_align_size", (7,7))
        roi_align_sampling_ratio = cfg.get("roi_align_sampling_ratio", 0)

        self.roi_align_rotated = RoIAlignRotatedWrapper(
                roi_align_size, 
                spatial_scale = 1, 
                sampling_ratio = roi_align_sampling_ratio
        )

    def forward(self, proposals: OrderedDict, fpn_feat: OrderedDict, anchors: OrderedDict):
        x = self.roi_align_rotated(fpn_feat, proposals, anchors)
        return x
