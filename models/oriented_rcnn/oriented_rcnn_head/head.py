import torch
import torch.nn as nn
from collections import OrderedDict

# todo:
# https://mmcv.readthedocs.io/en/latest/api/generated/mmcv.ops.RoIAlignRotated.html?highlight=rotated%20roi%20align#mmcv.ops.RoIAlignRotated

class OrientedRCNNHead(nn.Module):
    def __init__(self, cfg: dict = {}):
        super().__init__()
        if cfg is None:
            cfg = {}

        input_channels = cfg.get("input_channels", 256)
        roi_align_size = cfg.get("roi_align_size", (7,7))
        roi_align_sampling_ratio = cfg.get("roi_align_sampling_ratio", 2000)

        self.roi_align_rotated = ROIAlignRotated(
                roi_align_size, 
                spatial_scale = 1, 
                sampling_ratio = roi_align_sampling_ratio
        )
    
    def process_proposals(proposals: torch.Tensor):
        test = 5
        return None

    def forward(proposals: OrderedDict, fpn_feat: OrderedDict):
        x = []

        for p, feat in zip(proposal.values(), fpn_feat.values()):
            processed_proposals = self.process_proposals(proposals)
            x.append(self.roi_align_rotate(feat, processed_proposals))
