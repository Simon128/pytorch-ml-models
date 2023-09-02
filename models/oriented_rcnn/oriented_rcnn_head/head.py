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

        channels = cfg.get("channels", 256)
        self.num_classes = cfg.get("num_classes", 10)
        roi_align_size = cfg.get("roi_align_size", (7,7))
        roi_align_sampling_ratio = cfg.get("roi_align_sampling_ratio", 0)

        self.roi_align_rotated = RoIAlignRotatedWrapper(
                roi_align_size, 
                spatial_scale = 1, 
                sampling_ratio = roi_align_sampling_ratio
        )
        num_features = channels * roi_align_size[0] * roi_align_size[1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
            nn.ReLU()
        )
        self.classification = nn.Linear(num_features, self.num_classes)
        self.regression = nn.Sequential(
            nn.Linear(num_features, 6 * self.num_classes)
        )

    def forward(self, proposals: OrderedDict, fpn_feat: OrderedDict, anchors: OrderedDict):
        x = self.roi_align_rotated(fpn_feat, proposals, anchors)
        batch_size = x.shape[0]
        post_fc = self.fc(x["features"])
        classification = self.classification(post_fc)
        regression = self.regression(post_fc)
        regression = regression.reshape((batch_size, self.num_classes, -1))
        return {
            "classification": classification,
            "regression": regression,
            "filtered_rpn_boxes": x["boxes"],
            "filtered_rpn_scores": x["scores"],
            "filtered_rpn_vertices": x["vertices"]
        }
