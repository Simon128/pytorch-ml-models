import torch.nn as nn
import torch
from collections import OrderedDict

class OrientedRPN(nn.Module):
    '''
        Implementation following https://arxiv.org/abs/2108.05699
    '''
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.fpn_level_num = cfg.get("fpn_level_num", 5)
        self.fpn_channels = cfg.get("fpn_channels", 256)
        self.num_anchors = cfg.get("num_anchors", 3)

        self.conv = nn.ModuleDict(
            {str(i): nn.Conv2d(self.fpn_channels, 256, 3, 1, "same") for i in range(self.fpn_level_num)}
        )
        self.regression_branch = nn.ModuleDict(
            {str(i): nn.Conv2d(256, 6 * self.num_anchors, 1, 1) for i in range(self.fpn_level_num)}
        )
        self.objectness_branch = nn.ModuleDict(
            {str(i): nn.Conv2d(256, self.num_anchors, 1, 1) for i in range(self.fpn_level_num)}
        )

    def forward_single(self, x: torch.Tensor, fpn_level: str):
        x = self.conv[fpn_level](x)
        anchor_offsets = self.regression_branch[fpn_level](x)
        objectness_scores = self.objectness_branch[fpn_level](x)
        return {"anchor_offsets": anchor_offsets, "objectness_scores": objectness_scores}

    def forward(self, x):
        assert isinstance(x, torch.Tensor) and self.fpn_level_num == 1 or isinstance(x, OrderedDict)

        if isinstance(x, torch.Tensor):
            return self.forward_single(x, 0)

        output = OrderedDict()
        for idx, (k, v) in enumerate(x.items()):
            output[k] = self.forward_single(v, str(idx))

        return output