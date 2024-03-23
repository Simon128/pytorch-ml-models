import torch
from collections import OrderedDict
import math

class FPNAnchorGenerator:
    def __init__(self, sqrt_size_per_level=(32, 64, 128, 256, 512), aspect_ratios=((1,2), (1,1), (2,1))):
        self.sqrt_size_per_level = sqrt_size_per_level
        self.aspect_ratios = aspect_ratios

    def generate_single(self, x: torch.Tensor, sqrt_size: int, image_width: int, image_height: int, device: torch.device):
        assert len(x.shape) == 4 # expecting (B, C, H, W)
        b, _, h, w = x.shape
        feature_width_factor = image_width / w 
        feature_height_factor = image_height / h
        anchors = []

        for (ar_width, ar_height) in self.aspect_ratios:
            if ar_height > ar_width:
                w_ratio = ar_width / ar_height * (sqrt_size ** 2)
                width = math.sqrt(w_ratio) / feature_width_factor
                height = math.sqrt(w_ratio) * ar_height / feature_height_factor
            else:
                h_ratio = ar_height / ar_width * (sqrt_size ** 2)
                height = math.sqrt(h_ratio) / feature_height_factor
                width = math.sqrt(h_ratio) * ar_width / feature_width_factor

            anchor_width = torch.full((b, h, w), width, device=device)
            anchor_height = torch.full((b, h, w), height, device=device)
            anchor_x = torch.arange(w, device=device).repeat((b, h, 1)) + 0.5
            anchor_y = torch.arange(h, device=device).repeat((b, w, 1)).permute((0, 2, 1)) + 0.5
            anchors.append(torch.stack((anchor_x, anchor_y, anchor_width, anchor_height), dim=-1).flatten(1, -2))

        return torch.cat(anchors, dim=1)

    def generate_like_fpn(self, x: OrderedDict, image_width: int, image_height: int, device: torch.device = torch.device("cpu")):
        assert len(x.values()) == len(self.sqrt_size_per_level), \
            "number of FPN levels and number of size per level does not fit"

        result = OrderedDict()

        for (level_key, level_value), sqrt_size in zip(x.items(), self.sqrt_size_per_level):
            result[level_key] = self.generate_single(level_value, sqrt_size, image_width, image_height, device)

        return result
