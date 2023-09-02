import torch.nn as nn
import torch

from .oriented_rpn import OrientedRPN, FPNAnchorGenerator
from .oriented_rcnn_head import OrientedRCNNHead

class OrientedRCNN(nn.Module):
    '''
        Implementation following https://arxiv.org/abs/2108.05699
    '''
    def __init__(self, backbone: nn.Module, cfg: dict = {}) -> None:
        super().__init__()
        self.backbone = backbone
        self.oriented_rpn = OrientedRPN(cfg.get("rpn", {}))
        self.head = OrientedRCNNHead(cfg.get("head", {}))
        self.anchor_generator = FPNAnchorGenerator(
            sqrt_size_per_level=(32, 64, 128, 256, 512)
        )

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        backbone_out = self.backbone(x)
        anchors = self.anchor_generator.generate_like_fpn(backbone_out, w, h, x.device)
        proposals = self.oriented_rpn(backbone_out)
        head_out = self.head(proposals, backbone_out, anchors)
        return {
            "anchors": anchors,
            "proposals": proposals, 
            "feat": backbone_out, 
            "pred": head_out
        }
