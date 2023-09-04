import torch.nn as nn
import torch

from .oriented_rpn import OrientedRPN, FPNAnchorGenerator
from .oriented_rcnn_head import OrientedRCNNHead
from .data_formats import OrientedRCNNOutput

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
        _, _, h, w = x.shape
        feat = self.backbone(x)
        rpn_out = self.oriented_rpn(feat)
        anchors = self.anchor_generator.generate_like_fpn(feat, w, h, x.device)
        head_out = self.head(rpn_out, feat, anchors)
        return OrientedRCNNOutput(
            rpn_output=rpn_out,
            anchors=anchors,
            backbone_output=feat,
            head_output=head_out
        )
