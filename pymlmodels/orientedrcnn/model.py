import torch.nn as nn
import torch

from .rpn import OrientedRPN, FPNAnchorGenerator
from .head import OrientedRCNNHead
from .utils import OrientedRCNNOutput, Annotation

class OrientedRCNN(nn.Module):
    '''
        Implementation following https://arxiv.org/abs/2108.05699
    '''
    def __init__(self, backbone: nn.Module, image_width: int, image_height: int, cfg: dict = {}) -> None:
        super().__init__()
        self.backbone = backbone
        self.oriented_rpn = OrientedRPN(image_width, image_height, **cfg.get("rpn", {}))
        self.head = OrientedRCNNHead(**cfg.get("head", {}))
        self.anchor_generator = FPNAnchorGenerator(
            sqrt_size_per_level=(32, 64, 128, 256, 512)
        )

    def forward(self, x: torch.Tensor, annotation: Annotation | None = None):
        _, _, h, w = x.shape
        feat = self.backbone(x)
        rpn_out = self.oriented_rpn(feat, annotation, x.device)
        anchors = self.anchor_generator.generate_like_fpn(feat, w, h, x.device)
        head_out = self.head(rpn_out, feat, annotation)
        return OrientedRCNNOutput(
            rpn_output=rpn_out,
            anchors=anchors,
            backbone_output=feat,
            head_output=head_out
        )
