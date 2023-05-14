import torch.nn as nn
import torch

from .oriented_rpn import OrientedRPN

class OrientedRCNN(nn.Module):
    '''
        Implementation following https://arxiv.org/abs/2108.05699
    '''
    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        super().__init__()
        self.backbone = backbone
        self.oriented_rpn = OrientedRPN(cfg)

    def forward(self, x: torch.Tensor):
        backbone_out = self.backbone(x)
        proposals = self.oriented_rpn(backbone_out)
        return {"rpn_out": proposals, "backbone_out": backbone_out}

        
