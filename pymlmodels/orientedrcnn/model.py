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
        self.mem = []

    def forward(self, x: torch.Tensor, annotation: Annotation | None = None):
        _, _, h, w = x.shape

        allocated = torch.cuda.memory_allocated(0)/1024/1024/1024
        reserved = torch.cuda.memory_reserved(0)/1024/1024/1024
        max_reserved = torch.cuda.max_memory_reserved(0)/1024/1024/1024
        if not self.mem or \
            max_reserved > self.mem[2]:
            self.mem = [allocated, reserved, max_reserved]
            print("Forward begin")
            print("torch.cuda.memory_allocated: %fGB"%(self.mem[0]))
            print("torch.cuda.memory_reserved: %fGB"%(self.mem[1]))
            print("torch.cuda.max_memory_reserved: %fGB"%(self.mem[2]))

        feat = self.backbone(x)

        allocated = torch.cuda.memory_allocated(0)/1024/1024/1024
        reserved = torch.cuda.memory_reserved(0)/1024/1024/1024
        max_reserved = torch.cuda.max_memory_reserved(0)/1024/1024/1024
        if not self.mem or \
            max_reserved > self.mem[2]:
            self.mem = [allocated, reserved, max_reserved]
            print("Post backbone")
            print("torch.cuda.memory_allocated: %fGB"%(self.mem[0]))
            print("torch.cuda.memory_reserved: %fGB"%(self.mem[1]))
            print("torch.cuda.max_memory_reserved: %fGB"%(self.mem[2]))

        rpn_out = self.oriented_rpn(feat, annotation, x.device)

        allocated = torch.cuda.memory_allocated(0)/1024/1024/1024
        reserved = torch.cuda.memory_reserved(0)/1024/1024/1024
        max_reserved = torch.cuda.max_memory_reserved(0)/1024/1024/1024
        if not self.mem or \
            max_reserved > self.mem[2]:
            self.mem = [allocated, reserved, max_reserved]
            print("Post rpn")
            print("torch.cuda.memory_allocated: %fGB"%(self.mem[0]))
            print("torch.cuda.memory_reserved: %fGB"%(self.mem[1]))
            print("torch.cuda.max_memory_reserved: %fGB"%(self.mem[2]))

        anchors = self.anchor_generator.generate_like_fpn(feat, w, h, x.device)

        allocated = torch.cuda.memory_allocated(0)/1024/1024/1024
        reserved = torch.cuda.memory_reserved(0)/1024/1024/1024
        max_reserved = torch.cuda.max_memory_reserved(0)/1024/1024/1024
        if not self.mem or \
            max_reserved > self.mem[2]:
            self.mem = [allocated, reserved, max_reserved]
            print("Post anchors")
            print("torch.cuda.memory_allocated: %fGB"%(self.mem[0]))
            print("torch.cuda.memory_reserved: %fGB"%(self.mem[1]))
            print("torch.cuda.max_memory_reserved: %fGB"%(self.mem[2]))

        head_out = self.head(rpn_out, feat, annotation)

        allocated = torch.cuda.memory_allocated(0)/1024/1024/1024
        reserved = torch.cuda.memory_reserved(0)/1024/1024/1024
        max_reserved = torch.cuda.max_memory_reserved(0)/1024/1024/1024
        if not self.mem or \
            max_reserved > self.mem[2]:
            self.mem = [allocated, reserved, max_reserved]
            print("Post head")
            print("torch.cuda.memory_allocated: %fGB"%(self.mem[0]))
            print("torch.cuda.memory_reserved: %fGB"%(self.mem[1]))
            print("torch.cuda.max_memory_reserved: %fGB"%(self.mem[2]))

        return OrientedRCNNOutput(
            rpn_output=rpn_out,
            anchors=anchors,
            backbone_output=feat,
            head_output=head_out
        )
