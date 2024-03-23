import torch.nn as nn
import torch
from typing import OrderedDict, Any


from .rpn import OrientedRPN, FPNAnchorGenerator
from .head import OrientedRCNNHead
from .utils import OrientedRCNNOutput, Annotation, RPNOutput

class OrientedRCNN(nn.Module):
    '''
        Implementation following https://arxiv.org/abs/2108.05699
    '''
    def __init__(self, backbone: nn.Module, image_width: int, image_height: int, cfg: dict = {}) -> None:
        super().__init__()
        self.backbone = backbone
        self.oriented_rpn = OrientedRPN(image_width, image_height, **cfg.get("rpn", {}))
        self.head = OrientedRCNNHead(**cfg.get("head", {}))
        self.anchors = None
        self.anchor_generator = FPNAnchorGenerator(
            sqrt_size_per_level=(32, 64, 128, 256, 512)
        )

    def __filter_dict_by_max_key(self, d: OrderedDict[int, Any], max_key: int):
        result = OrderedDict()

        for key, val in d.items():
            if key <= max_key:
                result[key] = val

        return result

    def forward(self, x: torch.Tensor, annotation: Annotation | None = None):
        feat = self.backbone(x)

        numerical_ordered_feat = OrderedDict()
        for idx, v in enumerate(feat.values()):
            numerical_ordered_feat[idx] = v

        rpn_out = self.oriented_rpn(numerical_ordered_feat, annotation, x.device)

        rpn_for_head = RPNOutput(
            region_proposals=self.__filter_dict_by_max_key(rpn_out.region_proposals, 3),
            objectness_scores=self.__filter_dict_by_max_key(rpn_out.objectness_scores, 3)
        )
        feat_for_head = self.__filter_dict_by_max_key(numerical_ordered_feat, 3)

        head_out = self.head(rpn_for_head, feat_for_head, annotation)
        return OrientedRCNNOutput(
            rpn_output=rpn_out,
            backbone_output=numerical_ordered_feat,
            head_output=head_out
        )
