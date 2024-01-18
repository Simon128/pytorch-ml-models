import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops import nms
import torch.distributed as torchdist
import torch.nn.functional as F

from .loss import RPNLoss
from ..utils.sampler import BalancedSampler
from ..utils import RPNOutput, normalize, encode, Encodings, Annotation
from .anchor_generator import FPNAnchorGenerator

class OrientedRPN(nn.Module):
    def __init__(
            self, 
            image_width: int,
            image_height: int,
            fpn_level_num = 5, 
            fpn_channels = 256, 
            num_anchors = 3,
            fpn_strides = [4, 8, 16, 32, 64],
            anchor_generator = FPNAnchorGenerator(),
            loss_sampler: BalancedSampler = BalancedSampler(
                n_samples=256,
                pos_fraction=0.5,
                neg_thr=0.3,
                pos_thr=0.7,
                sample_max_inbetween_as_pos=True
            ),
            loss: RPNLoss | None = None
        ):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.fpn_level_num = fpn_level_num
        self.fpn_channels = fpn_channels
        self.fpn_strides = fpn_strides
        self.num_anchors = num_anchors
        self.anchor_generator = anchor_generator
        if loss is None:
            self.loss = RPNLoss(loss_sampler)
        else:
            self.loss = loss
        self.mem = []

        self.conv = nn.Conv2d(self.fpn_channels, 256, 3, 1, "same") #type:ignore
        self.regression_branch = nn.Conv2d(256, 6 * self.num_anchors, 1, 1) #type:ignore
        self.objectness_branch = nn.Conv2d(256, self.num_anchors, 1, 1) #type:ignore


    def __nms(self, vertices: torch.Tensor, objectness: torch.Tensor):
        hbb = encode(vertices, Encodings.VERTICES, Encodings.HBB_CORNERS)

        proposals: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []

        for b_idx in range(len(vertices)):
            # take the top 2000 rpn proposals and apply nms
            topk_k = min(2000, objectness.shape[1])
            topk_proposals = torch.topk(objectness[b_idx], k=topk_k)
            topk_idx = topk_proposals.indices
            topk_scores = topk_proposals.values
            keep = nms(hbb[b_idx, topk_idx], topk_scores, 0.8)
            topk_predictions = vertices[b_idx, topk_idx]
            proposals.append(topk_predictions[keep])
            scores.append(topk_scores[keep])
          
        return proposals, scores
          
    def __select_top_1000(
             self, 
             vertices: OrderedDict[str, list[torch.Tensor]], 
             objectness: OrderedDict[str, list[torch.Tensor]]
        ) :
        n_batches = len(list(vertices.values())[0])
        result_vertices = OrderedDict({k: [] for k in vertices.keys()})
        result_objectness = OrderedDict({k: [] for k in vertices.keys()})
        for b in range(n_batches):
            merged_objectness = []
            merged_levels = []

            for k in vertices.keys():
                merged_objectness.append(objectness[k][b])
                merged_levels.append(len(vertices[k][b]))
                
            merged_objectness = torch.cat(merged_objectness)

            topk_k = min(1000, merged_objectness.shape[0])
            topk_idx = torch.topk(merged_objectness, k=topk_k, sorted=False).indices
            topk_mask = torch.zeros((merged_objectness.shape[0],), dtype=torch.bool).to(merged_objectness.device)
            topk_mask[topk_idx] = True

            _min = 0
            for nl, k in zip(merged_levels, vertices.keys()):
                level_mask = topk_mask[_min:_min+nl]
                if level_mask.shape[0] > 0:
                    selected_vertices = vertices[k][b][level_mask]
                    selected_objectness = objectness[k][b][level_mask]
                else:
                    selected_vertices = []
                    selected_objectness = []
                result_vertices[k].append(selected_vertices)
                result_objectness[k].append(selected_objectness)
                _min += nl

        return result_vertices, result_objectness

    def forward(self, x: OrderedDict, annotation: Annotation | None = None, device: torch.device = torch.device("cpu"), images = None):
        levelwise_proposals = OrderedDict()
        levelwise_objectness = OrderedDict()
        if annotation:
            levelwise_loss = OrderedDict()
        # todo: only once
        anchors = self.anchor_generator.generate_like_fpn(x, self.image_width, self.image_height, device)
        for s_idx, (k, v) in enumerate(x.items()):
            z = self.conv(v)
            z = F.relu(z)
            regression = self.regression_branch(z)
            b, _, h, w = regression.shape
            regression = regression.view((b, self.num_anchors, 6, h, w)).movedim(2, -1).flatten(1, -2)
            objectness = self.objectness_branch(z).flatten(1)
            if annotation:
                loss = self.loss(
                    anchors[k], 
                    [b / self.fpn_strides[s_idx] for b in annotation.boxes], 
                    regression, 
                    objectness
                )
                levelwise_loss[k] = loss
                if torchdist.is_initialized() and torchdist.get_world_size() > 1:
                    # prevent unused parameters (which crashes DDP)
                    # is there a better way?
                    loss.total_loss = loss.total_loss + torch.sum(objectness * 0)
            # encode to vertices (includes midpoint offset encoding as intermediate step)
            proposals = encode(regression, Encodings.ANCHOR_OFFSET, Encodings.VERTICES, anchors[k])
            proposals, objectness = self.__nms(proposals, objectness)
            levelwise_proposals[k] = proposals
            levelwise_objectness[k] = objectness

        # get top 1000 proposals based on classification score
        levelwise_proposals, levelwise_objectness = self.__select_top_1000(
            levelwise_proposals, levelwise_objectness
        )
        output = OrderedDict()
        for k in levelwise_proposals.keys():
            output[k] = RPNOutput(
                region_proposals=levelwise_proposals[k],
                objectness_scores=levelwise_objectness[k],
                loss=levelwise_loss[k] if annotation else None
            )
        return output
