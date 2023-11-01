import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops import box_iou
from torchvision.ops import nms

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
            sampler: BalancedSampler = BalancedSampler(
                n_samples=256,
                pos_fraction=0.5,
                neg_thr=0.3,
                pos_thr=0.7,
                sample_max_inbetween_as_pos=True
            ),
            loss: RPNLoss = RPNLoss()
        ):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.fpn_level_num = fpn_level_num
        self.fpn_channels = fpn_channels
        self.fpn_strides = fpn_strides
        self.num_anchors = num_anchors
        self.anchor_generator = anchor_generator
        self.sampler = sampler
        self.loss = loss

        self.conv = nn.ModuleDict(
            {str(i): nn.Conv2d(self.fpn_channels, 256, 3, 1, "same") for i in range(self.fpn_level_num)}
        )
        self.regression_branch = nn.ModuleDict(
            {str(i): nn.Conv2d(256, 6 * self.num_anchors, 1, 1) for i in range(self.fpn_level_num)}
        )
        self.objectness_branch = nn.ModuleDict(
            {str(i): nn.Conv2d(256, self.num_anchors, 1, 1) for i in range(self.fpn_level_num)}
        )

    def rpn_anchor_iou(self, anchors: torch.Tensor, target_boxes: torch.Tensor):
        hbb_anchors = encode(anchors, Encodings.HBB_CENTERED, Encodings.HBB_CORNERS)
        hbb_target_boxes = encode(target_boxes, Encodings.VERTICES, Encodings.HBB_CORNERS)
        return box_iou(hbb_anchors, hbb_target_boxes)

    def filter_proposals(self, anchor_offsets: torch.Tensor, objectness: torch.Tensor, anchors: torch.Tensor, stride: float):
        predictions = encode(anchor_offsets, Encodings.ANCHOR_OFFSET, Encodings.VERTICES, anchors=anchors)
        hbb = encode(predictions, Encodings.VERTICES, Encodings.HBB_CORNERS)

        proposals: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        filt_anchors: list[torch.Tensor] = []

        # alternative ?
        #for b_idx in range(len(predictions)):
        #    # take the top 2000 rpn proposals and apply nms
        #    keep = nms(hbb[b_idx], objectness[b_idx], 0.8)
        #    topk_k = min(2000, keep.shape[0])
        #    topk_proposals = torch.topk(objectness[b_idx][keep], k=topk_k)
        #    topk_idx = topk_proposals.indices
        #    topk_scores = topk_proposals.values
        #    topk_predictions = predictions[b_idx][keep][topk_idx]
        #    topk_anchors = anchors[b_idx][keep][topk_idx]
        #    proposals.append(topk_predictions)
        #    scores.append(topk_scores)
        #    filt_anchors.append(topk_anchors)
        for b_idx in range(len(predictions)):
            # take the top 2000 rpn proposals and apply nms
            topk_k = min(2000, objectness.shape[1])
            topk_proposals = torch.topk(objectness[b_idx], k=topk_k)
            topk_idx = topk_proposals.indices
            topk_scores = topk_proposals.values
            keep = nms(hbb[b_idx, topk_idx], topk_scores, 0.5)
            topk_predictions = predictions[b_idx, topk_idx]
            topk_anchors = anchors[b_idx][topk_idx]
            proposals.append(topk_predictions[keep])
            scores.append(topk_scores[keep])
            filt_anchors.append(topk_anchors[keep])
          
        return proposals, scores, filt_anchors
          
    def select_top_1000(
             self, 
             anchor_offsets: OrderedDict[str, list[torch.Tensor]], 
             objectness: OrderedDict[str, list[torch.Tensor]],
             anchors: OrderedDict[str, list[torch.Tensor]]
        ) :
        n_batches = len(list(anchor_offsets.values())[0])
        result_offsets = OrderedDict({k: [] for k in anchor_offsets.keys()})
        result_objectness = OrderedDict({k: [] for k in anchor_offsets.keys()})
        result_anchors = OrderedDict({k: [] for k in anchor_offsets.keys()})
        for b in range(n_batches):
            merged_objectness = []
            merged_levels = []

            for k in anchor_offsets.keys():
                merged_objectness.append(objectness[k][b])
                merged_levels.append(len(anchor_offsets[k][b]))
                
            merged_objectness = torch.cat(merged_objectness)

            topk_k = min(1000, merged_objectness.shape[0])
            topk_idx = torch.topk(merged_objectness, k=topk_k, sorted=False).indices
            topk_mask = torch.zeros((merged_objectness.shape[0],), dtype=torch.bool).to(merged_objectness.device)
            topk_mask[topk_idx] = True

            _min = 0
            for nl, k in zip(merged_levels, anchor_offsets.keys()):
                level_mask = topk_mask[_min:_min+nl]
                if level_mask.shape[0] > 0:
                    selected_offsets = anchor_offsets[k][b][level_mask]
                    selected_objectness = objectness[k][b][level_mask]
                    selected_anchors = anchors[k][b][level_mask]
                else:
                    selected_offsets = []
                    selected_objectness = []
                    selected_anchors = []
                result_offsets[k].append(selected_offsets)
                result_objectness[k].append(selected_objectness)
                result_anchors[k].append(selected_anchors)
                _min += nl

        return result_offsets, result_objectness, result_anchors

    def forward(self, x: OrderedDict, annotation: Annotation | None = None, device: torch.device = torch.device("cpu")):
        assert isinstance(x, OrderedDict)
        if self.training:
            assert annotation is not None, "ground truth cannot be None if training"
        anchors = self.anchor_generator.generate_like_fpn(x, self.image_width, self.image_height, device)

        proc_proposals = OrderedDict()
        proc_objectness = OrderedDict()
        proc_anchors = OrderedDict()
        proc_loss = OrderedDict()

        for idx, (k, v) in enumerate(x.items()):
            offsets, objectness = self.forward_single(v, idx, anchors[k])
            stride = self.fpn_strides[idx]
            loss = None

            if annotation is not None:
                for b in range(len(v)):
                    ann_boxes = annotation.boxes[b]
                    iou = self.rpn_anchor_iou(anchors[k][b] * stride, ann_boxes)
                    pos_indices, neg_indices = self.sampler(iou)
                    pos_anchor_offsets = offsets[b][pos_indices[0]]
                    pos_objectness = objectness[b][pos_indices[0]]
                    neg_objectness = objectness[b][neg_indices[0]]
                    target_vertices = ann_boxes[pos_indices[1]]
                    target_offsets = encode(target_vertices / stride, Encodings.VERTICES, Encodings.ANCHOR_OFFSET, anchors[k][b][pos_indices[0]])
                    temp_loss = self.loss(pos_anchor_offsets, pos_objectness, neg_objectness, target_offsets)
                    if loss:
                        loss = loss + temp_loss
                    else:
                        loss = temp_loss

                loss = loss / len(v)

            proposals, objectness, filtered_anchors = self.filter_proposals(offsets, objectness, anchors[k], stride)
            proc_proposals[k] = proposals
            proc_objectness[k] = objectness
            proc_anchors[k] = filtered_anchors
            proc_loss[k] = loss

        output = OrderedDict()
        proposals, objectness, filtered_anchors = self.select_top_1000(proc_proposals, proc_objectness, proc_anchors)
        for k in proposals.keys():
            output[k] = RPNOutput(
                region_proposals=proposals[k], 
                objectness_scores=objectness[k], 
                anchors=filtered_anchors[k], 
                loss=proc_loss[k]
            )
        return output

    def forward_single(self, x: torch.Tensor, fpn_level: int, anchors):
        x = self.conv[str(fpn_level)](x)
        stride = self.fpn_strides[fpn_level]
        b, _, h, w = x.shape
        anchor_offsets = self.regression_branch[str(fpn_level)](x)
        anchor_offsets = anchor_offsets.reshape((b, self.num_anchors, -1, h, w))
        anchor_offsets = torch.movedim(anchor_offsets, 2, -1)
        anchor_offsets = anchor_offsets.flatten(1, -2)
        # normalize the anchor offsets to a reasonable mean and std
        anchor_offsets = normalize(
            anchor_offsets, 
            target_mean=[0.0] * 6,
            target_std=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
            dim=-2
        )
        objectness_scores = self.objectness_branch[str(fpn_level)](x)
        objectness = objectness_scores.flatten(1)
        return anchor_offsets, objectness
