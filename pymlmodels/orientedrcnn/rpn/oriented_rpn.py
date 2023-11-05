import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops import box_iou
from torchvision.ops import nms
import torch.distributed as torchdist

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
            keep = nms(hbb[b_idx, topk_idx], topk_scores, 0.8)
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

    def sample(
            self, 
            anchors: OrderedDict[str, torch.Tensor],
            offsets: OrderedDict[str, torch.Tensor],
            objectness: OrderedDict[str, torch.Tensor],
            ground_truth_boxes: list[torch.Tensor]
        ):
        n_batches = len(ground_truth_boxes)
        device = ground_truth_boxes[0].device
        sampled_anchors = OrderedDict()
        sampled_offsets = OrderedDict()
        sampled_objectness = OrderedDict()
        pos_masks = OrderedDict()
        sampled_gt_boxes = OrderedDict()
        sampled_gt_objectness = OrderedDict()
        
        for b in range(n_batches):
            merged_levels = []
            merged_iou = []

            for s_idx, k in enumerate(anchors.keys()):
                merged_iou.append(self.rpn_anchor_iou(anchors[k][b], ground_truth_boxes[b] / self.fpn_strides[s_idx]))
                merged_levels.append(len(merged_iou[-1]))
                
            merged_iou = torch.cat(merged_iou)
            pos_indices, neg_indices = self.sampler(merged_iou, not self.training)

            _min = 0
            for s_idx, (nl, k) in enumerate(zip(merged_levels, anchors.keys())):
                rel_pos_pro = pos_indices[0][(pos_indices[0] < _min+nl) & (pos_indices[0] >= _min)] - _min
                rel_neg_pro = neg_indices[0][(neg_indices[0] < _min+nl) & (neg_indices[0] >= _min)] - _min
                rel_pos_gt = pos_indices[1][(pos_indices[0] < _min+nl) & (pos_indices[0] >= _min)]
                n_pos = len(rel_pos_pro)
                n_neg = len(rel_neg_pro)
                
                rand_permutation = torch.randperm(n_pos + n_neg, device=merged_iou.device)
                rand_proposals = torch.cat((
                    anchors[k][b][rel_pos_pro],
                    anchors[k][b][rel_neg_pro]
                ))[rand_permutation]
                rand_offsets = torch.cat((
                    offsets[k][b][rel_pos_pro],
                    offsets[k][b][rel_neg_pro]
                ))[rand_permutation]
                rand_objectness = torch.cat((
                    objectness[k][b][rel_pos_pro],
                    objectness[k][b][rel_neg_pro]
                ))[rand_permutation]

                sampled_anchors.setdefault(k, [])
                sampled_anchors[k].append(rand_proposals)
                sampled_offsets.setdefault(k, [])
                sampled_offsets[k].append(rand_offsets)
                sampled_objectness.setdefault(k, [])
                sampled_objectness[k].append(rand_objectness)
                pos_mask = torch.zeros((n_pos + n_neg,), device=device, dtype=torch.bool)
                pos_mask[:n_pos] = True
                pos_mask = pos_mask[rand_permutation]
                pos_masks.setdefault(k, [])
                pos_masks[k].append(pos_mask)

                gt_boxes = torch.zeros([n_pos + n_neg,6], device=device) #type: ignore
                
                gt_boxes[:n_pos] = encode(
                    ground_truth_boxes[b][rel_pos_gt] / self.fpn_strides[s_idx], 
                    Encodings.VERTICES, 
                    Encodings.ANCHOR_OFFSET, 
                    anchors[k][b][rel_pos_pro]
                )
                sampled_gt_boxes.setdefault(k, [])
                sampled_gt_boxes[k].append(gt_boxes[rand_permutation])
                gt_cls = torch.zeros((n_pos + n_neg,), device=device, dtype=torch.float) #type: ignore
                gt_cls[:n_pos] = 1.0
                gt_cls = gt_cls[rand_permutation]
                sampled_gt_objectness.setdefault(k, [])
                sampled_gt_objectness[k].append(gt_cls)

                _min += nl

        return sampled_anchors, sampled_offsets, sampled_objectness, sampled_gt_boxes, sampled_gt_objectness, pos_masks

    def forward(self, x: OrderedDict, annotation: Annotation | None = None, device: torch.device = torch.device("cpu")):
        assert isinstance(x, OrderedDict)
        if self.training:
            assert annotation is not None, "ground truth cannot be None if training"
        anchors = self.anchor_generator.generate_like_fpn(x, self.image_width, self.image_height, device)

        proc_proposals = OrderedDict()
        proc_objectness = OrderedDict()
        proc_anchors = OrderedDict()
        proc_loss = OrderedDict()
        output = OrderedDict()

        all_offsets = OrderedDict()
        all_objectness = OrderedDict()

        for idx, (k, v) in enumerate(x.items()):
            offsets, objectness = self.forward_single(v, idx, anchors[k])
            all_offsets[k] = offsets
            all_objectness[k] = objectness
            stride = self.fpn_strides[idx]
            loss = None
            proposals, objectness, filtered_anchors = self.filter_proposals(offsets, objectness, anchors[k], stride)
            proc_proposals[k] = proposals
            proc_objectness[k] = objectness
            proc_anchors[k] = filtered_anchors
            proc_loss[k] = None

        if annotation is not None:
            (sampled_anchors, sampled_offsets, sampled_objectness, 
             sampled_gt_boxes, sampled_gt_objectness, pos_masks) = self.sample(anchors, all_offsets, all_objectness, annotation.boxes)
        for k, v in x.items():
            if annotation is not None:
                for b in range(len(v)):
                    pos_anchor_offsets = sampled_offsets[k][b][pos_masks[k][b]]
                    target_offsets = sampled_gt_boxes[k][b][pos_masks[k][b]]
                    temp_loss = self.loss(pos_anchor_offsets, sampled_objectness[k][b], sampled_gt_objectness[k][b] , target_offsets)
                    if loss:
                        loss = loss + temp_loss
                    else:
                        loss = temp_loss
                    if torchdist.is_initialized() and torchdist.get_world_size() > 1:
                        # prevent unused parameters (which crashes DDP)
                        # is there a better way?
                        loss.total_loss = loss.total_loss + torch.sum(all_objectness[k] * 0)

                proc_loss[k] = loss / len(v)

        proposals, objectness, filtered_anchors = self.select_top_1000(proc_proposals, proc_objectness, proc_anchors)
        for k in proposals.keys():
            output[k] = RPNOutput(
                region_proposals=proposals[k], 
                objectness_scores=objectness[k], 
                loss=proc_loss[k]
            )
        return output

    def forward_single(self, x: torch.Tensor, fpn_level: int, anchors):
        x = self.conv[str(fpn_level)](x)
        stride = self.fpn_strides[fpn_level]
        b, _, h, w = x.shape
        anchor_offsets = self.regression_branch[str(fpn_level)](x)
        anchor_offsets = anchor_offsets.view((b, self.num_anchors, -1, h, w))
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
