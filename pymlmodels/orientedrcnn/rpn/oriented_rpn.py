import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops import nms
import torch.nn.functional as F
from torchvision.ops import box_iou

from ..utils.sampler import BalancedSampler
from ..utils import RPNOutput, encode, Encodings, Annotation, LossOutput, normalize
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
            )
        ):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.fpn_level_num = fpn_level_num
        self.fpn_channels = fpn_channels
        self.fpn_strides = fpn_strides
        self.num_anchors = num_anchors
        self.anchor_generator = anchor_generator
        self.sampler = loss_sampler
        self.conv = nn.Conv2d(self.fpn_channels, 256, 3, 1, "same") #type:ignore
        self.regression_branch = nn.Conv2d(256, 6 * self.num_anchors, 1, 1) #type:ignore
        self.objectness_branch = nn.Conv2d(256, self.num_anchors, 1, 1) #type:ignore
        self.bce = nn.BCEWithLogitsLoss()

        # cache
        self.anchors = None
        self.hbb_anchors = None

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
             vertices: OrderedDict[int, list[torch.Tensor]], 
             objectness: OrderedDict[int, list[torch.Tensor]]
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

    def compute_loss(
            self, 
            batch_num: int,
            device: torch.device,
            levelwise_anchors: OrderedDict[int, torch.Tensor],
            levelwise_regression: OrderedDict[int, torch.Tensor],
            levelwise_objectness: OrderedDict[int, torch.Tensor],
            annotation: Annotation
        ):
        cls_loss = 0.0
        regression_loss = 0.0
        for b in range(batch_num):
            num_gt = len(annotation.boxes[b])
            num_anchors_per_level = [len(v[b]) for v in levelwise_anchors.values()]
            total_num_anchors = sum(num_anchors_per_level)
            flat_anchors = torch.zeros((total_num_anchors, 4), device=device)
            flat_hbb_anchors = torch.zeros((total_num_anchors, 4), device=device)
            flat_regression = torch.zeros((total_num_anchors, 6), device=device)
            flat_objectness = torch.zeros((total_num_anchors,), device=device)
            flat_stride = torch.zeros((total_num_anchors,num_gt), device=device)

            start = 0
            hbb_target_boxes = encode(annotation.boxes[b], Encodings.VERTICES, Encodings.HBB_CORNERS)
            for k, stride in enumerate(self.fpn_strides):
                anchors = levelwise_anchors[k][b]
                flat_anchors[start:start+num_anchors_per_level[k]] = anchors
                flat_hbb_anchors[start:start+num_anchors_per_level[k]] = self.hbb_anchors[k][b] * stride
                flat_regression[start:start+num_anchors_per_level[k]] = levelwise_regression[k][b]
                flat_objectness[start:start+num_anchors_per_level[k]] = levelwise_objectness[k][b]
                flat_stride[start:start+num_anchors_per_level[k]] = torch.full(
                    (num_anchors_per_level[k],num_gt), stride, device=device
                )
                start = start + num_anchors_per_level[k]

            iou = box_iou(flat_hbb_anchors, hbb_target_boxes)
            positives_idx, negative_idx = self.sampler(iou)
            n_pos = len(positives_idx[0])
            n_neg = len(negative_idx)
            all_pred_idx = torch.cat((positives_idx[0], negative_idx))
            sampled_obj_pred = flat_objectness[all_pred_idx]
            target_objectness = torch.cat([torch.ones(n_pos, device=device), torch.zeros(n_neg, device=device)])
            cls_loss = cls_loss + self.bce(sampled_obj_pred, target_objectness)

            if n_pos > 0:
                sampled_anchor_offsets = flat_regression[positives_idx[0]]
                sampled_targets = (
                    annotation.boxes[b][positives_idx[1]] / 
                    flat_stride[positives_idx[0], positives_idx[1]][:, None, None]
                )
                target_offsets = encode(
                    sampled_targets,
                    Encodings.VERTICES, Encodings.ANCHOR_OFFSET, 
                    flat_anchors[positives_idx[0]]
                )
                regression_loss = regression_loss + F.smooth_l1_loss(
                    sampled_anchor_offsets, 
                    target_offsets,
                    reduction="mean",
                    beta=0.1111111111111
                )

        cls_loss = cls_loss / batch_num
        regression_loss = regression_loss / batch_num

        return LossOutput(
            total_loss=cls_loss + regression_loss, 
            classification_loss=cls_loss, 
            regression_loss=regression_loss
        )

    def gen_anchors(self, feat: OrderedDict[int, torch.Tensor], device: torch.device):
        self.anchors = self.anchor_generator.generate_like_fpn(feat, self.image_width, self.image_height, device)
        self.hbb_anchors = OrderedDict()
        for k in range(len(self.fpn_strides)):
            self.hbb_anchors[k] = encode(self.anchors[k], Encodings.HBB_CENTERED, Encodings.HBB_CORNERS)


    def forward(self, x: OrderedDict, annotation: Annotation | None = None, device: torch.device = torch.device("cpu")):
        self.gen_anchors(x, device)
        levelwise_proposals = OrderedDict()
        levelwise_objectness = OrderedDict()
        levelwise_regression = OrderedDict()
        levelwise_nms_objectness = OrderedDict()

        for k, v in x.items():
            z = self.conv(v)
            z = F.relu(z)
            regression = self.regression_branch(z)
            b, _, h, w = regression.shape
            regression = regression.view((b, self.num_anchors, 6, h, w)).movedim(2, -1).flatten(1, -2)
            objectness = self.objectness_branch(z).flatten(1)
            levelwise_regression[k] = regression
            levelwise_objectness[k] = objectness
            # encode to vertices (includes midpoint offset encoding as intermediate step)
            proposals = encode(regression, Encodings.ANCHOR_OFFSET, Encodings.VERTICES, self.anchors[k])
            proposals, nms_objectness = self.__nms(proposals, torch.sigmoid(objectness))
            levelwise_proposals[k] = proposals
            levelwise_nms_objectness[k] = nms_objectness

        if annotation:
            loss = self.compute_loss(
                x[0].shape[0],
                device,
                self.anchors,
                levelwise_regression,
                levelwise_objectness,
                annotation
            )

        # get top 1000 proposals based on classification score
        levelwise_proposals, levelwise_objectness = self.__select_top_1000(
            levelwise_proposals, levelwise_nms_objectness
        )
        return RPNOutput(
            region_proposals=levelwise_proposals,
            objectness_scores=levelwise_objectness,
            loss=loss if annotation else None
        )
