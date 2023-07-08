import torch
from torchvision.ops import box_iou
import torch.nn.functional as F

from ..encodings import vertices_to_midpoint_offset_gt, midpoint_offset_to_anchor_offset_gt

def flatten_anchors(anchors: torch.Tensor):
    b, n, h, w = anchors.shape
    num_anchors = int(n/4)
    r_anchors = anchors.reshape((b, num_anchors, 4, h, w))
    return r_anchors.permute((0, 1, 3, 4, 2)).flatten(start_dim=1, end_dim=3)

def flatten_anchors_hbb(flat_anchors: torch.Tensor):
    anchor_x1 = flat_anchors[:, 0] - flat_anchors[:, 2] / 2
    anchor_x2 = flat_anchors[:, 0] + flat_anchors[:, 2] / 2
    anchor_y1 = flat_anchors[:, 1] - flat_anchors[:, 3] / 2
    anchor_y2 = flat_anchors[:, 1] + flat_anchors[:, 3] / 2
    return torch.stack((anchor_x1, anchor_y1, anchor_x2, anchor_y2), dim=1)

def ground_truth_hbb(ground_truth: torch.Tensor):
    assert len(ground_truth.shape) == 3, "expected ground truth shape of (n, 4, 2)"
    # ground truth shape: n, 4, 2
    gt_x1 = torch.min(ground_truth[:, :, 0], dim=1)[0]
    gt_x2 = torch.max(ground_truth[:, :, 0], dim=1)[0]
    gt_y1 = torch.min(ground_truth[:, :, 1], dim=1)[0]
    gt_y2 = torch.max(ground_truth[:, :, 1], dim=1)[0]
    gt_boxes = torch.stack((gt_x1, gt_y1, gt_x2, gt_y2), dim=1)
    return gt_boxes

def rpn_anchor_iou(flat_anchors: torch.Tensor, ground_truth_boxes: torch.Tensor):
    n, _ = flat_anchors.shape
    hbb_anchor_boxes = flatten_anchors_hbb(flat_anchors)
    hbb_gt_boxes = ground_truth_hbb(ground_truth_boxes)
    return box_iou(hbb_anchor_boxes, hbb_gt_boxes)

def flatten_regression(regression: torch.Tensor):
    b, n, h, w = regression.shape
    num_anchors = int(n/6)
    r_regression = regression.reshape((b, num_anchors, 6, h, w)).permute((0, 1, 3, 4, 2))
    return r_regression.flatten(start_dim=1, end_dim=3)

def flatten_objectness(objectness: torch.Tensor):
    b, num_anchors, h, w = objectness.shape
    return objectness.unsqueeze(4).flatten(start_dim=1, end_dim=3)

def get_positives_mask(iou_matrix: torch.Tensor):
    # iou >= 0.7 -> positive
    # highest iou for ground truth + 0.3 < iou < 0.7
    # iou is a MxN matrix: M = num of anchors, N = num of ground truth
    m, n = iou_matrix.shape
    above_07 = (iou_matrix > 0.7)

    max_per_gt = torch.max(iou_matrix, dim=0)
    max_between = iou_matrix[(max_per_gt.indices, torch.arange(n))] > 0.3
    anchor_idx = max_per_gt.indices[max_between]
    gt_idx = torch.arange(n).to(iou_matrix.device)[max_between]
    between = torch.zeros_like(iou_matrix, dtype=torch.int)
    between[(anchor_idx, gt_idx)] = 1

    positives = above_07 | between
    return positives == 1

def get_negatives_mask(iou_matrix: torch.Tensor):
    # iou < 0.3 -> negative
    # iou is a MxN matrix: M = num of anchors, N = num of ground truth
    return iou_matrix < 0.3

def get_ignore_mask(positives_mask: tuple[torch.tensor, torch.tensor], negatives_mask: tuple[torch.tensor, torch.tensor]):
    # ignore anchors -> all not (negative | positive)
    # ignore gt -> all not (negative | positive)
    mask = torch.ones_like(positives_mask, dtype=int)
    mask[positives_mask] = 0
    mask[negatives_mask] = 0
    return mask == 1

def rpn_loss(
        regression: torch.Tensor, 
        objectness: torch.Tensor, 
        anchors: torch.Tensor, 
        ground_truth: list[torch.Tensor],
        scale: float = 1.0
    ):
    flat_regression = flatten_regression(regression) * scale
    flat_objectness = flatten_objectness(objectness)
    flat_anchors = flatten_anchors(anchors) * scale
    num_anchors = len(flat_anchors[0])

    losses = []

    for i in range(len(ground_truth)):
        iou = rpn_anchor_iou(flat_anchors[i], ground_truth[i])
        positives = get_positives_mask(iou)
        negatives = get_negatives_mask(iou)
        ignore = get_ignore_mask(positives, negatives)
        positives_idx = torch.nonzero(positives, as_tuple=True)
        negatives_idx = torch.nonzero(negatives, as_tuple=True)
        ignore_idx = torch.nonzero(ignore, as_tuple=True)

        cls_target = torch.zeros_like(flat_objectness[i])
        cls_target[positives_idx[0]] = 1.0
        weight = torch.ones_like(cls_target)
        weight[ignore_idx[0]] = 0
        if len(positives_idx[0]) > 0 and False:
            weight[positives_idx[0]] = len(negatives_idx[0]) / len(positives_idx[0])
        cls_loss = F.binary_cross_entropy_with_logits(flat_objectness[i], cls_target, weight=weight, reduction='sum')

        if torch.count_nonzero(positives) <= 0:
            losses.append(cls_loss)
            continue

        relevant_gt = ground_truth[i][positives_idx[1]]
        relevant_pred = flat_regression[i][positives_idx[0]]
        relevant_anchor = flat_anchors[i][positives_idx[0]]
        gt_as_midpoint_offset = vertices_to_midpoint_offset_gt(relevant_gt)
        gt_as_anchor_offset = midpoint_offset_to_anchor_offset_gt(gt_as_midpoint_offset, relevant_anchor)
        regr_loss = F.smooth_l1_loss(relevant_pred, gt_as_anchor_offset, reduction='sum')
        losses.append(cls_loss + regr_loss)

    return torch.mean(torch.stack(losses))
