import torch
from torchvision.ops import box_iou
import torch.nn.functional as F

from .encodings import vertices_to_midpoint_offset_gt, midpoint_offset_to_anchor_offset_gt

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

def rpn_loss(
        regression: torch.Tensor, 
        objectness: torch.Tensor, 
        anchors: torch.Tensor, 
        ground_truth: list[torch.Tensor]
    ):
    flat_regression = flatten_regression(regression)
    flat_objectness = flatten_objectness(objectness)
    flat_anchors = flatten_anchors(anchors)
    num_anchors = len(flat_anchors[0])

    losses = []
    # todo: respect false negatives

    for i in range(len(ground_truth)):
        iou = rpn_anchor_iou(flat_anchors[i], ground_truth[i])
        max_for_each_anchor = torch.max(iou, dim=1)
        tp_mask = (max_for_each_anchor.values > 0.7).nonzero(as_tuple=True)
        max_for_each_gt = torch.max(iou, dim=0)
        cond_idx = ((max_for_each_gt[0] > 0.3) & (max_for_each_gt[0] < 0.7)).nonzero(as_tuple=True)
        additional_tp = (cond_idx[0], max_for_each_gt[1][cond_idx])

        cls_target = torch.zeros_like(flat_objectness[i])
        cls_target[tp_mask] = 1.0
        cls_target[additional_tp] = 1.0
        cls_loss = F.binary_cross_entropy_with_logits(flat_objectness[i], cls_target)

        if len(tp_mask[0]) == 0 and len(additional_tp[0]) == 0:
            losses.append(cls_loss)
            continue
        elif len(tp_mask[0]) == 0:
            tp_anchors = flat_anchors[i][additional_tp]
        elif len(additional_tp[0]) == 0:
            tp_anchors = flat_anchors[i][tp_mask]
        else:
            tp_anchors = torch.cat((flat_anchors[i][tp_mask], flat_anchors[i][additional_tp]), dim=0)

        gt_as_midpoint_offset = vertices_to_midpoint_offset_gt(ground_truth[i])
        gt_as_anchor_offset = midpoint_offset_to_anchor_offset_gt(gt_as_midpoint_offset, tp_anchors)
        predictions = torch.cat((flat_regression[i][tp_mask], flat_regression[i][additional_tp]))
        regr_loss = F.smooth_l1_loss(predictions, gt_as_anchor_offset)
        losses.append(cls_loss + regr_loss)

    return torch.mean(torch.tensor(losses))
