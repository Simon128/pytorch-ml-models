import torch
import torchvision
import torch.nn.functional as F

from .utils import offsets_to_proposal, midpoint_offset_representation_to_coords, rotated_iou

def gt_coords_to_offsets(boxes: torch.Tensor, anchor_boxes: torch.Tensor):
    # input: n, 4, 2
    x_min = torch.min(boxes[:, :, 0], dim=1)[0]
    x_max = torch.max(boxes[:, :, 0], dim=1)[0]
    y_min = torch.min(boxes[:, :, 1], dim=1)[0]
    y_max = torch.max(boxes[:, :, 1], dim=1)[0]
    # anchor: (ax, ay , aw, ah)
    x_a = anchor_boxes[:, 0]
    y_a = anchor_boxes[:, 1]
    w_a = anchor_boxes[:, 2]
    h_a = anchor_boxes[:, 3]

    # outer box (HBB) midpoint and width, height
    x_g = x_min + (x_max - x_min) / 2
    y_g = y_min + (y_max - y_min) / 2
    w_g = x_max - x_min
    h_g = y_max - y_min
    delta_a_g = x_max - x_g
    delta_b_g = y_max - y_g

    t_a = delta_a_g / w_g
    t_b = delta_b_g / h_g
    t_w = torch.log(w_g/ w_a)
    t_h = torch.log(h_g / h_a)
    t_x = (x_g - x_a) / w_a
    t_y = (y_g - y_a) / h_a

    return torch.stack((t_x, t_y, t_w, t_h, t_a, t_b), dim=1)

def get_tp(anchors, anchor_offsets, ground_truth_boxes):
    b, _, h, w = anchor_offsets.shape
    stacked_anchor_offsets = anchor_offsets.reshape(b, -1, 6, h, w).permute((0, 1, 3, 4, 2)).flatten(1, 3)
    a_x = anchors[:, :, 0, :, :]
    a_y = anchors[:, :, 1, :, :]
    a_w = anchors[:, :, 2, :, :]
    a_h = anchors[:, :, 3, :, :]
    coords_anchors = torch.stack([
        torch.stack((a_x, a_y), dim=2),
        torch.stack((a_x + a_w, a_y), dim=2),
        torch.stack((a_x + a_w, a_y + a_h), dim=2),
        torch.stack((a_x, a_y + a_h), dim=2)
    ], dim=2)
    iou = rotated_iou(coords_anchors, ground_truth_boxes)
    tp_mask = (torch.max(iou, dim=2).values > 0.7).nonzero(as_tuple=True)
    max_for_each_gt = torch.max(iou, dim=1)
    cond_idx = ((max_for_each_gt[0] > 0.3) & (max_for_each_gt[0] < 0.7)).nonzero(as_tuple=True)
    additional_tp = (cond_idx[0], max_for_each_gt[1][cond_idx])
    pred = torch.cat((stacked_anchor_offsets[tp_mask], stacked_anchor_offsets[additional_tp]))
    pos_anchors = torch.cat((anchors[tp_mask], anchors[additional_tp]))
    return pos_anchors, pred



def rpn_loss(anchors: torch.Tensor, anchor_offsets: torch.Tensor, objectness_score: torch.Tensor, ground_truth_boxes: torch.Tensor, cls_sigmoid: bool = True):
    b, _, h, w = anchor_offsets.shape
    stacked_anchor_offsets = anchor_offsets.reshape(b, -1, 6, h, w).permute((0, 1, 3, 4, 2)).flatten(1, 3)
    # proposals = offsets_to_proposal(anchors, anchor_offsets.reshape(b, -1, 6, h, w))
    # proposals = midpoint_offset_representation_to_coords(proposals)
    stacked_objectness_scores = objectness_score.flatten(1)
    a_x = anchors[:, :, 0, :, :]
    a_y = anchors[:, :, 1, :, :]
    a_w = anchors[:, :, 2, :, :]
    a_h = anchors[:, :, 3, :, :]
    coords_anchors = torch.stack([
        torch.stack((a_x, a_y), dim=2),
        torch.stack((a_x + a_w, a_y), dim=2),
        torch.stack((a_x + a_w, a_y + a_h), dim=2),
        torch.stack((a_x, a_y + a_h), dim=2)
    ], dim=2)
    iou = rotated_iou(coords_anchors, ground_truth_boxes)
    
    # every anchor without iou < 0.3 is considered a FP regarding objectness_score
    fp_mask = (torch.max(iou, dim=2).values < 0.3).nonzero(as_tuple=True)
    # every anchor with iou > 0.7 is considered a TP regarding objectness_score
    tp_mask = (torch.max(iou, dim=2).values > 0.7).nonzero(as_tuple=True)
    # every anchor with iou > 0.3 and the highest overlap for a ground truth bb is also considered TP
    max_for_each_gt = torch.max(iou, dim=1)
    cond_idx = ((max_for_each_gt[0] > 0.3) & (max_for_each_gt[0] < 0.7)).nonzero(as_tuple=True)
    additional_tp = (cond_idx[0], max_for_each_gt[1][cond_idx])

    # cls loss (cross entropy as in the paper)
    cls_loss = 0

    if len(stacked_objectness_scores[fp_mask]) > 0:
        cls_loss = cls_loss + F.binary_cross_entropy_with_logits(stacked_objectness_scores[fp_mask], torch.zeros_like(stacked_objectness_scores[fp_mask]))
    
    positives = torch.cat((stacked_objectness_scores[tp_mask], stacked_objectness_scores[additional_tp]))
    
    if len(positives) > 0:
        positive_cls = F.binary_cross_entropy_with_logits(positives, torch.ones_like(positives))
        cls_loss = cls_loss + positive_cls
    
    cls_loss = 1/b * cls_loss

    # for regression loss we only consider the true positives
    anchors = anchors.permute((0,1,3,4,2)).flatten(1,3)
    pos_anchors = torch.cat((anchors[tp_mask], anchors[additional_tp]))
    corr_gt_boxes = torch.cat((ground_truth_boxes[tp_mask], ground_truth_boxes[cond_idx]))
    gt_targets = gt_coords_to_offsets(corr_gt_boxes, pos_anchors)
    pred = torch.cat((stacked_anchor_offsets[tp_mask], stacked_anchor_offsets[additional_tp]))
    if len(pred) > 0:
        regr_loss =  1/b * F.smooth_l1_loss(pred, gt_targets)
    else: 
        regr_loss = 0
    return cls_loss + regr_loss