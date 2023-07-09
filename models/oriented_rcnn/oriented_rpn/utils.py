import torch

from ..encodings import anchor_offset_to_midpoint_offset, midpoint_offset_to_vertices
from .loss import flatten_anchors, flatten_regression, rpn_anchor_iou, get_positives_mask, flatten_objectness

def pred_to_vertices_by_gt(anchors: torch.Tensor, ground_truth: torch.Tensor, regression: torch.Tensor, scale: float):
    flat_regression = flatten_regression(regression.unsqueeze(0)) * scale
    flat_anchors = flatten_anchors(anchors.unsqueeze(0)) * scale
    iou = rpn_anchor_iou(flat_anchors[0], ground_truth)
    positives_idx = get_positives_mask(iou)
    pos_gt_idx = positives_idx.sum(dim=0).nonzero().flatten()
    pos_anchor_idx = positives_idx.sum(dim=1).nonzero().flatten()
    relevant_gt = ground_truth[pos_gt_idx]
    relevant_pred = flat_regression[0][pos_anchor_idx]
    relevant_anchor = flat_anchors[0][pos_anchor_idx]

    if len(relevant_anchor) > 0:
        relevant_pred = torch.cat([rp for rp in relevant_pred]).view((1, -1, 1, 1))
        relevant_anchor = torch.cat([ra for ra in relevant_anchor]).view((1, -1, 1, 1))
        pred_midpoint = anchor_offset_to_midpoint_offset(relevant_pred, relevant_anchor)
        pred_vertices = midpoint_offset_to_vertices(pred_midpoint)
        pred_vertices = pred_vertices.view((-1, 2))
        pred_vertices = pred_vertices.reshape((len(pos_anchor_idx), 4, 2))
    else:
        pred_vertices = torch.Tensor()

    anchors_vertices = flat_anchors[0][pos_anchor_idx]
    anchors_x_min = anchors_vertices[:, 0] - anchors_vertices[:, 2] / 2
    anchors_y_min = anchors_vertices[:, 1] - anchors_vertices[:, 3] / 2
    anchors_x_max = anchors_vertices[:, 0] + anchors_vertices[:, 2] / 2
    anchors_y_max = anchors_vertices[:, 1] + anchors_vertices[:, 3] / 2
    a_1 = torch.stack((anchors_x_min, anchors_y_min), dim=1)
    a_2 = torch.stack((anchors_x_max, anchors_y_min), dim=1)
    a_3 = torch.stack((anchors_x_max, anchors_y_max), dim=1)
    a_4 = torch.stack((anchors_x_min, anchors_y_max), dim=1)
    anchors_vertices = torch.stack((a_1, a_2, a_3, a_4), dim=1)
    return relevant_gt, pred_vertices, anchors_vertices

def topk_pred_to_vertices(anchors: torch.Tensor, ground_truth: torch.Tensor, regression: torch.Tensor, scale: float, objectness: torch.Tensor, k=50, cutoff = 0.5):
    flat_regression = flatten_regression(regression.unsqueeze(0)) * scale
    flat_anchors = flatten_anchors(anchors.unsqueeze(0)) * scale
    flat_objectness = flatten_objectness(objectness.unsqueeze(0))
    mask = (flat_objectness[0] >= cutoff).squeeze()
    flat_regression = flat_regression[0][mask]
    flat_anchors = flat_anchors[0][mask]
    flat_objectness = flat_objectness[0][mask]
    
    positives_idx = torch.topk(flat_objectness.squeeze(), min(k, len(flat_objectness)), dim=0).indices.squeeze()
    positives_idx = (positives_idx, torch.arange(len(ground_truth)).repeat(k).to(flat_objectness.device))
    relevant_gt = ground_truth[positives_idx[1]]
    relevant_pred = flat_regression[positives_idx[0]]
    relevant_anchor = flat_anchors[positives_idx[0]]

    if len(relevant_anchor) > 0:
        # special case -> only one prediction
        if len(relevant_anchor.shape) == 1:
            relevant_pred = relevant_pred.unsqueeze(0)
            relevant_anchor = relevant_anchor.unsqueeze(0)

        relevant_pred = torch.cat([rp for rp in relevant_pred]).view((1, -1, 1, 1))
        relevant_anchor = torch.cat([ra for ra in relevant_anchor]).view((1, -1, 1, 1))
        pred_midpoint = anchor_offset_to_midpoint_offset(relevant_pred, relevant_anchor)
        pred_vertices = midpoint_offset_to_vertices(pred_midpoint)
        pred_vertices = pred_vertices.view((-1, 2))
        pred_vertices = pred_vertices.reshape((min(k, len(flat_objectness)), 4, 2))
    else:
        pred_vertices = torch.Tensor()

    anchors_vertices = flat_anchors[positives_idx[0]]
    if len(anchors_vertices.shape) == 1:
        anchors_vertices = anchors_vertices.unsqueeze(0)

    anchors_x_min = anchors_vertices[:, 0] - anchors_vertices[:, 2] / 2
    anchors_y_min = anchors_vertices[:, 1] - anchors_vertices[:, 3] / 2
    anchors_x_max = anchors_vertices[:, 0] + anchors_vertices[:, 2] / 2
    anchors_y_max = anchors_vertices[:, 1] + anchors_vertices[:, 3] / 2
    a_1 = torch.stack((anchors_x_min, anchors_y_min), dim=1)
    a_2 = torch.stack((anchors_x_max, anchors_y_min), dim=1)
    a_3 = torch.stack((anchors_x_max, anchors_y_max), dim=1)
    a_4 = torch.stack((anchors_x_min, anchors_y_max), dim=1)
    anchors_vertices = torch.stack((a_1, a_2, a_3, a_4), dim=1)
    return relevant_gt, pred_vertices, anchors_vertices
