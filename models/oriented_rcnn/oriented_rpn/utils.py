import torch

from ..encodings import anchor_offset_to_midpoint_offset, midpoint_offset_to_vertices
from .loss import flatten_anchors, flatten_regression, rpn_anchor_iou, get_positives_mask, flatten_objectness

def get_coords_of_ground_truth_and_output(anchors: torch.Tensor, ground_truth: torch.Tensor, regression: torch.Tensor, scale: float, objectness: torch.Tensor):
    flat_regression = flatten_regression(regression.unsqueeze(0)) * scale
    flat_anchors = flatten_anchors(anchors.unsqueeze(0)) * scale
    flat_objectness = flatten_objectness(objectness)
    num_anchors = len(flat_anchors[0])
    losses = []
    iou = rpn_anchor_iou(flat_anchors[0], ground_truth)
    positives_idx = torch.topk(flat_objectness[0].squeeze(), 10, dim=0).indices.squeeze()#get_positives_mask(iou)
    positives_idx = (positives_idx, torch.arange(len(ground_truth)).repeat(50).to(flat_objectness.device))
    relevant_gt = ground_truth[positives_idx[1]]
    relevant_pred = flat_regression[0][positives_idx[0]]
    relevant_anchor = flat_anchors[0][positives_idx[0]]

    relevant_pred = torch.cat([rp for rp in relevant_pred]).view((1, -1, 1, 1))
    relevant_anchor = torch.cat([ra for ra in relevant_anchor]).view((1, -1, 1, 1))
    pred_midpoint = anchor_offset_to_midpoint_offset(relevant_pred, relevant_anchor)
    pred_vertices = midpoint_offset_to_vertices(pred_midpoint)
    pred_vertices = pred_vertices.view((-1, 2))
    num_coords = len(relevant_gt)
    pred_vertices = pred_vertices.reshape((10, 4, 2))
    anchors_vertices = flat_anchors[0][positives_idx[0]]
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
