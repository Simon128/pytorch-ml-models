from collections import OrderedDict
import torch

def anchor_offset_to_midpoint_offset(anchor_offset: torch.Tensor, anchors: torch.Tensor):
    b, n, h, w = anchors.shape
    num_anchors = int(n/4)
    # prediction has 6 * num_anchors in dim=1 (they are concatenated) we reshape 
    # for easier handling (same for anchors)
    r_offset = anchor_offset.reshape((b, num_anchors, 6, h, w))
    r_anchors = anchors.reshape((b, num_anchors, 4, h, w))

    w = r_anchors[:, :, 2, :, :] * torch.exp(r_offset[:, :, 2, :, :])
    h = r_anchors[:, :, 3, :, :] * torch.exp(r_offset[:, :, 3, :, :])
    x = r_offset[:, :, 0, :, :] * r_anchors[:, :, 2, :, :] + r_anchors[:, :, 0, :, :]
    y = r_offset[:, :, 1, :, :] * r_anchors[:, :, 3, :, :] + r_anchors[:, :, 1, :, :]
    delta_alpha = r_offset[:, :, 4, :, :] * w
    delta_beta = r_offset[:, :, 5, :, :] * h

    r_midpoint_offset = torch.stack((x, y, w, h, delta_alpha, delta_beta), dim=2)
    return torch.cat([r_midpoint_offset[:, i, :, :, :] for i in range(num_anchors)], dim=1).float()

def midpoint_offset_to_anchor_offset(midpoint_offset: torch.tensor, anchors: torch.tensor):
    b, n, h, w = anchors.shape
    num_anchors = int(n/4)
    # reshape for easier handling
    r_midpoint_offset = midpoint_offset.reshape((b, num_anchors, 6, h, w))
    r_anchors = anchors.reshape((b, num_anchors, 4, h, w))

    d_a = r_midpoint_offset[:, :, 4, :, :] / r_midpoint_offset[:, :, 2, :, :]
    d_b = r_midpoint_offset[:, :, 5, :, :] / r_midpoint_offset[:, :, 3, :, :]
    d_w = torch.log(r_midpoint_offset[:, :, 2, :, :] / r_anchors[:, :, 2, :, :])
    d_h = torch.log(r_midpoint_offset[:, :, 3, :, :] / r_anchors[:, :, 3, :, :])
    d_x = (r_midpoint_offset[:, :, 0, :, :] - r_anchors[:, :, 0, :, :]) / r_anchors[:, :, 2, :, :]
    d_y = (r_midpoint_offset[:, :, 1, :, :] - r_anchors[:, :, 1, :, :]) / r_anchors[:, :, 3, :, :]

    r_anchor_offset = torch.stack((d_x, d_y, d_w, d_h, d_a, d_b), dim=2)
    return torch.cat([r_anchor_offset[:, i, :, :, :] for i in range(num_anchors)], dim=1).float()

def midpoint_offset_to_anchor_offset_gt(midpoint_offset_gt: torch.tensor, tp_anchors: torch.tensor):
    num_anchors = len(tp_anchors)
    d_a = midpoint_offset_gt[:, 4] / midpoint_offset_gt[:, 2]
    d_b = midpoint_offset_gt[:, 5] / midpoint_offset_gt[:, 3]
    d_w = torch.log(midpoint_offset_gt[:, 2] / tp_anchors[:, 2])
    d_h = torch.log(midpoint_offset_gt[:, 3] / tp_anchors[:, 3])
    d_x = (midpoint_offset_gt[:, 0] - tp_anchors[:, 0]) / tp_anchors[:, 2]
    d_y = (midpoint_offset_gt[:, 1] - tp_anchors[:, 1]) / tp_anchors[:, 3]
    return torch.stack((d_x, d_y, d_w, d_h, d_a, d_b), dim=1)

def midpoint_offset_to_vertices(midpoint_offset: torch.Tensor):
    b, n, h, w = midpoint_offset.shape
    num_anchors = int(n/6)
    # prediction has 6 * num_anchors in dim=1 (they are concatenated) we reshape 
    # for easier handling 
    r_midpoint_offset = midpoint_offset.reshape((b, num_anchors, 6, h, w))
    
    x = r_midpoint_offset[:, :, 0, :, :]
    y = r_midpoint_offset[:, :, 1, :, :]
    w = r_midpoint_offset[:, :, 2, :, :]
    h = r_midpoint_offset[:, :, 3, :, :]
    d_alpha = r_midpoint_offset[:, :, 4, :, :]
    d_beta = r_midpoint_offset[:, :, 5, :, :]

    v1 = torch.stack([x + d_alpha, y - h / 2], dim=2)
    v2 = torch.stack([x + w / 2, y + d_beta], dim=2)
    v3 = torch.stack([x - d_alpha, y + h / 2], dim=2)
    v4 = torch.stack([x - w / 2, y - d_beta], dim=2)

    r_vertices = torch.stack((v1, v2, v3, v4), dim=2)
    return torch.cat([r_vertices[:, i, :, :, :, :] for i in range(num_anchors)], dim=1).float()

def vertices_to_midpoint_offset(vertices: torch.Tensor):
    # vertices shape: b, num_anchors * 4, 2, H, W
    b, n, _, h, w = vertices.shape
    num_anchors = int(n/4)
    # reshape for easier handling
    r_vertices = vertices.reshape((b, num_anchors, 4, 2, h, w))

    x_min = torch.min(r_vertices[:, :, :, 0, :, :], dim=2)[0]
    x_max = torch.max(r_vertices[:, :, :, 0, :, :], dim=2)[0]
    y_min = torch.min(r_vertices[:, :, :, 1, :, :], dim=2)[0]
    y_max = torch.max(r_vertices[:, :, :, 1, :, :], dim=2)[0]
    
    w = x_max - x_min
    h = y_max - y_min
    x_center = x_min + w / 2
    y_center = y_min + h / 2
    delta_a = r_vertices[:, :, 0, 0, :, :] - x_center
    delta_b = r_vertices[:, :, 1, 1, :, :] - y_center

    r_midpoint_offset = torch.stack((x_center, y_center, w, h, delta_a, delta_b), dim=2)
    return torch.cat([r_midpoint_offset[:, i, :, :, :] for i in range(num_anchors)], dim=1)

def vertices_to_midpoint_offset_gt(vertices: torch.Tensor):
    # vertices shape: n, 4, 2
    n, _, _ = vertices.shape

    x_min = torch.min(vertices[:, :, 0], dim=1)[0]
    x_max = torch.max(vertices[:, :, 0], dim=1)[0]
    y_min = torch.min(vertices[:, :, 1], dim=1)[0]
    y_max = torch.max(vertices[:, :, 1], dim=1)[0]
    
    # assuming clockwise 
    # (argmin returns first idx)
    top_left_idx = (torch.arange(n), torch.argmin(vertices[:, :, 1], dim=1))
    cl_next_idx = (top_left_idx[0], (top_left_idx[1] + 1) % 4)
    w = x_max - x_min
    h = y_max - y_min
    x_center = x_min + w / 2
    y_center = y_min + h / 2
    delta_a = vertices[top_left_idx][:, 0] - x_center
    delta_b = vertices[cl_next_idx][:, 1] - y_center

    return torch.stack((x_center, y_center, w, h, delta_a, delta_b), dim=1)
