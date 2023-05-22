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
    pass

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
