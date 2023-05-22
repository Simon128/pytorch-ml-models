from collections import OrderedDict
import torch

def rpn_anchor_offset_to_midpoint_offset(prediction: torch.Tensor, anchors: torch.Tensor):
    b, n, h, w = anchors.shape
    num_anchors = int(n/4)
    # prediction has 6 * num_anchors in dim=1 (they are concatenated) we reshape 
    # for easier handling (same for anchors)
    r_pred = prediction.reshape((b, num_anchors, 6, h, w))
    r_anchors = anchors.reshape((b, num_anchors, 4, h, w))

    w = r_anchors[:, :, 2, :, :] * torch.exp(r_pred[:, :, 2, :, :])
    h = r_anchors[:, :, 3, :, :] * torch.exp(r_pred[:, :, 3, :, :])
    x = r_pred[:, :, 0, :, :] * r_anchors[:, :, 2, :, :] + r_anchors[:, :, 0, :, :]
    y = r_pred[:, :, 1, :, :] * r_anchors[:, :, 3, :, :] + r_anchors[:, :, 1, :, :]
    delta_alpha = r_pred[:, :, 4, :, :] * w
    delta_beta = r_pred[:, :, 5, :, :] * h

    r_midpoint_offset = torch.stack((x, y, w, h, delta_alpha, delta_beta), dim=2)
    return torch.cat([r_midpoint_offset[:, i, :, :, :] for i in range(num_anchors)], dim=1).float()


def midpoint_offset_to_vertices(midpoint_offset: torch.Tensor):
    pass
