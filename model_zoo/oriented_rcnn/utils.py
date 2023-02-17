import torch
from pytorch3d.ops import box3d_overlap

def offsets_to_proposal(anchor: torch.Tensor, offset: torch.Tensor):
    # offset: (δx, δy , δw, δh, δα, δβ)
    # anchor: (ax, ay , aw, ah)
    # ∆α = δα · w
    delta_alpha = offset[:, 4, :, :] * anchor[:, 2, :, :]
    # ∆β = δβ · h
    delta_beta = offset[:, 5, :, :] * anchor[:, 3, :, :]
    # w = aw · e^{δw}
    w = anchor[:, 2, :, :] * torch.exp(offset[:, 2, :, :])
    # h = ah · eδh
    h = anchor[:, 3, :, :] * torch.exp(offset[:, 3, :, :])
    # x = δx · aw + ax
    x = offset[:, 0, :, :] * anchor[:, 2, :, :] + anchor[:, 0, :, :]
    # y = δy · ah + ay
    y = offset[:, 1, :, :] * anchor[:, 3, :, :] + anchor[:, 1, :, :]
    # out (x, y, w, h, ∆α, ∆β)
    return torch.stack((x, y, w, h, delta_alpha, delta_beta), dim=1)

def midpoint_offset_representation_to_coords(repr: torch.Tensor):
    # input: bhw with (x, y, w, h, ∆α, ∆β)
    x = repr[:, 0, :, :]
    y = repr[:, 1, :, :] 
    w = repr[:, 2, :, :]
    h = repr[:, 3, :, :]
    delta_a = repr[:, 4, :, :]
    delta_b = repr[:, 5, :, :]
    v1 = torch.tensor([x + delta_a, y - h/2])
    v2 = (x + w/2, y + delta_b)
    v3 = (x - delta_a, y + h/2)
    v4 = (x - w/2, y - delta_b)
    return torch.stack((v1, v2, v3, v4), dim=1)

def rotated_iou(boxes1: torch.tensor, boxes2: torch.tensor):
    # using 3d for 2d for speed and convenience https://pytorch3d.org/docs/iou3d
    test = 5
    # box3d_overlap