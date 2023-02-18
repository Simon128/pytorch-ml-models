import torch
from pytorch3d.ops import box3d_overlap

import numpy as np


def offsets_to_proposal(anchor: torch.Tensor, offset: torch.Tensor):
    # offset: (δx, δy , δw, δh, δα, δβ)
    # anchor: (ax, ay , aw, ah)
    # ∆α = δα · w
    delta_alpha = offset[:, :, 4, :, :] * anchor[:, :, 2, :, :]
    # ∆β = δβ · h
    delta_beta = offset[:, :, 5, :, :] * anchor[:, :, 3, :, :]
    # w = aw · e^{δw}
    w = anchor[:, :, 2, :, :] * torch.exp(offset[:, :, 2, :, :])
    # h = ah · eδh
    h = anchor[:, :, 3, :, :] * torch.exp(offset[:, :, 3, :, :])
    # x = δx · aw + ax
    x = offset[:, :, 0, :, :] * anchor[:, :, 2, :, :] + anchor[:, :, 0, :, :]
    # y = δy · ah + ay
    y = offset[:, :, 1, :, :] * anchor[:, :, 3, :, :] + anchor[:, :, 1, :, :]
    # out (x, y, w, h, ∆α, ∆β)
    return torch.stack((x, y, w, h, delta_alpha, delta_beta), dim=2)

def midpoint_offset_representation_to_coords(repr: torch.Tensor):
    # input: bhw with (x, y, w, h, ∆α, ∆β)
    x = repr[:, :, 0, :, :]
    y = repr[:, :, 1, :, :] 
    w = repr[:, :, 2, :, :]
    h = repr[:, :, 3, :, :]
    delta_a = repr[:, :, 4, :, :]
    delta_b = repr[:, :, 5, :, :]
    v1 = torch.stack([y - h/2, x + delta_a], dim=2)#[:, :, 0, 0]
    v2 = torch.stack((y + delta_b, x + w/2), dim=2)#[:, :, 0, 0]
    v3 = torch.stack((y + h/2, x - delta_a), dim=2)#[:, :, 0, 0]
    v4 = torch.stack((y - delta_b, x - w/2), dim=2)#[:, :, 0, 0]
    return torch.stack((v1, v2, v3, v4), dim=2)

def fake_2d_to_3d(boxes: torch.Tensor):
    '''
        box3d_overlap expects:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)
    '''
    b, c, _, _ = boxes.shape
    # just repeat 2d rectangle to create a 3d box with 0 depth
    result = boxes.repeat(1, 1, 2, 1)
    # set depth of boxes to 1 (we don't care about volume anyway)
    one_depth = torch.concat((torch.zeros((b, c, 4)), torch.ones((b, c, 4))), dim=2).to(boxes.device)
    result = torch.concat((result, one_depth.unsqueeze(-1)), dim=3)
    return result

import torch.nn.functional as F

_box_planes = [
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [4, 5, 6, 7],
]

def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    data =  boxes.squeeze().detach().clone().to("cpu").numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.concatenate((data[:4,0], [data[0,0]])), np.concatenate((data[:4,1], [data[0,1]])))
    plt.show()
    plt.waitforbuttonpress()

    if not (mat1.bmm(mat2).abs() < eps).all().item():
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        data =  boxes.squeeze().detach().clone().to("cpu").numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.concatenate((data[:4,0], [data[0,0]])), np.concatenate((data[:4,1], [data[0,1]])))
        plt.show()
        plt.waitforbuttonpress()

        msg = "Plane vertices are not coplanar"
        raise ValueError(msg)

    return

_box_triangles = [
    [0, 1, 2],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [1, 5, 6],
    [1, 6, 2],
    [0, 4, 7],
    [0, 7, 3],
    [3, 2, 6],
    [3, 6, 7],
    [0, 1, 5],
    [0, 4, 5],
]

def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    """
    Checks that the sides of the box have a non zero area
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    if (face_areas < eps).any().item():
        msg = "Planes have zero areas"
        raise ValueError(msg)

    return


def rotated_iou(boxes1: torch.tensor, boxes2: torch.tensor):
    # using 3d for 2d for speed and convenience https://pytorch3d.org/docs/iou3d

    d3_boxes1 = fake_2d_to_3d(boxes1)
    d3_boxes2 = fake_2d_to_3d(boxes2)

    for b in d3_boxes1[0]:
        _check_coplanar(b.unsqueeze(0))

    for b in d3_boxes1[0]:
        _check_nonzero(b.unsqueeze(0))

    int_area, iou = box3d_overlap(d3_boxes1[0], d3_boxes2[0])
    # https://github.com/facebookresearch/detectron2/blob/07c0910c359bb317b075f9456fa29378c8384b9a/detectron2/structures/rotated_boxes.py


    test = 5
    # box3d_overlap