import torch
from pytorch3d.ops import box3d_overlap

import numpy as np



def offsets_to_proposal(anchor: torch.Tensor, offset: torch.Tensor):
    # offset: (δx, δy , δw, δh, δα, δβ)
    # anchor: (ax, ay , aw, ah)
    # w = aw · e^{δw}
    w = anchor[:, :, 2, :, :] * torch.exp(offset[:, :, 2, :, :])
    # deviation from original paper: preventing crooked rectangles
    w = torch.maximum(w, torch.ones_like(w))
    # h = ah · eδh
    h = anchor[:, :, 3, :, :] * torch.exp(offset[:, :, 3, :, :])
    # deviation from original paper: preventing crooked rectangles
    h = torch.maximum(h, torch.ones_like(h))
    # ∆α = δα · w 
    delta_alpha = offset[:, :, 4, :, :] * w
    # deviation from original paper: preventing crooked rectangles
    delta_alpha = torch.maximum(delta_alpha, w/2)
    delta_alpha = torch.minimum(delta_alpha, -w/2)
    # ∆β = δβ · h 
    delta_beta = offset[:, :, 5, :, :] * h
    # deviation from original paper: preventing crooked rectangles
    delta_beta = torch.maximum(delta_beta, h/2)
    delta_beta = torch.minimum(delta_beta, -h/2)
    # x = δx · aw + ax
    # my anchor generation deviates from the paper
    # the paper uses x,y midpoints, I'm using top left
    # so I'll have to add w/2 and h/2 resp.
    x = offset[:, :, 0, :, :] * anchor[:, :, 2, :, :] + anchor[:, :, 0, :, :] + anchor[:, :, 2, :, :]/2
    # y = δy · ah + ay
    y = offset[:, :, 1, :, :] * anchor[:, :, 3, :, :] + anchor[:, :, 1, :, :] + anchor[:, :, 2, :, :]/2
    # out (x, y, w, h, ∆α, ∆β)
    return torch.stack((x, y, w, h, delta_alpha, delta_beta), dim=2)
    
    boxes = midpoint_offset_representation_to_coords(torch.stack((x, y, w, h, delta_alpha, delta_beta), dim=2))
    d3_boxes = fake_2d_to_3d(boxes)
        

    test = d3_boxes[0,0,:,:,0,0]
    shape1 = d3_boxes[:,0,:,:,0,0].shape
    shape2 = d3_boxes[0,:,:,:,0,0].shape
    shape3 = d3_boxes[0,0,:,:,:,0].shape
    shape5 = d3_boxes[0,0,:,:,:,0].permute((2, 0, 1)).shape
    shape4 = d3_boxes[0,0,:,:,0,:].shape

    check = d3_boxes.permute((0, 1, 4, 5, 2, 3))
    check = check.flatten(0, 3)
    
    for idx, b in enumerate(check):
        assert b[0, 2].detach().item() == 0
        assert b[1, 2].detach().item() == 0
        assert b[2, 2].detach().item() == 0
        assert b[3, 2].detach().item() == 0
        assert b[4, 2].detach().item() == 1
        assert b[5, 2].detach().item() == 1
        assert b[6, 2].detach().item() == 1
        assert b[7, 2].detach().item() == 1
    
    test2 = check[0]
    # for x in check:
    #     _check_coplanar(x.unsqueeze(0))
    #return torch.stack((x, y, w, h, delta_alpha, delta_beta), dim=2)

    
    test_anchor = anchor[0, 1, :, 0, 0].detach().cpu().numpy()
    new_outer_box = torch.stack((x, y, w, h, delta_alpha, delta_beta), dim=2)[0,1, :, 0, 0].detach().cpu().numpy()

    x_ta = [test_anchor[0], test_anchor[0] + test_anchor[2], test_anchor[0] + test_anchor[2], test_anchor[0], test_anchor[0]]
    y_ta = [test_anchor[1], test_anchor[1], test_anchor[1] + test_anchor[3], test_anchor[1] + test_anchor[3], test_anchor[1]]

    half_width = new_outer_box[2]/2
    half_height = new_outer_box[3]/2
    x_l = new_outer_box[0] - half_width
    x_r = new_outer_box[0] + half_width
    y_t = new_outer_box[1] - half_height
    y_b = new_outer_box[1] + half_height

    x_out = [x_l, x_r, x_r, x_l, x_l]
    y_out = [y_t, y_t, y_b, y_b, y_t]

    x_out_mid = new_outer_box[0]
    da = new_outer_box[4]

    test2= midpoint_offset_representation_to_coords(torch.stack((x, y, w, h, delta_alpha, delta_beta), dim=2))[0,1, :, :, 0, 0].detach().cpu().numpy()

    pts2_x = [t[0] for t in test2] + [test2[0][0]]
    pts2_y = [t[1] for t in test2] + [test2[0][1]]

    
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_ta, y_ta, linestyle="dotted", color="black")
    ax.plot(x_out, y_out, linestyle="dashed", color="blue")
    ax.plot(pts2_x, pts2_y, linestyle="solid", color="red")
    ax.plot([x_out_mid, x_out_mid + da], [y_t, y_t], color="gray", marker="x")

    # ax.plot([mid_x, mid_x - da], [test[1] + test[3], test[1] + test[3]], color="green", marker="x")
    # ax.plot([test[0] + test[2], test[0] + test[2]], [test[1] + test[3]/2, test[1] + test[3]/2 + db], color="yellow", marker="x")
    # ax.plot([test[0], test[0]], [test[1] + test[3]/2, test[1] + test[3]/2 - db], color="yellow", marker="x")
    
    plt.gca().invert_yaxis()
    plt.savefig("test.png")
    a = 5

def draw_rotated_2d_box(box, filename):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    fig = plt.figure()
    ax = fig.add_subplot(111)
    box = box.detach().cpu().numpy()
    ptsx = [b[0] for b in box[:4]] + [box[0][0]]
    ptsy = [b[1] for b in box[:4]] + [box[0][1]]
    ax.plot(ptsx, ptsy , linestyle="dotted", color="blue")
    ax.scatter(ptsx, ptsy, marker="x", color="red")
    plt.gca().invert_yaxis()
    plt.savefig(filename)


def midpoint_offset_representation_to_coords(repr: torch.Tensor):
    # input: bhw with (x, y, w, h, ∆α, ∆β)
    x = repr[:, :, 0, :, :]
    y = repr[:, :, 1, :, :] 
    w = repr[:, :, 2, :, :]
    h = repr[:, :, 3, :, :]
    delta_a = repr[:, :, 4, :, :]
    delta_b = repr[:, :, 5, :, :]
    v1 = torch.stack([x + delta_a, y - h/2], dim=2)#[:, :, 0, 0]
    v2 = torch.stack(( x + w/2, y + delta_b), dim=2)#[:, :, 0, 0]
    v3 = torch.stack((x - delta_a, y + h/2), dim=2)#[:, :, 0, 0]
    v4 = torch.stack((x - w/2, y - delta_b), dim=2)#[:, :, 0, 0]

    result = torch.stack((v1, v2, v3, v4), dim=2)
    #stacked = repr.permute((0, 1, 3, 4, 2)).flatten(0, 3)

    # for r in stacked:
    #     x = r[0]
    #     y = r[1]
    #     w = r[2]
    #     h = r[3]
    #     delta_a = r[4] # both negative
    #     delta_b = r[5] 
    #     v1 = torch.tensor([x + delta_a, y - h/2])
    #     v2 = torch.tensor(( x + w/2, y + delta_b))
    #     v3 = torch.tensor((x - delta_a, y + h/2))
    #     v4 = torch.tensor((x - w/2, y - delta_b))
    #     test = torch.stack((v1, v2, v3, v4), dim=0)
    #     d3_box = fake_2d_to_3d(test.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    #     d3_box = d3_box.squeeze().unsqueeze(0)
    #     try:
    #         _check_coplanar(d3_box, eps=0.01)
    #         _check_nonzero(d3_box, 0.01)
    #     except:
    #         test = 5

    return result

    

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
    b, ar, v, c, h, w = boxes.shape

    # just repeat 2d rectangle to create a 3d box with 0 depth
    result = boxes.repeat(1, 1, 2, 1, 1, 1)
    # set depth of boxes to 1 (we don't care about volume anyway)
    dim = (b, ar, v, h, w)
    one_depth = torch.concat((torch.zeros(dim), torch.ones((dim))), dim=2).to(boxes.device)
    result = torch.concat((result, one_depth.unsqueeze(3)), dim=3)
    return result

def rotated_iou(boxes1: torch.tensor, boxes2: torch.tensor):
    # using 3d for 2d for speed and convenience https://pytorch3d.org/docs/iou3d
    b = boxes1.shape[0]
    d3_boxes1 = fake_2d_to_3d(boxes1)
    d3_boxes2 = fake_2d_to_3d(boxes2.unsqueeze(-1).unsqueeze(-1))

    stacked_boxes1 = d3_boxes1.permute((0, 1, 4, 5, 2, 3)).flatten(1, 3)
    stacked_boxes2 = d3_boxes2.permute((0, 1, 4, 5, 2, 3)).flatten(1, 3)
    result = []

    # for x in stacked_boxes1[0]:
    #     #_check_coplanar(x.unsqueeze(0))
    #     _check_nonzero(x.unsqueeze(0), eps=0.01)

    # for x in stacked_boxes2[0]:
    #     _check_coplanar(x.unsqueeze(0))
    #   _check_nonzero(x.unsqueeze(0))
    
    for idx in range(b):
        # predicted boxes can have comically large coordinates
        # in the beginning of the training, so we increase eps to 0.1
        _, iou = box3d_overlap(stacked_boxes1[idx], stacked_boxes2[idx], eps=0.01)
        result.append(iou)

    return torch.stack(result)


_box_planes = [
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [4, 5, 6, 7],
]
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


def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    import torch.nn.functional as F

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)
    if not (mat1.bmm(mat2).abs() < eps).all().item():

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        box = boxes.detach().cpu().numpy()[0]
        ptsx = [b[0] for b in box[:4]] + [box[0][0]]
        ptsy = [b[1] for b in box[:4]] + [box[0][1]]
        ptsz = [b[2] for b in box[:4]] + [box[0][2]]

        ptsx2 = [b[0] for b in box[4:]] + [box[4][0]]
        ptsy2 = [b[1] for b in box[4:]] + [box[4][1]]
        ptsz2 = [b[2] for b in box[4:]] + [box[4][2]]

        #ax.plot(x_ta, y_ta, linestyle="dotted", color="black")
        #ax.plot(x_out, y_out, linestyle="dashed", color="blue")
        ax.plot(ptsx, ptsy , linestyle="solid", color="red")
        #ax.plot(ptsx2, ptsz2, ptsy2 , linestyle="solid", color="blue")
        #ax.plot([x_out_mid, x_out_mid + da], [y_t, y_t], color="gray", marker="x")

        # ax.plot([mid_x, mid_x - da], [test[1] + test[3], test[1] + test[3]], color="green", marker="x")
        # ax.plot([test[0] + test[2], test[0] + test[2]], [test[1] + test[3]/2, test[1] + test[3]/2 + db], color="yellow", marker="x")
        # ax.plot([test[0], test[0]], [test[1] + test[3]/2, test[1] + test[3]/2 - db], color="yellow", marker="x")
        
        plt.gca().invert_yaxis()
        plt.savefig("test.png")
        a = 5
        msg = "Plane vertices are not coplanar"
        raise ValueError(msg)

    return


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
        draw_rotated_2d_box(boxes[0].detach().cpu(), "test.png")
        msg = "Planes have zero areas"
        raise ValueError(msg)

    return