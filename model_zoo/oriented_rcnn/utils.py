import torch
from pytorch3d.ops import box3d_overlap

import numpy as np



def offsets_to_proposal(anchor: torch.Tensor, offset: torch.Tensor):
    # offset: (δx, δy , δw, δh, δα, δβ)
    # anchor: (ax, ay , aw, ah)
    # w = aw · e^{δw}
    w = anchor[:, :, 2, :, :] * torch.exp(offset[:, :, 2, :, :])
    # h = ah · eδh
    h = anchor[:, :, 3, :, :] * torch.exp(offset[:, :, 3, :, :])
    # ∆α = δα · w 
    delta_alpha = offset[:, :, 4, :, :] * w
    # ∆β = δβ · h 
    delta_beta = offset[:, :, 5, :, :] * h
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
    
    for idx in range(b):
        _, iou = box3d_overlap(stacked_boxes1[idx], stacked_boxes2[idx])
        result.append(iou)

    return torch.stack(result)