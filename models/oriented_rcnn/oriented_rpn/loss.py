import torch
from torchvision.ops import box_iou

def rpn_anchor_iou(anchors: torch.Tensor, ground_truth: torch.Tensor):
    assert len(ground_truth.shape) == 4, "expected ground truth shape b, n, 4, 2"
    # anchors are (x, y, w, h) with x,y being the center of the anchor
    # we need to convert to vertices for torchvision box_iou
    b, n, h, w = anchors.shape
    num_anchors = int(n/4)
    # reshape for easier handling
    r_anchors = anchors.reshape((b, num_anchors, 4, h, w))

    anchor_x1 = r_anchors[:, :, 0, :, :] - r_anchors[:, :, 2, :, :] / 2
    anchor_x2 = r_anchors[:, :, 0, :, :] + r_anchors[:, :, 2, :, :] / 2
    anchor_y1 = r_anchors[:, :, 1, :, :] - r_anchors[:, :, 3, :, :] / 2
    anchor_y2 = r_anchors[:, :, 1, :, :] + r_anchors[:, :, 3, :, :] / 2
    
    r_anchor_boxes = torch.stack((anchor_x1, anchor_y1, anchor_x2, anchor_y2), dim=2)
    anchor_boxes = r_anchor_boxes.permute((0, 1, 3, 4, 2)).flatten(start_dim=0, end_dim=3)
    print(anchor_boxes[0])

    # ground truth shape: b, n, 4, 2
    gt_x1 = torch.min(ground_truth[:, :, :, 0], dim=2)[0]
    gt_x2 = torch.max(ground_truth[:, :, :, 0], dim=2)[0]
    gt_y1 = torch.min(ground_truth[:, :, :, 1], dim=2)[0]
    gt_y2 = torch.max(ground_truth[:, :, :, 1], dim=2)[0]
    
    r_gt_boxes = torch.stack((gt_x1, gt_y1, gt_x2, gt_y2), dim=2)
    gt_boxes = r_gt_boxes.flatten(start_dim=0, end_dim=1)
    return box_iou(anchor_boxes, gt_boxes)
