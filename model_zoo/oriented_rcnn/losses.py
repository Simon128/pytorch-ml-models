import torch
import torchvision

from .utils import offsets_to_proposal, midpoint_offset_representation_to_coords, rotated_iou

def rpn_loss(anchors: torch.Tensor, anchor_offsets: torch.Tensor, objectness_score: torch.Tensor, ground_truth_boxes: torch.Tensor):
    b, _, h, w = anchor_offsets.shape
    proposals = offsets_to_proposal(anchors.reshape((b, -1, 4, h ,w)), anchor_offsets.reshape(b, -1, 6, h, w))
    proposals = midpoint_offset_representation_to_coords(proposals)
    # dims: b, 2 (x,y), 4 (vertices), h, w
    proposals = proposals.view((b, -1, 4, 2))
    #b, c, h, w = proposals.shape
    iou = rotated_iou(proposals, ground_truth_boxes)
    batch_size = anchors.shape[0]
    loss = 1/batch_size * () + 1/batch_size * ()

    