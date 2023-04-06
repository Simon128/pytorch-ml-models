import json
import cv2
import os
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch
import torchvision
from ..model import OrientedRCNN
from ..anchor_generation import AnchorGenerator
from ..losses import rpn_loss, get_tp
import numpy as np
from ..utils import offsets_to_proposal, midpoint_offset_representation_to_coords
import copy

# DETECTRON 2!

device = torch.device("cuda")

if __name__ == "__main__":
    dir = os.path.dirname(os.path.abspath(__file__))
    image = cv2.imread(os.path.join(dir, "test_img.tif"), cv2.IMREAD_UNCHANGED)
    h, w, _ = image.shape

    with open(os.path.join(dir, "test_gt2.json"), "r") as fp:
        ground_truth = json.load(fp)
        ground_truth["boxes"] = torch.tensor(ground_truth["boxes"])[:, :4]
        ground_truth["labels"] = torch.tensor(ground_truth["labels"], dtype=torch.float)

    x = torch.tensor(image).permute((2, 0, 1)).unsqueeze(0)[:, :3, :, :] / 255
    
    cfg = {

    }

    backbone = resnet_fpn_backbone('resnet50', pretrained=False, norm_layer=None, trainable_layers=5).to(device)
    model = OrientedRCNN(backbone, cfg).to(device)
    anchor_generator = AnchorGenerator(cfg, device=device)

    # test iou 
    # gt_box = torch.tensor([
    #     [0, 0],
    #     [2, 0],
    #     [2, 4],
    #     [0, 4]
    # ]).unsqueeze(0).unsqueeze(1)
    # test_box = torch.tensor([
    #     [0, 0],
    #     [1, 0],
    #     [1, 2],
    #     [0, 2]
    # ]).unsqueeze(0).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    # test = rotated_iou(test_box, gt_box)

    # [8, 8, 8, 8, 8] corresponds to [32, 64, 128, 256, 512] in orig image

    gt  = ground_truth["boxes"].unsqueeze(0).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0

    for i in range(1000):
        model.train(True)
        optimizer.zero_grad()

        out = model(x.to(device))
        anchors = anchor_generator.generate_like_fpn(out["backbone_out"], [8, 8, 8, 8, 8])

        loss = 0

        for (k, v), ds in zip(out["rpn_out"].items(), [4, 8, 16, 32, 64]):
            loss = loss + rpn_loss(anchors[k] / ds, v["anchor_offsets"], v["objectness_scores"], gt / ds)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 10 == 0:
            last_loss = running_loss / 10 # loss every 10 epochs
            print('  epoch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)6
            running_loss = 0.
            offsets = copy.deepcopy(out['rpn_out']['0']['anchor_offsets'].detach().clone().cpu().numpy())
            anchors = copy.deepcopy(anchors['0'].detach().clone().cpu().numpy())
            gt_test = copy.deepcopy(gt.detach().clone().cpu().numpy())
            offsets = torch.tensor(offsets).cuda()
            anchors = torch.tensor(anchors).cuda()
            gt_test = torch.tensor(gt_test).cuda()

            b, _ , h, w = offsets.shape
            tp_anchors, tp_offsets = get_tp(anchors, offsets, gt_test / 4)

            # if len(tp_offsets) > 1:
            #     proposals = offsets_to_proposal(tp_anchors, tp_offsets)
            #     coords = midpoint_offset_representation_to_coords(proposals)
            #     coords = coords.permute((0, 1, 4, 5, 2, 3)).flatten(1, 3)
            #     coords = coords * 4
                #cv2.polylines(image, coords)

        model.train(False)
