import cv2
import numpy as np
import torch
import os
import json
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from ..oriented_rpn import rpn_loss, FPNAnchorGenerator, pred_to_vertices_by_gt, topk_pred_to_vertices
from ..model import OrientedRCNN

if __name__ == "__main__":
    device = "cuda"
    dir = os.path.dirname(os.path.abspath(__file__))
    image = cv2.imread(os.path.join(dir, "test_img.tif"), cv2.IMREAD_UNCHANGED)
    h, w, _ = image.shape

    with open(os.path.join(dir, "test_gt2.json"), "r") as fp:
        ground_truth = json.load(fp)
        ground_truth["boxes"] = torch.tensor(ground_truth["boxes"])[:, :4]
        ground_truth["labels"] = torch.tensor(ground_truth["labels"], dtype=torch.float)

    x = torch.tensor(image).permute((2, 0, 1)).unsqueeze(0)[:, :3, :, :] / 255
    backbone = resnet_fpn_backbone('resnet18', pretrained=False, norm_layer=None, trainable_layers=5).to(device)
    backbone.train()
    model = OrientedRCNN(backbone, {}).to(device)

    gt = ground_truth["boxes"].unsqueeze(0).to(device)
    optimizer = torch.optim.Adam([
        {"params": model.parameters()}
        ], lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.1)

    running_loss = 0
    model.train()

    for i in range(100000):
        optimizer.zero_grad()
        out = model(x.to(device))
        rpn_out = out["proposals"]
        feat = out["feat"]
        pred = out["pred"]
        anchors = out["anchors"]

        loss_dict = {}

        for (k, v), ds in zip(rpn_out.items(), [4, 8, 16, 32, 64]):
            b_loss_dict = rpn_loss(
                    v["anchor_offsets"], 
                    v["objectness_scores"], 
                    anchors[k].to(device), 
                    gt,
                    ds
            )
            for k, v in b_loss_dict.items():
                loss_dict.setdefault(k, [])
                loss_dict[k].append(v)

        loss = torch.stack(loss_dict["loss"]).sum()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        scheduler.step()
        
        if i % 100 == 0:
            last_loss = running_loss / 100 # loss every 10 epochs
            print('  epoch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

        if i % 1000 == 0:
            image_grid = cv2.imread(os.path.join(dir, "test_img.tif"), cv2.IMREAD_UNCHANGED)
            for k, scale in zip(rpn_out.keys(), [4, 8, 16, 32, 64]):
                curr_image = cv2.imread(os.path.join(dir, "test_img.tif"), cv2.IMREAD_UNCHANGED)
                regression = rpn_out[k]['anchor_offsets']
                objectness = rpn_out[k]['objectness_scores']
                #gt_coords, out_coords, anchor_coords = pred_to_vertices_by_gt(
                #        anchors[k][0].to(device), gt[0], regression[0], scale#, objectness[0]
                #)
                gt_coords, out_coords, anchor_coords = topk_pred_to_vertices(
                        anchors[k][0].to(device), gt[0], regression[0], scale, objectness[0]
                )


                thickness = 1
                isClosed = True
                gt_color = (255, 0, 0)
                pred_color = (0, 0, 255)
                a_color = (0, 255, 0)

                for g in gt_coords:
                    pts_gt = g.cpu().detach().numpy().astype(np.int32)
                    curr_image = cv2.polylines(curr_image, [pts_gt], isClosed, gt_color, thickness)

                for o in out_coords:
                    pts_pred = o.cpu().detach().numpy().astype(np.int32)
                    curr_image = cv2.polylines(curr_image, [pts_pred], isClosed, pred_color, thickness)

                for a in anchor_coords:
                    pts_anchors = a.cpu().detach().numpy().astype(np.int32)
                    curr_image = cv2.polylines(curr_image, [pts_anchors], isClosed, a_color, thickness)

                image_grid = np.concatenate((image_grid, curr_image), axis=1)

            cv2.imwrite(f"pred_{i}.png", image_grid)
