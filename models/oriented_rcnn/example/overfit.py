import cv2
import numpy as np
import torch
import os
import json
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from ..oriented_rpn import OrientedRPN, rpn_loss, FPNAnchorGenerator, get_positives_mask, get_coords_of_ground_truth_and_output

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
    model = OrientedRPN({}).to(device)
    backbone = resnet_fpn_backbone('resnet50', pretrained=True, norm_layer=None, trainable_layers=3).to(device)
    backbone.train()

    gt = ground_truth["boxes"].unsqueeze(0).to(device)
    optimizer = torch.optim.Adam([
        {"params": backbone.parameters()}, 
        {"params": model.parameters()}
        ], lr=0.0001)
    anchor_generator = FPNAnchorGenerator(sqrt_size_per_level=(16, 32, 64, 128, 256))

    running_loss = 0
    model.train()

    for i in range(10000):
        optimizer.zero_grad()

        feat = backbone(x.to(device))
        out = model(feat)
        out_ao = {k: v['anchor_offsets'] for k, v in out.items()}
        out_os = {k: v['objectness_scores'] for k, v in out.items()}
        anchors = anchor_generator.generate_like_fpn(feat, w, h)

        loss = 0

        for (k, v), ds in zip(out.items(), [4, 8, 16, 32, 64]):
            loss = loss + rpn_loss(
                    v["anchor_offsets"], 
                    v["objectness_scores"], 
                    anchors[k].to(device), 
                    gt,
                    ds
            )

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 100 == 0:
            last_loss = running_loss / 10 # loss every 10 epochs
            print('  epoch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

        if i % 5000 == 0:
            for k, scale in zip(out.keys(), [4, 8, 16, 32, 64]):
                regression = out[k]['anchor_offsets']
                gt_coords, out_coords, anchor_coords = get_coords_of_ground_truth_and_output(
                        anchors[k][0].to(device), gt[0], regression[0], scale
                )

                for g, o, a in zip(gt_coords, out_coords, anchor_coords):
                    pts_gt = g.cpu().detach().numpy().astype(np.int32)
                    pts_pred = o.cpu().detach().numpy().astype(np.int32)
                    pts_anchors = a.cpu().detach().numpy().astype(np.int32)
                    gt_color = (255, 0, 0)
                    pred_color = (0, 0, 255)
                    a_color = (0, 255, 0)
                    thickness = 1
                    isClosed = True
                    image = cv2.polylines(image, [pts_gt], isClosed, gt_color, thickness)
                    image = cv2.polylines(image, [pts_pred], isClosed, pred_color, thickness)
                    image = cv2.polylines(image, [pts_anchors], isClosed, a_color, thickness)

            cv2.imshow("test", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
