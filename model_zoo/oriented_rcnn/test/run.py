import json
import cv2
import os
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch
import torchvision
from ..model import OrientedRCNN
from ..anchor_generation import AnchorGenerator
from ..losses import rpn_loss
import numpy as np

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
    x = x.repeat((2, 1, 1, 1))

    for b in ground_truth["boxes"]:
        x = torch.min(b[:, 0]).to(torch.int).cpu().item() // 2
        y = torch.min(b[:, 1]).to(torch.int).cpu().item() // 2
        xmax = torch.max(b[:, 0]).to(torch.int).cpu().item() // 2
        ymax = torch.max(b[:, 1]).to(torch.int).cpu().item() // 2
        tl = [x,y] 
        br = [xmax,ymax] 
        image = cv2.resize(image, (512, 512))
        image = cv2.rectangle(image, tl, br, (0, 255, 0), 2)

    cv2.imwrite("test.png", image)

    # boxes = []
    # labels = []

    # for oc in ground_truth:
    #     box = shapely.wkt.loads(oc["bounding_box"])
    #     pts = list([x,y] for x,y in zip(*box.exterior.coords.xy))
    #     pts[0][0] = max(pts[0][0], 0)
    #     pts[0][1] = max(pts[0][1], 0)
    #     pts[4][0] = max(pts[4][0], 0)
    #     pts[4][1] = max(pts[4][1], 0)
    #     pts[3][0] = min(pts[3][0], w)
    #     pts[3][1] = min(pts[3][1], h)
    #     boxes.append(pts)
    #     labels.append(oc["label_class_id"])

    # test = {
    #     "boxes": boxes,
    #     "labels": labels
    # }
    # with open("test_gt2.json", "w") as fp:
    #     json.dump(test, fp)
    
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

    out = model(x.to(device))
    # [8, 8, 8, 8, 8] corresponds to [32, 64, 128, 256, 512] in orig image
    anchors = anchor_generator.generate_like_fpn(out["backbone_out"], [8, 8, 8, 8, 8])

    gt  = ground_truth["boxes"].unsqueeze(0).to(device)
    gt = gt.repeat((2, 1, 1, 1))

    loss = 0
    for k, v in out["rpn_out"].items():
        #sanity = rpn_loss(anchors[k], ground_truth["boxes"], ground_truth["labels"], ground_truth["boxes"]) 
        loss = rpn_loss(anchors[k], v["anchor_offsets"], v["objectness_scores"], gt)

    a = 5
