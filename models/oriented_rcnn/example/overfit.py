import cv2
import numpy as np
import torch
import os
import json
import math
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from ..data_formats import Annotation, OrientedRCNNOutput, LossOutput
from ..encoder import encode, Encodings

from ..model import OrientedRCNN
from ..loss import OrientedRCNNLoss

def load_test_image(device: torch.device):
    dir = os.path.dirname(os.path.abspath(__file__))
    image = cv2.imread(os.path.join(dir, "image.tif"), cv2.IMREAD_UNCHANGED) # type: ignore
    tensor = torch.tensor(image).permute((2, 0, 1)).unsqueeze(0)[:, :3, :, :] / 255
    return image, tensor.to(device)

def load_test_annotation(device: torch.device):
    dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dir, "annotation.json"), "r") as fp:
        ground_truth = json.load(fp)
        ground_truth["boxes"] = torch.tensor(ground_truth["boxes"])[:, :4]
        ground_truth["labels"] = torch.tensor(ground_truth["labels"], dtype=torch.float)

    boxes = ground_truth["boxes"].unsqueeze(0).to(device).to(torch.float)
    labels = ground_truth["labels"].unsqueeze(0).to(device).to(torch.int64)
    return Annotation(boxes=boxes, classifications=labels)

def visualize_rpn_predictions(
        image: np.ndarray, 
        prediction: OrientedRCNNOutput, 
        index: int,
        fpn_strides: list[int]
    ):
    rpn_out = prediction.rpn_output
    all_anchors = prediction.anchors
    image_grid = image.copy() 

    for k, stride in zip(rpn_out.keys(), fpn_strides):
        curr_image = image.copy()
        regression = rpn_out[k].anchor_offsets[0] 
        objectness = rpn_out[k].objectness_scores[0] 
        anchors = all_anchors[k][0] 
        mask = torch.sigmoid(objectness) > 0.8
        k = min(mask.count_nonzero().item(), 100)
        thr_objectness = objectness[mask]
        thr_regression = regression[mask]
        thr_anchors = anchors[mask]
        top_idx = torch.topk(thr_objectness, k=int(k)).indices
        top_regr = torch.gather(thr_regression, 0, top_idx.unsqueeze(-1).repeat((1, 6)))
        top_anchors = torch.gather(thr_anchors, 0, top_idx.unsqueeze(-1).repeat((1, 4)))
        top_boxes = encode(top_regr, Encodings.ANCHOR_OFFSET, Encodings.VERTICES, top_anchors) * stride
        top_anchors_vertices = encode(top_anchors, Encodings.HBB_CENTERED, Encodings.HBB_VERTICES) * stride

        thickness = 1
        isClosed = True
        pred_color = (0, 0, 255)
        a_color = (0, 255, 0)

        for o in top_boxes:
            pts_pred = o.cpu().detach().numpy().astype(np.int32)
            curr_image = cv2.polylines(curr_image, [pts_pred], isClosed, pred_color, thickness) # type: ignore

        for a in top_anchors_vertices:
            anchor_pts = a.cpu().detach().numpy().astype(np.int32)
            curr_image = cv2.polylines(curr_image, [anchor_pts], isClosed, a_color, thickness) # type: ignore

        image_grid = np.concatenate((image_grid, curr_image), axis=1)

    cv2.imwrite(f"rpn_{index}.png", image_grid) # type: ignore

def visualize_head_predictions(image: np.ndarray, prediction: OrientedRCNNOutput, index: int):
    image_clone = image.copy()
    _cls = prediction.head_output.classification.squeeze(0)
    mask = torch.sigmoid(_cls[..., -1]) < 0.2
    k = min(mask.count_nonzero().item(), 100)
    thr_cls = _cls[mask]
    thr_regression = prediction.head_output.boxes.squeeze(0)[mask]
    top_idx = torch.topk(thr_cls[..., -1], k=int(k), largest=False).indices
    top_boxes = torch.gather(thr_regression, 0, top_idx.unsqueeze(-1).repeat((1, 5)))
    top_boxes = top_boxes.detach().clone().cpu().numpy()

    for box in top_boxes:
        rot = ((box[0].item(), box[1].item()), (box[2].item(), box[3].item()), box[4].item() * 180 / math.pi)
        pts = cv2.boxPoints(rot) # type: ignore
        pts = np.int0(pts) # type: ignore
        image_clone = cv2.drawContours(image_clone, [pts], 0, (0, 255, 0), 2) # type: ignore

    cv2.imwrite(f"head_{index}.png", image_clone) # type: ignore

if __name__ == "__main__":
    device = torch.device("cuda")
    image, tensor = load_test_image(device)
    annotation = load_test_annotation(device)

    # debug
    image_clone = image.copy()
    boxes = encode(annotation.boxes[0], Encodings.VERTICES, Encodings.ORIENTED_CV2_FORMAT)
    for box in boxes:
        rot = ((box[0].item(), box[1].item()), (box[2].item(), box[3].item()), box[4].item() * 180 / math.pi)
        pts = cv2.boxPoints(rot) # type: ignore
        pts = np.int0(pts) # type: ignore
        image_clone = cv2.drawContours(image_clone, [pts], 0, (0, 255, 0), 1) # type: ignore
    cv2.imwrite(f"test.png", image_clone) # type: ignore
    image_clone = image.copy()
    for box in annotation.boxes[0]:
        pts = box.detach().clone().cpu().numpy()
        pts = np.int0(pts) # type: ignore
        image_clone = cv2.drawContours(image_clone, [pts], 0, (0, 0, 255), 1) # type: ignore
    cv2.imwrite(f"test_gt.png", image_clone) # type: ignore
    #

    h, w, _ = image.shape
    fpn_strides = [4, 8, 16, 32, 64]

    backbone = resnet_fpn_backbone('resnet18', pretrained=False, norm_layer=None, trainable_layers=5).to(device)
    backbone.train()
    model = OrientedRCNN(backbone, {}).to(device)
    model.train()
    optimizer = torch.optim.Adam([
        {"params": model.parameters()}
        ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    criterion = OrientedRCNNLoss(fpn_strides=fpn_strides)

    running_loss_rpn: LossOutput | None = None
    running_loss_head: LossOutput | None = None
    epochs = 10000

    for e in range(epochs):
        optimizer.zero_grad()
        out: OrientedRCNNOutput = model(tensor.to(device))

        rpn_loss, head_loss = criterion(out, annotation)
        loss = rpn_loss.total_loss + head_loss.total_loss
        loss.backward()
        optimizer.step()

        if running_loss_rpn is None:
            running_loss_rpn = rpn_loss
        else:
            running_loss_rpn = running_loss_rpn + rpn_loss.detach().clone()
        if running_loss_head is None:
            running_loss_head = head_loss
        else:
            running_loss_head = running_loss_rpn + head_loss.detach().clone()

        scheduler.step()
        
        if e % 10 == 0:
            if running_loss_rpn is None or running_loss_head is None:
                break
            running_loss_rpn.total_loss /= 10
            running_loss_rpn.classification_loss /= 10
            running_loss_rpn.regression_loss /= 10
            running_loss_head.total_loss /= 10
            running_loss_head.classification_loss /= 10
            running_loss_head.regression_loss /= 10
            print(f"Epoch: {e}")
            print("RPN losses")
            print(running_loss_rpn)
            print("HEAD losses")
            print(running_loss_head)
            running_loss_rpn = None
            running_loss_head = None

        if e % 100 == 0:
            visualize_rpn_predictions(image, out, e, fpn_strides)
            visualize_head_predictions(image, out, e)
