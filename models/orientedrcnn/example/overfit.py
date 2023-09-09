import cv2
import numpy as np
import torch
from torch import autograd
import os
import json
import math
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter

from ..utils import Annotation, OrientedRCNNOutput, LossOutput, encode, Encodings
from ..model import OrientedRCNN
from ..loss import OrientedRCNNLoss

def load_test_image(device: torch.device):
    dir = os.path.dirname(os.path.abspath(__file__))
    image = cv2.imread(os.path.join(dir, "image.tif"), cv2.IMREAD_UNCHANGED) # type: ignore
    tensor = torch.tensor(image).permute((2, 0, 1)).unsqueeze(0)[:, :3, :, :] / 255
    b, c, h, w = tensor.shape
    mean = torch.mean(tensor.flatten(-2), dim=-1)
    std = torch.std(tensor.flatten(-2), dim=-1)
    tensor = (tensor - mean[..., None, None].repeat((1, 1, h, w))) / std[..., None, None].repeat((1, 1, h, w))
    return image, tensor.to(device)

def load_test_annotation(device: torch.device):
    dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dir, "annotation.json"), "r") as fp:
        ground_truth = json.load(fp)
        ground_truth["boxes"] = torch.tensor(ground_truth["boxes"])[:, :4]
        ground_truth["labels"] = torch.tensor(ground_truth["labels"], dtype=torch.float)

    boxes = ground_truth["boxes"][None].to(device).to(torch.float)
    labels = ground_truth["labels"][None].to(device).to(torch.int64)
    return Annotation(boxes=boxes, classifications=labels)

def visualize_rpn_predictions(
        image: np.ndarray, 
        prediction: OrientedRCNNOutput, 
        index: int,
        fpn_strides: list[int],
        writer: SummaryWriter
    ):
    rpn_out = prediction.rpn_output
    all_anchors = prediction.anchors
    image_grid = image.copy() 

    for k, stride in zip(rpn_out.keys(), fpn_strides):
        curr_image = image.copy()
        regression = rpn_out[k].anchor_offsets[0] 
        objectness = rpn_out[k].objectness_scores[0] 
        anchors = all_anchors[k][0] 
        mask = torch.sigmoid(objectness) > 0.7
        k = min(mask.count_nonzero().item(), 10)
        thr_objectness = objectness[mask]
        thr_regression = regression[mask]
        thr_anchors = anchors[mask]
        top_idx = torch.topk(thr_objectness, k=int(k)).indices
        top_regr = torch.gather(thr_regression, 0, top_idx.unsqueeze(-1).repeat((1, 6)))
        top_anchors = torch.gather(thr_anchors, 0, top_idx.unsqueeze(-1).repeat((1, 4)))
        top_boxes = encode(top_regr, Encodings.ANCHOR_OFFSET, Encodings.VERTICES, top_anchors) * stride
        top_anchors_vertices = encode(top_anchors, Encodings.HBB_CENTERED, Encodings.HBB_VERTICES) * stride

        thickness = 2
        isClosed = True
        pred_color = (0, 0, 255)
        a_color = (0, 255, 0)

        for o in top_boxes:
            pts_pred = o.cpu().detach().numpy().astype(np.int32)
            curr_image = cv2.polylines(curr_image, [pts_pred], isClosed, pred_color, thickness) # type: ignore

        #for a in top_anchors_vertices:
        #    anchor_pts = a.cpu().detach().numpy().astype(np.int32)
        #    curr_image = cv2.polylines(curr_image, [anchor_pts], isClosed, a_color, thickness) # type: ignore

        image_grid = np.concatenate((image_grid, curr_image), axis=1)

    writer.add_image('rpn', torch.tensor(image_grid).permute((2, 0, 1))/255, index)

def visualize_head_predictions(image: np.ndarray, prediction: OrientedRCNNOutput, index: int, writer: SummaryWriter):
    image_clone = image.copy()
    _cls = prediction.head_output.classification[0]
    _b = prediction.head_output.boxes[0]
    for c in range(_cls.shape[-1]):
        mask = torch.sigmoid(_cls[..., c]) > 0.5
        thr_cls = _cls[mask]
        thr_regression = _b[mask]
        k = min(len(thr_cls), 20)
        top_idx = torch.topk(thr_cls[..., c], k=int(k)).indices
        top_boxes = torch.gather(thr_regression, 0, top_idx.unsqueeze(-1).repeat((1, 5)))
        top_boxes = top_boxes.detach().clone().cpu().numpy()

        for box in top_boxes:
            rot = ((box[0].item(), box[1].item()), (box[2].item(), box[3].item()), box[4].item() * 180 / math.pi)
            pts = cv2.boxPoints(rot) # type: ignore
            pts = np.intp(pts) 
            image_clone = cv2.drawContours(image_clone, [pts], 0, (0, 255, 0), 2) # type: ignore

        writer.add_image(f'class {c}/head', torch.tensor(image_clone).permute((2, 0, 1))/255, index)

if __name__ == "__main__":
    device = torch.device("cuda")
    image, tensor = load_test_image(device)
    annotation = load_test_annotation(device)
    writer = SummaryWriter()

    h, w, _ = image.shape
    fpn_strides = [4, 8, 16, 32, 64]

    backbone = resnet_fpn_backbone(
        backbone_name='resnet18', 
        weights=None, 
        trainable_layers=5,
        norm_layer=None
    ).to(device)
    backbone.train()
    model = OrientedRCNN(
        backbone, 
        {
            "head": {
                "num_classes": 3,
                "out_channels": 1024,
                "inject_annotation": True,
                "n_injected_samples": 500
            }
        }
    ).to(device)
    model.train()
    optimizer = torch.optim.SGD([
        {"params": model.parameters()}
        ], lr=1e-3, momentum=0., weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    criterion = OrientedRCNNLoss(fpn_strides=fpn_strides, n_samples=256)

    running_loss_rpn: LossOutput | None = None
    running_loss_head: LossOutput | None = None
    running_loss_total: float = 0.0
    epochs = 10000

    for e in range(epochs):
        optimizer.zero_grad()
        #with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
        #    with record_function("model_inference"):
        #        out: OrientedRCNNOutput = model(tensor.to(device))
        #    with record_function("loss_calc"):
        #        rpn_loss, head_loss = criterion(out, annotation)
        #        loss = rpn_loss.total_loss + head_loss.total_loss
        #    with record_function("backward"):
        #        loss.backward()
        #    with record_function("optim"):
        #        optimizer.step()
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

        #with autograd.detect_anomaly():
        out: OrientedRCNNOutput = model(tensor.to(device), annotation, e)
        rpn_loss, head_loss = criterion(out, annotation)
        loss = rpn_loss.total_loss + head_loss.total_loss
        loss.backward()
        optimizer.step()
        running_loss_total += loss.item()
        rpn_loss.to_writer(writer, "rpn", e)
        head_loss.to_writer(writer, "head", e)

        scheduler.step()
        
        if e % 10 == 0:
            print(f"Epoch: {e} [{running_loss_total / 10.0}]")
            running_loss_total = 0.0

        if e % 100 == 0:
            visualize_rpn_predictions(image, out, e, fpn_strides, writer)
            visualize_head_predictions(image, out, e, writer)

    writer.close()
