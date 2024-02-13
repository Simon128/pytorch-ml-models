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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time

from ..utils import Annotation, OrientedRCNNOutput, LossOutput, encode, Encodings
from ..model import OrientedRCNN

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
    image_grid = image.copy() 

    for k, stride in zip(rpn_out.keys(), fpn_strides):
        curr_image = image.copy()
        proposals = rpn_out[k].region_proposals[0] 
        objectness = rpn_out[k].objectness_scores[0] 
        mask = torch.sigmoid(objectness) > 0.7
        k = min(mask.count_nonzero().item(), 50)
        thr_objectness = objectness[mask]
        thr_proposals = proposals[mask]
        top_idx = torch.topk(thr_objectness, k=int(k)).indices
        top_proposals = thr_proposals[top_idx]
        top_boxes = top_proposals * stride

        thickness = 1
        isClosed = True
        pred_color = (0, 0, 255)

        for o in top_boxes:
            pts_pred = o.cpu().detach().numpy().astype(np.int32)
            curr_image = cv2.polylines(curr_image, [pts_pred], isClosed, pred_color, thickness) # type: ignore

        image_grid = np.concatenate((image_grid, curr_image), axis=1)

    #cv2.imwrite(f"{index} RPN.png", image_grid)
    writer.add_image('rpn', torch.tensor(image_grid).permute((2, 0, 1))/255, index)

def visualize_head_predictions(image: np.ndarray, prediction: OrientedRCNNOutput, index: int, writer: SummaryWriter):
    _cls = prediction.head_output.classification[0]
    image_clone = image.copy()
    scores = torch.softmax(_cls, dim=-1)
    regr = prediction.head_output.boxes[0]
    for c in range(_cls.shape[-1] - 1):
        mask = scores[..., c] > 0.1
        thr_cls = _cls[mask]
        thr_regression = regr[mask]
        if len(thr_regression) == 0:
            continue
        k = min(len(thr_cls), 35)
        top_idx = torch.topk(thr_cls[..., c], k=int(k)).indices
        top_boxes = torch.gather(thr_regression, 0, top_idx.unsqueeze(-1).repeat((1, 5)))
        top_boxes = top_boxes.detach().clone().cpu().numpy()

        for box in top_boxes:
            rot = ((box[0].item(), box[1].item()), (box[2].item(), box[3].item()), box[4].item())
            pts = cv2.boxPoints(rot) # type: ignore
            pts = np.intp(pts) 
            image_clone = cv2.drawContours(image_clone, [pts], 0, (0, 255, 0), 1) # type: ignore

    #cv2.imwrite(f"{index} HEAD.png", image_clone)
    writer.add_image(f'class/head', torch.tensor(image_clone).permute((2, 0, 1))/255, index)

def visualize_targets(image: np.ndarray, boxes: torch.Tensor, index: int, writer: SummaryWriter):
    image_clone = image.copy()

    top_boxes = encode(boxes, Encodings.VERTICES, Encodings.THETA_FORMAT_TL_RT)
    for box in top_boxes:
        rot = ((box[0].item(), box[1].item()), (box[2].item(), box[3].item()), box[4].item())
        pts = cv2.boxPoints(rot) # type: ignore
        pts = np.intp(pts) 
        image_clone = cv2.drawContours(image_clone, [pts], 0, (0, 255, 0), 1) # type: ignore

    #cv2.imwrite(f"{index} HEAD.png", image_clone)
    writer.add_image(f'class/ann', torch.tensor(image_clone).permute((2, 0, 1))/255, index)

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
            else:
                ave_grads.append(0.0)
                max_grads.append(0.0)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

if __name__ == "__main__":
    device = torch.device("cuda")
    image, tensor = load_test_image(device)
    annotation = load_test_annotation(device)
    writer = SummaryWriter()

    #curr_image = image.copy()
    #for o in annotation.boxes[0]:
    #    pts_pred = o.cpu().detach().numpy().astype(np.int32)
    #    curr_image = cv2.polylines(curr_image, [pts_pred], True, (255, 0,0), 1) # type: ignore
    #hbb = encode(annotation.boxes[0], Encodings.VERTICES, Encodings.HBB_VERTICES)
    #for o in hbb:
    #    pts_pred = o.cpu().detach().numpy().astype(np.int32)
    #    curr_image = cv2.polylines(curr_image, [pts_pred], True, (0, 255, 0), 1) # type: ignore
    #cv2.imwrite("obb_vs_hbb.png", curr_image)

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
        tensor.shape[3],
        tensor.shape[2],
        {
            "head": {
                "num_classes": 3,
                "out_channels": 1024,
                "inject_annotation": True
            }
        }
    ).to(device)
    model.train()
    optimizer = torch.optim.SGD([
        {"params": model.parameters()}
        ], lr=0.005, momentum=0.9, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    running_loss_rpn: LossOutput | None = None
    running_loss_head: LossOutput | None = None
    running_loss_total: float = 0.0
    epochs = 10000

    #with profile(
    #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #    schedule=torch.profiler.schedule(
    #        wait=1,
    #        warmup=1,
    #        active=2
    #    ),
    #    record_shapes=True,
    #    profile_memory=True,
    #    with_stack=True,
    #    on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/test')
    #) as p:
    last_total_loss = 0
    for e in range(epochs):
        optimizer.zero_grad()
        out: OrientedRCNNOutput = model.forward(tensor.to(device), annotation)
        # val test:
        #model.eval()
        #with torch.inference_mode():
        #    val_uot: OrientedRCNNOutput = model.forward(tensor.to(device), annotation)
        #model.train(

        total_loss = torch.tensor(0.0)
        for k, v in out.rpn_output.items():
            total_loss = total_loss + v.loss.total_loss

        total_loss = total_loss + out.head_output.loss.total_loss
        if total_loss > last_total_loss + 1:
            test =5
        last_total_loss = total_loss.detach().clone()
        total_loss.backward()
        #plot_grad_flow(model.named_parameters())
        optimizer.step()
        #p.step()
        running_loss_total += total_loss.item()

        for k, v in out.rpn_output.items():
            v.loss.to_writer(writer, f"rpn_{k}", e)

        out.head_output.loss.to_writer(writer, "head", e)

        #scheduler.step()
        
        if e % 10 == 0:
            print(f"Epoch: {e} [{running_loss_total / 10.0}]")
            running_loss_total = 0.0

        if e % 100 == 0:
            image = tensor[0].detach().clone().cpu().permute((1, 2, 0)).numpy() * 255
            visualize_rpn_predictions(image, out, e, fpn_strides, writer)
            visualize_head_predictions(image, out, e, writer)
            visualize_targets(image, annotation.boxes[0], e, writer)

    writer.close()
