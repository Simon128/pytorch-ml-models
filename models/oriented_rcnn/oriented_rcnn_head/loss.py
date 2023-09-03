import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from ..encodings import rectangular_vertices_to_5_param

def random_sample(positives_idx: torch.Tensor, negatives_idx: torch.Tensor, num: int):
    num_samples_positive = min(num // 2, len(positives_idx))
    num_samples_negative = min(num // 2 + (num // 2 - num_samples_positive), len(negatives_idx))
    pos_p = torch.ones_like(positives_idx) / len(positives_idx)
    neg_p = torch.ones_like(negatives_idx) / len(negatives_idx)

    if num_samples_positive > 0:
        positive_samples = pos_p.multinomial(num_samples=num_samples_positive)
    else:
        positive_samples = torch.Tensor([]).to(torch.int64)
    if num_samples_negative > 0:
        negative_samples = neg_p.multinomial(num_samples=num_samples_negative)
    else:
        negative_samples = torch.Tensor([]).to(torch.int64)

    return positive_samples, negative_samples

def get_positives_mask(iou_matrix: torch.Tensor):
    # iou >= 0.7 -> positive
    # highest iou for ground truth + 0.3 < iou < 0.7
    # iou is a MxN matrix: M = num of anchors, N = num of ground truth
    m, n = iou_matrix.shape
    above_07 = (iou_matrix > 0.7)

    max_per_gt = torch.max(iou_matrix, dim=0)
    max_between = iou_matrix[(max_per_gt.indices, torch.arange(n))] > 0.3
    anchor_idx = max_per_gt.indices[max_between]
    gt_idx = torch.arange(n).to(iou_matrix.device)[max_between]
    between = torch.zeros_like(iou_matrix, dtype=torch.int)
    between[(anchor_idx, gt_idx)] = 1

    positives = above_07 | between
    return positives == 1

def get_negatives_mask(iou_matrix: torch.Tensor):
    # iou < 0.3 -> negative
    # iou is a MxN matrix: M = num of anchors, N = num of ground truth
    return iou_matrix < 0.3

def get_box_iou_format(params_5: torch.Tensor):
    x_center = params_5[..., 0]
    y_center = params_5[..., 1]
    w = params_5[..., 2]
    h = params_5[..., 3]
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)

def oriented_rcnn_head_loss(
        classification: torch.Tensor,
        regression: torch.Tensor,
        rpn_regression: torch.Tensor,
        classification_targets: list[torch.Tensor],
        regression_targets: list[torch.Tensor]
    ):
    batch_size = len(classification_targets)
    cls_losses = []
    regr_losses = []

    for b in range(batch_size):
        box_targets = rectangular_vertices_to_5_param(regression_targets[b])
        # prob wrong
        adjusted_regression = rpn_regression[b] + regression[b]
        #
        iou = box_iou(get_box_iou_format(adjusted_regression), get_box_iou_format(box_targets))
        positives = get_positives_mask(iou)
        negatives = get_negatives_mask(iou)
        positives_idx = torch.nonzero(positives, as_tuple=True)
        negatives_idx = torch.nonzero(negatives, as_tuple=True)
        positive_samples, negative_samples = random_sample(positives_idx[0], negatives_idx[0], 256)
        positives_idx = (positives_idx[0][positive_samples], positives_idx[1][positive_samples])
        negatives_idx = (negatives_idx[0][negative_samples], negatives_idx[1][negative_samples])

        background_cls = classification.shape[-1] - 1
        cls_target = torch.full([len(classification[b])], background_cls).to(classification.device)
        if len(positives_idx[0]) > 0:
            cls_target[positives_idx[0]] = classification_targets[b][positives_idx[1]]
        mask = torch.zeros_like(cls_target)
        mask[positives_idx[0]] = 1
        mask[negatives_idx[0]] = 1
        mask = (mask == 1)

        cls_loss = F.cross_entropy(classification[b][mask], cls_target[mask].to(torch.long), reduction='mean')
        cls_losses.append(cls_loss)

        if torch.count_nonzero(positives) <= 0:
            continue

        relevant_gt = box_targets[positives_idx[1]]
        relevant_pred = adjusted_regression[positives_idx[0]]
        regr_loss = F.smooth_l1_loss(relevant_pred, relevant_gt, reduction='mean')
        regr_losses.append(regr_loss)

    classification_loss = sum(cls_losses) / batch_size
    regression_loss = sum(regr_losses) / batch_size

    return {
        "loss": classification_loss + regression_loss,
        "classification_loss": classification_loss,
        "regression_loss": regression_loss
    }
