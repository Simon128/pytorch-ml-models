import torch

def relevant_samples_mask(iou: torch.Tensor, n_samples: int, simple: bool = False):
    if simple:
        positives = get_positives_mask_simple(iou)
        negatives = get_negatives_mask_simple(iou)
    else:
        positives = get_positives_mask(iou)
        negatives = get_negatives_mask(iou)
    positives_idx = torch.nonzero(positives, as_tuple=True)
    negatives_idx = torch.nonzero(negatives, as_tuple=True)
    positive_samples, negative_samples = random_sample(positives_idx[0], negatives_idx[0], n_samples)
    positives_idx = (positives_idx[0][positive_samples], positives_idx[1][positive_samples])
    negatives_idx = (negatives_idx[0][negative_samples], negatives_idx[1][negative_samples])
    mask = torch.zeros(iou.shape[0])
    mask[positives_idx[0]] = 1
    mask[negatives_idx[0]] = 1
    mask = (mask == 1)
    return mask, positives_idx, negatives_idx

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

def get_positives_mask_simple(iou_matrix: torch.Tensor):
    return iou_matrix > 0.5

def get_negatives_mask(iou_matrix: torch.Tensor):
    # iou < 0.3 -> negative
    # iou is a MxN matrix: M = num of anchors, N = num of ground truth
    return iou_matrix < 0.3

def get_negatives_mask_simple(iou_matrix: torch.Tensor):
    return iou_matrix < 0.5

