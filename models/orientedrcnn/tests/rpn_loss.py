import unittest
import torch
import torch.nn.functional as F
import numpy as np
from ..oriented_rpn import loss

class TestRPNLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_rpn_anchor_iou(self):
        anchors = torch.tensor([10, 10, 10, 5]).view((1, -1, 1, 1)).float().repeat((1, 4, 1, 1))
        ground_truth = torch.tensor([[5, 5], [15, 5], [15, 15], [5, 15]]).unsqueeze(0).float()
        flat_anchors = loss.flatten_anchors(anchors)
        iou = loss.rpn_anchor_iou(flat_anchors[0], ground_truth)
        self.assertTrue(torch.allclose(iou[0][0], torch.tensor([0.5])))

    #def test_rpn_loss(self):
    #    anchor = torch.tensor([10, 10, 10, 10]).view((1, -1, 1, 1)).repeat((1, 4, 1, 1))
    #    ground_truth = list([torch.tensor([[5, 5], [15, 5], [15, 15], [5, 15]]).unsqueeze(0).float()])
    #    prediction = torch.tensor([12, 19, 10, 10, 0, 0]).view((1, -1, 1, 1)).float().repeat((1, 4, 1, 1))
    #    objectness = torch.tensor([0.8]).view(1, -1, 1, 1).float().repeat((1, 4, 1, 1))
    #    _loss = loss.rpn_loss(prediction, objectness, anchor, ground_truth)

    def test_rpn_loss_zero(self):
        anchor = torch.tensor([10, 10, 10, 10]).view((1, -1, 1, 1)).repeat((1, 4, 1, 1))
        ground_truth = list([torch.tensor([[5, 5], [15, 5], [15, 15], [5, 15]]).unsqueeze(0).float()])
        prediction = torch.tensor([0, 0, 0, 0, -0.5, -0.5]).view((1, -1, 1, 1)).float().repeat((1, 4, 1, 1))
        objectness = torch.tensor([10e20]).view(1, -1, 1, 1).float().repeat((1, 4, 1, 1))
        _loss = loss.rpn_loss(prediction, objectness, anchor, ground_truth)
        self.assertTrue(_loss == 0)

    def test_rpn_loss_cls_5(self):
        anchor = torch.tensor([10, 10, 10, 10]).view((1, -1, 1, 1)).repeat((1, 4, 1, 1))
        ground_truth = list([torch.tensor([[5, 5], [15, 5], [15, 15], [5, 15]]).unsqueeze(0).float()])
        prediction = torch.tensor([0, 0, 0, 0, -0.5, -0.5]).view((1, -1, 1, 1)).float().repeat((1, 4, 1, 1))
        objectness = torch.tensor([0]).view(1, -1, 1, 1).float().repeat((1, 4, 1, 1))
        _loss = loss.rpn_loss(prediction, objectness, anchor, ground_truth)
        expected = F.binary_cross_entropy(torch.tensor(0.5), torch.tensor(1.0))
        self.assertEqual(_loss, expected)

    def test_positives_mask(self):
        iou = torch.tensor([
            [0.1, 0.2, 0.5, 0.2, 0.8],
            [0, 0.9, 0, 0, 0.4],
            [0.4, 0, 0.35, 0, 0],
            [0, 0, 0, 0, 0.35]
            ])
        target = torch.tensor([
            [False, False, True, False, True],
            [False, True, False, False, False],
            [True, False, False, False, False],
            [False, False, False, False, False]
            ])
        pred = loss.get_positives_mask(iou)
        self.assertEqual(torch.count_nonzero(pred==target), iou.shape[0] * iou.shape[1])

    def test_negatives_mask(self):
        iou = torch.tensor([
            [0.1, 0.2, 0.5, 0.2, 0.8],
            [0, 0.9, 0, 0, 0.4],
            [0.4, 0, 0.35, 0, 0],
            [0, 0, 0, 0, 0.35]
            ])
        target = torch.tensor([
            [False, False, True, False, True],
            [False, True, False, False, True],
            [True, False, True, False, False],
            [False, False, False, False, True]
            ]) == False
        pred = loss.get_negatives_mask(iou)
        self.assertEqual(torch.count_nonzero(pred==target), iou.shape[0] * iou.shape[1])

    def test_ignore_mask(self):
        iou = torch.tensor([
            [0.1, 0.2, 0.5, 0.2, 0.8],
            [0, 0.9, 0, 0, 0.4],
            [0.4, 0, 0.35, 0, 0],
            [0, 0, 0, 0, 0.35]
            ])
        target = torch.tensor([
            [False, False, False, False, False],
            [False, False, False, False, True],
            [False, False, True, False, False],
            [False, False, False, False, True]
            ])
        pos_mask = loss.get_positives_mask(iou)
        neg_mask = loss.get_negatives_mask(iou)
        pred = loss.get_ignore_mask(pos_mask, neg_mask)
        self.assertEqual(torch.count_nonzero(pred==target), iou.shape[0] * iou.shape[1])

if __name__ == "__main__":
    unittest.main()
