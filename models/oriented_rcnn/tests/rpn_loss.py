import unittest
import torch
import numpy as np
from ..oriented_rpn import loss

class TestRPNLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_rpn_anchor_iou(self):
        anchors = torch.tensor([10, 10, 10, 5]).view((1, -1, 1, 1)).float().repeat((1, 4, 1, 1))
        ground_truth = torch.tensor([[5, 5], [15, 5], [15, 15], [5, 15]]).unsqueeze(0).unsqueeze(0).float()
        iou = loss.rpn_anchor_iou(anchors, ground_truth)
        print(iou.shape)
        self.assertTrue(torch.allclose(iou[0][0], torch.tensor([0.5])))

if __name__ == "__main__":
    unittest.main()
