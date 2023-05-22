import unittest
import torch
from collections import OrderedDict
from torchvision.models.detection.anchor_utils import AnchorGenerator

from ..oriented_rpn import OrientedRPN, decode_rpn_regression_output

class TestRPN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = {

                }
        cls.RPN = OrientedRPN(cfg)
        b = 4
        c = 256
        h, w = 256, 256
        cls.backbone_out = OrderedDict()
        for i in range(5):
            cls.backbone_out[str(i)] = torch.ones((b, c, int(h / (2**i)), int(w / (2**i))))
        cls.b = 4
        cls.c = 256
        cls.h = h
        cls.w = w
        cls.num_anchors_per_spatial_location = 3
        
    def test_rpn_input(self):
        try:
            self.RPN(self.backbone_out)
        except Exception as inst:
            self.fail(f"oriented RPN raises an exception for FPN input \n{inst}")

    def test_rpn_output_shape(self):
        rpn_out = self.RPN(self.backbone_out)
        self.assertTrue(isinstance(rpn_out, OrderedDict))

        for i, level_out in enumerate(rpn_out.values()):
            expected_shape_regression = (self.b, 6 * self.num_anchors_per_spatial_location, int(self.h / (2**i)), int(self.w / (2**i)))
            expected_shape_objectness = (self.b, 1 * self.num_anchors_per_spatial_location, int(self.h / (2**i)), int(self.w / (2**i)))
            self.assertTrue("anchor_offsets" in level_out)
            self.assertTrue("objectness_scores" in level_out)
            self.assertTupleEqual(level_out["anchor_offsets"].shape, expected_shape_regression)
            self.assertTupleEqual(level_out["objectness_scores"].shape, expected_shape_objectness)

    def test_rpn_loss(self):
        pass

    def test_rpn_loss_regression(self):
        pass

    def test_rpn_loss_objectness(self):
        pass

    def test_rpn_loss_tp(self):
        pass

    def test_rpn_loss_fp(self):
        pass

    def test_decode_rpn_regression_output_rotation(self):
        anchor = torch.Tensor([10, 10, 10, 10]).float()
        delta_alpha = 0.2 * 10 # rotate by 20% of width
        delta_beta = 0.5 * 10 # rotate by 50% of height
        target = torch.Tensor([10, 10, 10, 10, 2, 5]).float()

        anchor = anchor.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        target = target.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        

    def test_midpoint_offset_to_coordinates(self):
        pass

    def test_ground_truth_anchor_offset(self):
        pass

if __name__ == "__main__":
    unittest.main()
