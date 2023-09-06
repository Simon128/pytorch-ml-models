import unittest
import torch
from collections import OrderedDict

from ..oriented_rpn import OrientedRPN

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

if __name__ == "__main__":
    unittest.main()
