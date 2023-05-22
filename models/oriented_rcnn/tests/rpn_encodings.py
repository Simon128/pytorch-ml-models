import unittest
import torch
import numpy as np
from ..oriented_rpn import encodings

class TestRPNEncodings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_rpn_anchor_offset_to_midpoint_offset_rotation(self):
        anchor = torch.tensor([10, 10, 10, 10]).view((1, -1, 1, 1)).repeat((1, 4, 1, 1))
        w_factor = 0.2
        h_factor = 0.5
        prediction = torch.tensor([0, 0, 0, 0, w_factor, h_factor]).view((1, -1, 1, 1)).repeat((1, 4, 1, 1))
        target = torch.tensor([10, 10, 10, 10, 2, 5]).view((1, -1, 1, 1)).float().repeat((1, 4, 1, 1))
        out = encodings.rpn_anchor_offset_to_midpoint_offset(prediction, anchor)
        self.assertTrue(torch.allclose(out, target))

    def test_rpn_anchor_offset_to_midpoint_offset_size(self):
        anchor = torch.tensor([10, 10, 10, 10]).view((1, -1, 1, 1)).repeat((1, 4, 1, 1))
        prediction = torch.tensor([0, 0, np.log(1.5), np.log(2), 0, 0]).view((1, -1, 1, 1)).repeat((1, 4, 1, 1))
        target = torch.tensor([10, 10, 15, 20, 0, 0]).view((1, -1, 1, 1)).float().repeat((1, 4, 1, 1))
        out = encodings.rpn_anchor_offset_to_midpoint_offset(prediction, anchor)
        self.assertTrue(torch.allclose(out, target))

    def test_rpn_anchor_offset_to_midpoint_offset_position(self):
        anchor = torch.tensor([10, 10, 10, 10]).view((1, -1, 1, 1)).repeat((1, 4, 1, 1))
        prediction = torch.tensor([0.2, 0.9, 0, 0, 0, 0]).view((1, -1, 1, 1)).repeat((1, 4, 1, 1))
        target = torch.tensor([12, 19, 10, 10, 0, 0]).view((1, -1, 1, 1)).float().repeat((1, 4, 1, 1))
        out = encodings.rpn_anchor_offset_to_midpoint_offset(prediction, anchor)
        self.assertTrue(torch.allclose(out, target))

    def test_rpn_anchor_offset_to_midpoint_offset_all(self):
        anchor = torch.tensor([10, 10, 10, 10]).view((1, -1, 1, 1)).repeat((1, 4, 1, 1))
        prediction = torch.tensor([0.2, 0.9, np.log(1.5), np.log(2), 0.2, 0.5]).view((1, -1, 1, 1)).repeat((1, 4, 1, 1))
        target = torch.tensor([12, 19, 15, 20, 3, 10]).view((1, -1, 1, 1)).float().repeat((1, 4, 1, 1))
        out = encodings.rpn_anchor_offset_to_midpoint_offset(prediction, anchor)
        self.assertTrue(torch.allclose(out, target))

    def test_midpoint_offset_to_vertices(self):
        anchor = torch.tensor([10, 10, 10, 10]).view((1, -1, 1, 1)).repeat((1, 4, 1, 1))
        midpoint_offset = torch.tensor([12, 19, 15, 20, 3, 10]).view((1, -1, 1, 1)).float().repeat((1, 4, 1, 1))
        vertices = torch.tensor([[15, 9], [19.5, 29], [9, 29], [4.5, 9]]).view((1, -1, 2, 1, 1)).repeat((1, 4, 1, 1, 1))
        out = encodings.midpoint_offset_to_vertices(midpoint_offset)
        self.assertTrue(torch.allclose(out, vertices))
    
if __name__ == "__main__":
    unittest.main()
