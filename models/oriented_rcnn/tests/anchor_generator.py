import unittest
import torch
from collections import OrderedDict

from ..oriented_rpn import FPNAnchorGenerator

class TestAnchorGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_anchor_generator_single(self):
        image_height = 1024
        image_width = 1024
        sizes = (32, 64)
        aspect_ratios = ((1, 2), (1, 1), (2, 1))
        shape = (4, 6 * len(aspect_ratios), 256, 256)
        features = torch.ones(shape)

        generator = FPNAnchorGenerator(sizes, aspect_ratios)
        output = generator.generate_single(features, 32, image_width, image_height)
        self.assertTupleEqual(output.shape, (4, 4 * len(aspect_ratios), 256, 256))
        example1 = output[0, :4, 100, 100]
        example2 = output[0, 4:8, 100, 100]
        example3 = output[0, 8:, 100, 100]
        # 16x64 -> 4x16 in feature size
        self.assertTrue(all(example1 == torch.tensor([100, 100, 4, 16])))
        # 32x32 -> 8x8 in feature size
        self.assertTrue(all(example2 == torch.tensor([100, 100, 8, 8])))
        # 64x16 -> 16x4 in feature size
        self.assertTrue(all(example3 == torch.tensor([100, 100, 16, 4])))

if __name__ == "__main__":
    unittest.main()
