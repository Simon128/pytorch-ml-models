import unittest
import torch

class TestBackbone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # allow for FPN and non-FPN
        cls.backbone = None
    
    def test_backbone_input(self):
        try:
            # fake image
            x = torch.ones(4, 3, 1024, 1024)
            features = cls.backbone(x)
        except Exception as inst:
            self.fail(f"backbone raises an Exception for an input shape of (4, 3, 1024, 1024) [BCHW] \n{inst}")

    def test_backbone_output(self):
        # fake image
        x = torch.ones(4, 3, 1024, 1024)
        features = cls.backbone(x)
        self.assertTrue(isinstance(features, dict) or isinstance(features, torch.Tensor), 
                        f"backbone output is neither a dict nor a torch.Tensor ({type(features)})")

if __name__ == "__main__":
    unittest.main()
