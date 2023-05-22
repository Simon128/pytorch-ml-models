import unittest
import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet50_Weights

class TestBackbone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # change this, if you want to use your own backbone
        cls.backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=ResNet50_Weights.DEFAULT, trainable_layers=5)
    
    def test_backbone_input(self):
        try:
            # fake image
            x = torch.ones(4, 3, 1024, 1024)
            features = self.backbone(x)
        except Exception as inst:
            self.fail(f"backbone raises an Exception for an input shape of (4, 3, 1024, 1024) [BCHW] \n{inst}")

    def test_backbone_output(self):
        # fake image
        x = torch.ones(4, 3, 1024, 1024)
        features = self.backbone(x)
        self.assertTrue(isinstance(features, dict),
                        f"backbone output is not a dict ({type(features)})")

if __name__ == "__main__":
    unittest.main()
