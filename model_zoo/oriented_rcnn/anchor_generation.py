import torch
from collections import OrderedDict
from typing import List



class AnchorGenerator:
    def __init__(self, cfg: dict, device = torch.device("cpu")) -> None:
        self.num_anchors = cfg.get("num_anchors", 3)
        self.aspect_ratios = cfg.get("aspect_ratios", [(1,2), (1,1), (2,1)])
        self.device = device
        
    def generate_like_fpn(self, fpn_output: OrderedDict, sqrt_pixel_areas: List[int]):
        assert len(fpn_output.keys()) == len(sqrt_pixel_areas)
        result = OrderedDict()

        for (k, v), area in zip(fpn_output.items(), sqrt_pixel_areas):
            b, _, h, w = v.shape
            anchors = self._generate(w, h, area)
            result[k] = torch.stack([anchors for _ in range(b)])

        return result

    def generate_like(self, tensor: torch.Tensor, sqrt_pixel_area: int):
        b, _, h, w = tensor.shape
        anchors = self._generate(w, h, sqrt_pixel_area)
        return torch.stack([anchors for _ in range(b)])

    def _generate(self, w: int, h: int, sqrt_pixel_area: int):
        w_arange = torch.arange(0, w, 1).to(self.device)
        w_zeros = torch.zeros_like(w_arange).to(self.device)
        h_arange = torch.arange(0, h, 1).to(self.device)
        h_zeros = torch.zeros_like(h_arange).to(self.device)
        result = torch.zeros((len(self.aspect_ratios), 4, h, w)).to(self.device)
        
        for idx, ar in enumerate(self.aspect_ratios):
            aw = torch.full_like(w_arange, ar[0]/max(ar) * sqrt_pixel_area * ar[0]).to(self.device)
            ah = torch.full_like(h_arange, ar[1]/max(ar) * sqrt_pixel_area * ar[1]).to(self.device)
            ax = torch.maximum(w_zeros, w_arange - aw/2)
            ay = torch.maximum(h_zeros, h_arange - ah/2)

            w_max = torch.full_like(w_arange, torch.max(w_arange).item())
            h_max = torch.full_like(h_arange, torch.max(h_arange).item())
            aw = aw + torch.minimum(w_zeros, w_arange - aw/2) + torch.minimum(w_zeros, w_max - (w_arange + aw/2))
            ah = ah + torch.minimum(h_zeros, h_arange - ah/2) + torch.minimum(h_zeros, h_max - (h_arange + ah/2))

            aw = torch.stack([aw for _ in range(h)])
            ah = torch.stack([ah for _ in range(w)], dim=1)
            ax = torch.stack([ax for _ in range(h)])
            ay = torch.stack([ay for _ in range(w)], dim=1)

            result[idx, :, :, :] = torch.stack((ax, ay, aw, ah), dim=0)

        # import cv2 
        # import os
        # test = result[1, :, 180, 140]
        # image = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)),"test", "test_img.tif"), cv2.IMREAD_UNCHANGED)
        # # Blue color in BGR
        # color = (255, 0, 0)
        
        # # Line thickness of 2 px
        # thickness = 2
        # # Using cv2.rectangle() method
        # # Draw a rectangle with blue line borders of thickness of 2 px
        # start_point = (int(max(0, test[0].item() - test[2].item()/2)), int(max(0, test[1].item() - test[3].item()/2)))
        # end_point = (int(test[0].item() + test[2].item()/2), int(test[1].item() + test[3].item()/2))
        # image = cv2.rectangle(image, start_point, end_point, color, thickness)
        
        # # Displaying the image 
        # cv2.imshow("test", image) 
        # cv2.waitKey()

        return result


