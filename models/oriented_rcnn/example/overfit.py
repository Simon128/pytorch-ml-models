import cv2
import torch

if __name__ == "__main__":
    image = cv2.imread("example.png")
    image = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(0)
