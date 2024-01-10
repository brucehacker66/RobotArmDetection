import torch
import cv2

print(cv2.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)