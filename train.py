from ultralytics import YOLO
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)

print("Device:", device, "\n")
model = YOLO('runs/pose/train/weights/best.pt')  # load a pretrained model (recommended for training)
if __name__ == '__main__':
    model.train(data='config.yaml', epochs=100, imgsz=640)