from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


model_path = "runs/pose/train/weights/best.pt"
model = YOLO(model_path)
image_path = "samples/test1.png"
results = model.predict(source=image_path)

for r in results:
    print(r.keypoints)

    # this line is changed
    keypoints = r.keypoints.xy.cpu().int().numpy()  # get the keypoints
    img_array = r.plot(kpt_line=True, kpt_radius=6)  # plot a BGR array of predictions
    im = Image.fromarray(img_array[..., ::-1])  # Convert array to a PIL Image

    draw = ImageDraw.Draw(im)
    draw.line([(keypoints[0][0][0], keypoints[0][0][1]), (keypoints[0][1][0],
            keypoints[0][1][1]), (keypoints[0][2][0], keypoints[0][2][1])],
             fill=(0, 0,255), width=5)
    im.show()