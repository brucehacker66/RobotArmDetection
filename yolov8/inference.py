from ultralytics import YOLO
from PIL import Image, ImageDraw
import time

start_time = time.perf_counter()
model_path = "runs/pose/train/weights/best.pt"
model = YOLO(model_path)
folder_path = "../samples"
result_folder_path = "../result"
results = model.predict(source=folder_path)
end_time = time.perf_counter()
elapsed_time = end_time - start_time # calculate the elapsed time
print(f"Time taken Prediction: {elapsed_time:.2f} seconds") # print the elapsed time
num = 0

for i, r in enumerate(results): # iterate over the results list
    print(f"Image {i}: {r.keypoints}")

    keypoints = r.keypoints.xy.cpu().int().numpy()  # get the keypoints
    img_array = r.plot(kpt_line=True, kpt_radius=6)  # plot a BGR array of predictions
    im = Image.fromarray(img_array[..., ::-1])  # Convert array to a PIL Image

    draw = ImageDraw.Draw(im)
    draw.line([(keypoints[0][0][0], keypoints[0][0][1]), (keypoints[0][1][0],
            keypoints[0][1][1]), (keypoints[0][2][0], keypoints[0][2][1]), (keypoints[0][3][0], keypoints[0][3][1])],
             fill=(0, 0,255), width=5)
    im.save(f"{result_folder_path}/result_{num}.png", format="PNG") # save the image with a different name
    num += 1