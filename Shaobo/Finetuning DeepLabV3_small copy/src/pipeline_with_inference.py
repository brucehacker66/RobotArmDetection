import pyrealsense2 as rs

import torch
import argparse
import cv2
import os
from utils import get_segment_labels, draw_segmentation_map, image_overlay
from PIL import Image
from config import ALL_CLASSES
from model import prepare_model

import json
import numpy as np
from scipy.spatial import KDTree
import tqdm
from nearest_pixels import find_all_pairs, pairs_dict_to_list

robot_color = (255, 0, 0)  # Blue color for the robot point
human_color = (0, 255, 0)  # Green color for the human point

# New code for inference and nearest pixels: 
# Set computation device and load pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = prepare_model(len(ALL_CLASSES))
ckpt = torch.load('../outputs/best_model.pth')
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, r"D:\Industrial-Robot-Safety\code\robot-safety-detection\f.bag")

    # Configure the pipeline to stream the depth color stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)


    # Start streaming from file
    pipeline.start(config)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

    # Create colorizer object
    colorizer = rs.colorizer()

    # Streaming loop
    for i in range(1):
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)
        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Get color frame
        color_frame = frames.get_color_frame()

        # Convert color_frame to numpy array to render image in opencv
        color_image = np.asanyarray(color_frame.get_data())







        # New code for infernce and nearest pixels
        # Construct the argument parser.

        image = Image.fromarray(np.uint8(color_image)).convert('RGB')

        # # Resize very large images (if width > 1024.) to avoid OOM on GPUs.
        # if image.size[0] > 1024:
        #     image = image.resize((800, 800))

        # Do forward pass and get the output dictionary.
        outputs = get_segment_labels(image, model, device)
        # Get the data from the `out` key.
        outputs = outputs['out']
        segmented_image = draw_segmentation_map(outputs)

        # Find all pairs for all images in the output directory, write to a json file
        out_dict = find_all_pairs(image)
        # output[masks] = out_dict
        out_list = pairs_dict_to_list(out_dict['all_pairs'])








        # pass RGB image to model
        # points = [((100, 300), (103, 300))]
        points = out_list
        for point in points:
            robot_point = point[0]
            human_point = point[1]
            # get the depth value of the robot and human in cm
            depth_robot = depth_frame.get_distance(robot_point[0], robot_point[1])*100
            depth_human = depth_frame.get_distance(human_point[0], human_point[1])*100
            # visualize point on the image:
            # Visualize points on the images
            cv2.circle(color_image, robot_point, 2, robot_color, -1)
            cv2.circle(color_image, human_point, 2, human_color, -1)
            cv2.circle(depth_color_image, robot_point, 2, robot_color, -1)
            cv2.circle(depth_color_image, human_point, 2, human_color, -1)

            # Display the depth values beside the points
            cv2.putText(color_image, f"{depth_robot:.2f}cm", (robot_point[0] + 30, robot_point[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, robot_color, 2)
            cv2.putText(color_image, f"{depth_human:.2f}cm", (human_point[0] - 30, human_point[1]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, human_color, 2)
            cv2.putText(depth_color_image, f"{depth_robot:.2f}cm", (robot_point[0] + 30, robot_point[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, robot_color, 2)
            cv2.putText(depth_color_image, f"{depth_human:.2f}cm", (human_point[0] - 30, human_point[1]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, human_color, 2)

            # Render images in opencv windows
            cv2.imshow("Depth Stream", depth_color_image)
            cv2.imshow("Color Stream", color_image)

            # Save one frame of depth image and the corresponding RGB image
            cv2.imwrite('saved_depth_image.png', depth_color_image)
            cv2.imwrite('saved_color_image.png', color_image)

        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pass