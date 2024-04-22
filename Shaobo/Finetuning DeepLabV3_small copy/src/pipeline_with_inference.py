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
deptth_threshold = 10 #threshold in cm

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
    rs.config.enable_device_from_file(config, r"../config/f.bag")

    # Configure the pipeline to stream the depth color stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)


    # Start streaming from file
    pipeline.start(config)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

    # Create colorizer object
    colorizer = rs.colorizer()
    count = 0
    # Streaming loop
    while True:
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

        if count % 50 == 0: # infer every 50 frames

            # Segmentation INference
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
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

            # print(segmented_image.size)
            # cv2.imshow('Segmented image', segmented_image)
            # cv2.waitKey(5)

            # Find all pairs for all images in the output directory, write to a json file
            out_dict = find_all_pairs(segmented_image)
            # output[masks] = out_dict
            out_list = pairs_dict_to_list(out_dict['all_pairs'])

            # iterate to find the closest points
            points = out_list
            closest_pair = points[0] #closest pair of pixels
            distance = 500
            if points is None:
                continue
            print(points)
            for point in points:
                robot_point = point[0]
                human_point = point[1]
                # get the depth value of the robot and human in cm
                depth_robot = depth_frame.get_distance(robot_point[0], robot_point[1])*100
                depth_human = depth_frame.get_distance(human_point[0], human_point[1])*100
                depth_diff = abs(depth_robot - depth_human)
                if depth_diff < distance:
                    closest_pair = point
                    distance = depth_diff
                if depth_diff < depth_threshold:
                    print("too close")
                    print("robot:", robot_point, depth_robot)
                    print("human:", human_point, depth_human)
                    break
            
            # visualize the closest pair
            closest_robot_point, closest_human_point = closest_pair
            closest_depth_robot = depth_frame.get_distance(closest_robot_point[0], closest_robot_point[1])*100
            closest_depth_human = depth_frame.get_distance(closest_human_point[0], closest_human_point[1])*100
            # visualize point on the image:
            cv2.circle(color_image, closest_robot_point, 2, robot_color, -1)
            cv2.circle(color_image, closest_human_point, 2, human_color, -1)
            cv2.circle(depth_color_image, closest_robot_point, 2, robot_color, -1)
            cv2.circle(depth_color_image, closest_human_point, 2, human_color, -1)

            # Display the depth values beside the points
            cv2.putText(color_image, f"{closest_depth_robot:.2f}cm", (closest_robot_point[0] + 30, closest_robot_point[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, robot_color, 2)
            cv2.putText(color_image, f"{closest_depth_human:.2f}cm", (closest_human_point[0] - 30, closest_human_point[1]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, human_color, 2)
            cv2.putText(depth_color_image, f"{closest_depth_robot:.2f}cm", (closest_robot_point[0] + 30, closest_robot_point[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, robot_color, 2)
            cv2.putText(depth_color_image, f"{closest_depth_human:.2f}cm", (closest_human_point[0] - 30, closest_human_point[1]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, human_color, 2)

            # Render images in opencv windows
            cv2.imshow(f"Depth Stream{count}", depth_color_image)
            cv2.imshow(f"Color Stream{count}", color_image)

            # Save one frame of depth image and the corresponding RGB image
            cv2.imwrite(f'saved_depth_image{count}.png', depth_color_image)
            cv2.imwrite(f'saved_color_image{count}.png', color_image)

            key = cv2.waitKey(1)
            # if pressed escape exit program
            if key == 27:
                cv2.destroyAllWindows()
                break
        count += 1

finally:
    pass