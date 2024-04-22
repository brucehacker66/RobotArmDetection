import pyrealsense2 as rs
import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from ultralytics import YOLO
import torch
from PIL import Image

robot_color = (255, 0, 0)  # Blue color for robot-arm
human_color = (0, 255, 0)  # Green color for person
line_color = (255, 255, 0)  # Yellow for connecting lines
depth_threshold = 10  # Threshold in cm
dist_threshold = 70

# Load the pre-trained YOLOv8 model
model_path = "/Users/yuenanhuang/Desktop/RobotArmDetection-main/yolov8/best.pt"
model = YOLO(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, r"/Users/yuenanhuang/Desktop/RobotArmDetection-main/yolov8/f.bag")

    # Configure the pipeline to stream the depth color stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming from file
    pipeline.start(config)

    # Create OpenCV windows to render images
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

    # Create a colorizer object
    colorizer = rs.colorizer()  # Default colorization for depth

    # Counter to control inference frequency
    frame_count = 0

    # Streaming loop
    while True:
        frames = pipeline.wait_for_frames()

        # Get depth and color frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Check if the depth frame is valid before colorizing
        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays for OpenCV
        depth_color_frame = colorizer.colorize(depth_frame)  # Use default colorization
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if frame_count % 50 == 0:  # Perform inference every 50 frames
            # Convert to PIL format for YOLO inference
            image = Image.fromarray(color_image).convert("RGB")
            results = model.predict(image)

            # Extract keypoints for robot-arm and person
            robot_keypoints = []
            human_keypoints = []

            for result in results:
                if hasattr(result, "keypoints"):
                    keypoints = result.keypoints.xy.cpu().tolist()  # Get keypoints
                    classes = result.boxes.cls.cpu().numpy()  # Get class indices

                    # Separate keypoints into robot-arm and person
                    for cls_idx, kp in zip(classes, keypoints):
                        if cls_idx == 0:
                            robot_keypoints.extend(kp)
                        elif cls_idx == 1:
                            human_keypoints.extend(kp)

            if robot_keypoints and human_keypoints:  # Ensure both lists are non-empty
                # Calculate the 10 closest pairs
                closest_pairs = []

                for rk in robot_keypoints:
                    for hk in human_keypoints:
                        dist = euclidean(rk, hk)  # Calculate Euclidean distance
                        closest_pairs.append((dist, (tuple(rk), tuple(hk))))

                # Sort pairs by distance and get the 10 closest
                closest_pairs.sort(key=lambda x: x[0])
                top_pairs = closest_pairs[:10]

                distance = 500

                # Check for threshold condition and mark the closest pair with the least distance
                for dist, (robot_point, human_point) in top_pairs:
                    # Convert to integer
                    robot_point = tuple(map(int, robot_point))
                    human_point = tuple(map(int, human_point))

                    # Get depth values
                    depth_robot = depth_frame.get_distance(robot_point[0], robot_point[1]) * 100
                    depth_human = depth_frame.get_distance(human_point[0], human_point[1]) * 100

                    depth_diff = abs(depth_robot - depth_human)

                    # Check if the points are too close based on threshold
                    # if depth_diff < distance:
                    #     closest_pair = (robot_point, human_point)
                    #     distance = depth_diff
                    if (depth_diff < depth_threshold) & (dist < dist_threshold):
                        print("Too close!")
                        print("Robot:", robot_point, depth_robot)
                        print("Human:", human_point, depth_human)
                        break

                    # Draw lines between closest pairs on both depth and color images
                    cv2.line(depth_color_image, robot_point, human_point, line_color, 2)
                    cv2.line(color_image, robot_point, human_point, line_color, 2)

                    # Visualize the closest pair
                    cv2.circle(color_image, robot_point, 2, robot_color, -1)
                    cv2.circle(color_image, human_point, 2, human_color, -1)
                    cv2.circle(depth_color_image, robot_point, 2, robot_color, -1)
                    cv2.circle(depth_color_image, human_point, 2, human_color, -1)

                    # Display depth values beside the points
                    cv2.putText(
                        color_image,
                        f"{depth_robot:.2f}cm",
                        (robot_point[0] + 30, robot_point[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        robot_color,
                        2,
                    )
                    cv2.putText(
                        color_image,
                        f"{depth_human:.2f}cm",
                        (human_point[0] - 30, human_point[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        human_color,
                        2,
                    )

                    # Display depth values beside the points
                    cv2.putText(
                        depth_color_image,
                        f"{depth_robot:.2f}cm",
                        (robot_point[0] + 30, robot_point[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        robot_color,
                        2,
                    )
                    cv2.putText(
                        depth_color_image,
                        f"{depth_human:.2f}cm",
                        (human_point[0] - 30, human_point[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        human_color,
                        2,
                    )

                # Render images in OpenCV
                cv2.imshow("Depth Stream", depth_color_image)
                cv2.imshow("Color Stream", color_image)

                # Save one frame of depth image and the corresponding RGB image
                cv2.imwrite(f'saved_depth_image{frame_count}.png', depth_color_image)
                cv2.imwrite(f'saved_color_image{frame_count}.png', color_image)

                # Exit if escape key is pressed
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    break

        frame_count += 1  # Increment frame counter

finally:
    pipeline.stop()  # Stop the pipeline
