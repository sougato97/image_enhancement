import cv2
import numpy as np
import rosbag

# Define the rosbag file name and the topic name that contains the image messages
bag_file = "/home/sougato97/Thesis/datasets/euroc_mav/V1_01_easy.bag"
topic_name = "/cam0/image_raw"

# Define the output video file name and format
output_file = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Define the desired brightness value (0-255)
brightness = 200

# Create a video writer object
out = cv2.VideoWriter(output_file, fourcc, 30.0, (640, 480))

# Create a rosbag object
bag = rosbag.Bag(bag_file)

# Loop through the messages in the topic
for topic, msg, t in bag.read_messages(topics=[topic_name]):
    # Convert the message to an OpenCV image
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    
    # Convert the image from RGB to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Extract the value channel
    v = hsv[:, :, 2]
    
    # Calculate the brightness factor
    factor = brightness / np.mean(v)
    
    # Multiply the value channel by the factor and clip it to 0-255
    v = np.clip(v * factor, 0, 255)
    
    # Replace the value channel in the HSV image
    hsv[:, :, 2] = v
    
    # Convert the image back to RGB
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Write the frame to the output video
    out.write(frame)

# Release the video writer and the rosbag objects
out.release()
bag.close()
