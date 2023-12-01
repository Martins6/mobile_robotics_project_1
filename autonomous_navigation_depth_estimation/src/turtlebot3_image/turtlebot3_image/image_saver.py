#!/usr/bin/env python3

# ROS2 module imports
import rclpy # ROS2 client library (rcl) for Python (built on rcl C API)
from rclpy.node import Node # Node class for Python nodes
from geometry_msgs.msg import Twist # Twist (linear and angular velocities) message class
from sensor_msgs.msg import Image # Image (camera frame) message class
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy # Ouality of Service (tune communication between nodes)
from rclpy.duration import Duration # Time duration class

# Python mudule imports
import cv2 # OpenCV
from cv_bridge import CvBridge, CvBridgeError # OpenCV bridge for ROS2
import numpy as np

import os


# Node class
class ImageSaver(Node):

    def __init__(self):
        # Information and debugging
        info = "\nClassify the image coming from the robot's camera.\n"
        print(info)
        # ROS2 infrastructure
        super().__init__('image_classifier') # Create a node with name 'image_classifier'
        qos_profile = QoSProfile( # Ouality of Service profile
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE, # Reliable (not best effort) communication
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, # Keep/store only up to last N samples
            depth=10 # Queue size/depth of 10 (only honored if the “history” policy was set to “keep last”)
        )
        self.robot_image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.robot_image_callback, qos_profile) # Subscriber which will subscribe to Image message on the topic 'camera/image_raw' adhering to 'qos_profile' QoS profile
        self.robot_image_sub # Prevent unused variable warning
        self.cv_bridge = CvBridge() # Initialize object to capture and convert the image
        timer_period = 3 # seconds
        self.timer = self.create_timer(timer_period, self.process_image) # Define timer to execute 'process_image()' every 'timer_period' seconds
        self.start_time = self.get_clock().now() # Record current time in seconds

        if not os.path.exists('data/raw_images'):
            os.makedirs('data/raw_images')


    def robot_image_callback(self, msg):
        try:
            self.cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") # Capture and convert most recent image 'msg' to OpenCV image with 'bgr8' encoding
        except CvBridgeError as error:
             print(error)
    
    def process_image(self):
        DELAY = 4.0 # Time delay (s)
        if self.get_clock().now() - self.start_time > Duration(seconds=DELAY):
            print(f"Image shape: {self.cv_image.shape}")
            print(f"Image type: {self.cv_image.dtype}")
            time_str = str(self.get_clock().now())
            cv2.imwrite(f'data/raw_images/{time_str}.jpg', self.cv_image)
            

def main(args=None):
    rclpy.init(args=args) # Start ROS2 communications
    node = ImageSaver() # Create node
    rclpy.spin(node) # Execute node
    node.destroy_node() # Destroy node explicitly (optional - otherwise it will be done automatically when garbage collector destroys the node object)
    rclpy.shutdown() # Shutdown ROS2 communications


if __name__ == "__main__":
    main()