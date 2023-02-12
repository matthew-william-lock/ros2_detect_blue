import rclpy
from rclpy.node import Node

import numpy as np
import ros2_numpy as rnp

# Messages
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

# Tools
from .blue_detection_algorithm import BlueDetectionAlgorithm
import matplotlib.pyplot as plt
import cv2

class DetectBlueNode(Node):
    def __init__(self):
        super().__init__('detect_blue')
        self.get_logger().info('detect_blue node is starting')
        self.image_sub = self.create_subscription(Image, '/camera', self.image_callback, 1)
        plt.ion() # Turn on interactive mode
        self.fig, self.axs = plt.subplots(1, 3, figsize=(10, 5))

        # Detection method
        self.declare_parameter('detection_method', 0)
        self.detection_method = self.get_parameter('detection_method').value

        # Declare boolean publisher and time to publish every 1 second
        self.blue_detected = False
        self.blue_detected_pub = self.create_publisher(Bool, '/blue_detected', 1)
        self.timer = self.create_timer(1, self.timer_callback)

    def timer_callback(self):
        # Publish the blue detected boolean
        msg = Bool()
        msg.data = self.blue_detected
        self.blue_detected_pub.publish(msg)

    def image_callback(self, msg):

        # Log callback 
        self.get_logger().info('Image received')

        # Convert the ROS Image message to a numpy array
        image = rnp.numpify(msg)
        image = image[:, :, :3]
        original_image = image.copy()
        
        # Apply the blue detection algorithm

        # Case 1: Simple HSV thresholding
        if self.detection_method == 0:
            self.get_logger().info('Using simple HSV thresholding')
            mask, thresholded_image, blue_detected  = BlueDetectionAlgorithm.simple_HSV_theshhold(image=image)

        # Case 2: HSV thresholding wih k-means clustering to remove sky
        elif self.detection_method == 1:
            self.get_logger().info('Using HSV thresholding with k-means clustering to remove sky')
            mask, thresholded_image, blue_detected  = BlueDetectionAlgorithm.simple_HSV_theshhold_with_k_clustering_to_remove_sky(image=image)

        # Case 3: HSV thresholding with k-means and coverage thresholding
        elif self.detection_method == 2:
            self.get_logger().info('Using HSV thresholding with k-means and coverage thresholding')
            mask, thresholded_image, blue_detected  = BlueDetectionAlgorithm.simple_HSV_theshhold_with_k_clustering_and_coverage_thresholding(image=image)

        # Default case: Simple HSV thresholding
        else:
            mask, thresholded_image, blue_detected  = BlueDetectionAlgorithm.simple_HSV_theshhold(image=image)

        # Conver images to BGR
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

        # Add text to the image
        if blue_detected:
            text = "Blue Detected"
            original_image = BlueDetectionAlgorithm.add_text_to_image(image=original_image, text=text)

        else:
            text = "No Blue Detected"
            original_image = BlueDetectionAlgorithm.add_text_to_image(image=original_image, text=text)

        # Change value of blue detected
        # conver numpy boolean to python boolean
        blue_detected = bool(blue_detected)
        self.blue_detected = blue_detected

        self.axs[0].imshow(original_image, cmap='gray')
        self.axs[0].set_title("Original Image")
        self.axs[1].imshow(mask, cmap='gray')
        self.axs[1].set_title("Mask")
        self.axs[2].imshow(thresholded_image, cmap='gray')
        self.axs[2].set_title("Masked Image")
        plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)

    node = DetectBlueNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()