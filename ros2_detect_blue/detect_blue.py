import rclpy
from rclpy.node import Node

import numpy as np
import ros2_numpy as rnp

# Messages
from sensor_msgs.msg import Image

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

    def image_callback(self, msg):

        # Log callback 
        self.get_logger().info('Image received')

        # Convert the ROS Image message to a numpy array
        image = rnp.numpify(msg)

        original_image = image.copy()
        
        # Apply the blue detection algorithm
        # Case 1: Simple HSV thresholding
        # Default case: Simple HSV thresholding

        if self.detection_method == 0:
            mask, thresholded_image, blue_detected  = BlueDetectionAlgorithm.simple_HSV_theshhold(image=image)

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