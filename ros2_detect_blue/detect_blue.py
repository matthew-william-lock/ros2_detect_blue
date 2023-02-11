import rclpy
from rclpy.node import Node

import numpy as np

import matplotlib.pyplot as plt
import cv2
from sensor_msgs.msg import Image

class DetectBlueNode(Node):
    def __init__(self):
        super().__init__('detect_blue')
        self.image_sub = self.create_subscription(Image, '/image', self.image_callback, 1)

    def image_callback(self, msg):
        # Convert the ROS Image message to a numpy array
        image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, msg.step)

        # Display the image using matplotlib
        plt.imshow(image)
        plt.show()

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