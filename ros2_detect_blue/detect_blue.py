# BSD 3-Clause License

# Copyright (c) 2023, Matthew Lock

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

        # Get parameters
        self.declare_parameter('camera_topic', '/camera')
        self.declare_parameter('verbose', False)

        self.camera_topic = self.get_parameter('camera_topic').value
        self.verbose = self.get_parameter('verbose').value

        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 1)
        self.get_logger().info('Subscribed to ' + self.camera_topic)

        plt.ion() # Turn on interactive mode
        self.fig, self.axs = plt.subplots(1, 3, figsize=(10, 5))

        # Detection method
        self.declare_parameter('detection_method', 0)
        self.detection_method = self.get_parameter('detection_method').value

        # Declare boolean publisher and time to publish every 1 second
        self.blue_detected = False
        self.blue_detected_pub = self.create_publisher(Bool, 'blue_detected', 1)
        self.timer = self.create_timer(1, self.timer_callback)

    def timer_callback(self):
        # Publish the blue detected boolean
        msg = Bool()
        msg.data = self.blue_detected
        self.blue_detected_pub.publish(msg)

    def image_callback(self, msg):

        # Log callback 
        if self.verbose:
            self.get_logger().info('Image received')

        # Convert the ROS Image message to a numpy array
        image = rnp.numpify(msg)
        image = image[:, :, :3]
        original_image = image.copy()
        
        # Apply the blue detection algorithm

        # Case 0: Simple HSV thresholding
        debug_string = ''
        if self.detection_method == 0:
            debug_string = 'Simple HSV thresholding'
            mask, thresholded_image, blue_detected  = BlueDetectionAlgorithm.simple_HSV_theshhold(image=image)

        # Case 1: HSV thresholding wih k-means clustering to remove sky
        elif self.detection_method == 1:
            debug_string = 'HSV thresholding with k-means clustering to remove sky'
            mask, thresholded_image, blue_detected  = BlueDetectionAlgorithm.simple_HSV_theshhold_with_k_clustering_to_remove_sky(image=image)

        # Case 2: HSV thresholding with k-means and coverage thresholding
        elif self.detection_method == 2:
            debug_string = 'HSV thresholding with k-means and coverage thresholding'
            mask, thresholded_image, blue_detected  = BlueDetectionAlgorithm.simple_HSV_theshhold_with_k_clustering_and_coverage_thresholding(image=image)

        # Default case: Simple HSV thresholding
        else:
            debug_string = 'Simple HSV thresholding'
            mask, thresholded_image, blue_detected  = BlueDetectionAlgorithm.simple_HSV_theshhold(image=image)

        # Log the debug string
        if self.verbose:
            self.get_logger().info(debug_string)

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