# Copyright 2023 Matthew Lock
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BLUE DETECTION ALGORITHM"""

import cv2
import numpy as np

# Types
from numpy import ndarray

class BlueDetectionAlgorithm:
    """BLUE DETECTION ALGORITHM"""

    def __init__(self):
        """INITIALISE BLUE DETECTION ALGORITHM"""
        self.name = "Blue Detection Algorithm"

    @staticmethod 
    def add_text_to_image(image: ndarray, text: str):
        """
        Add text to the bottom left of the image

        Parameters
        ----------
        image : ndarray
            The image to add text to
        text : str
            The text to add to the image

        Returns
        -------
        ndarray
            The image with the text added
        """

        # Get the image dimensions
        height, width, _ = image.shape

        # Add the text to the image
        cv2.putText(image, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return image

    @staticmethod
    def simple_HSV_theshhold(image: ndarray):

        """
        Simple thresholding of the image based on the blue color range. Works by detecting the presence of blue in the image 
        by finding the pixels that fall within the threshold range.

        Parameters
        ----------
        image : ndarray
            The image to be processed

        Returns
        -------
        ndarray
            The mask of the image
        ndarray
            The thresholded image
        bool
            Boolean value indicating if blue was detected
        """

        # Convert the image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the blue color range
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(image, image, mask=mask)

        # Return boolean value indicating if blue was detected
        blue_detected = np.any(mask)

        return mask, res, blue_detected