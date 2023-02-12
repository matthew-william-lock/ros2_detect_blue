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

        # Set the text scale factor to 20% of the image height
        text_scale = 0.002 * height
        text_scale*= 2

        # Add the text to the image
        cv2.putText(image, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), 1)

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

    @staticmethod
    def simple_HSV_theshhold_with_k_clustering_to_remove_sky(image: ndarray, K= 8):

        """
        Simple thresholding of the image based on the blue color range. Works by detecting the presence of blue in the image 
        by finding the pixels that fall within the threshold range. Then the images is clustered using K means 
        and the clusters with centroids in the top 10% of the image are removed.

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

        # Use the simple HSV thresholding algorithm
        mask, image, _ = BlueDetectionAlgorithm.simple_HSV_theshhold(image)

        # Cluster the image using K means
        height, width, _ = image.shape
        reshaped_image = image.reshape((-1, 3))
        reshaped_image = np.float32(reshaped_image)

         # Perform k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, label, center = cv2.kmeans(reshaped_image, K, None, criteria, 10, flags)

        # Calculate the centroids of each cluster
        def calculate_centroids(data, labels, k):
            centroids = np.zeros((k, data.shape[1]), dtype=np.float32)
            for i in range(k):
                points = data[labels.ravel()==i]
                centroids[i] = np.sum(points, axis=0) / points.shape[0]
            return centroids
    
        centroids = calculate_centroids(reshaped_image, label, K)

        # Remove the segments with centroids in the top 10% of the image
        threshold = int(height * 0.1)
        segmented_filtered = np.zeros((height*width, 3), dtype=np.uint8)
        for i, c in enumerate(centroids):
            if c[1] >= threshold:
                continue
            segmented_filtered[label.ravel()==i] = center[i]
        
        # Reshape the filtered segmented image
        segmented_filtered = segmented_filtered.reshape((height, width, 3))

        # Threshold again to get the mask
        mask, image, _ = BlueDetectionAlgorithm.simple_HSV_theshhold(segmented_filtered)

        # Return boolean value indicating if blue was detected
        blue_detected = np.any(mask)

        return mask, image, blue_detected

    @staticmethod
    def simple_HSV_theshhold_with_k_clustering_and_coverage_thresholding(image: ndarray, K= 8, coverage=0.25):
        """
        Simple thresholding of the image based on the blue color range. Works by detecting the presence of blue in the image
        by finding the pixels that fall within the threshold range. Then the images is clustered using K means
        and the clusters with centroids in the top 10% of the image are removed. Finally the small segments are by seeing what percentage of the image
        each segment covers and removing the segments that cover less than the coverage threshold.
        
        Parameters
        ----------
        image : ndarray
            The image to be processed
        K : int, optional
            The number of clusters to use in the K means clustering, by default 8
        coverage : float, optional
            The percentage of the image that a segment must cover to be considered a valid segment, by default 0.25
            
        Returns
        -------
        ndarray
            The mask of the image
        ndarray
            The thresholded image
        bool
            Boolean value indicating if blue was detected
        """

        # Use the simple HSV thresholding algorithm
        mask, image, _ = BlueDetectionAlgorithm.simple_HSV_theshhold(image)

        # Cluster the image using K means
        height, width, _ = image.shape
        reshaped_image = image.reshape((-1, 3))
        reshaped_image = np.float32(reshaped_image)

         # Perform k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, label, center = cv2.kmeans(reshaped_image, K, None, criteria, 10, flags)

        # Calculate the centroids of each cluster
        def calculate_centroids(data, labels, k):
            centroids = np.zeros((k, data.shape[1]), dtype=np.float32)
            for i in range(k):
                points = data[labels.ravel()==i]
                centroids[i] = np.sum(points, axis=0) / points.shape[0]
            return centroids
    
        centroids = calculate_centroids(reshaped_image, label, K)

        # Remove the segments with centroids in the top 10% of the image
        threshold = int(height * 0.1)
        segmented_filtered = np.zeros((height*width, 3), dtype=np.uint8)
        for i, c in enumerate(centroids):
            if c[1] >= threshold:
                continue
            segmented_filtered[label.ravel()==i] = center[i]

        # Calculate the percentage of the image each segment covers
        def calculate_segment_coverage(data, labels, k):
            coverage = np.zeros(k, dtype=np.float32)
            for i in range(k):
                points = data[labels.ravel()==i]
                coverage[i] = points.shape[0] / data.shape[0]
            return coverage

        # Calculate the segment coverage
        segment_coverage = calculate_segment_coverage(reshaped_image, label, K)

        # Remove the segments that cover less than the coverage threshold
        for i, c in enumerate(segment_coverage):
            if c < coverage:
                segmented_filtered[label.ravel()==i] = 0
        
        # Reshape the filtered segmented image
        segmented_filtered = segmented_filtered.reshape((height, width, 3))

        # Threshold again to get the mask
        mask, image, _ = BlueDetectionAlgorithm.simple_HSV_theshhold(segmented_filtered)

        # Return boolean value indicating if blue was detected
        blue_detected = np.any(mask)

        return mask, image, blue_detected
