# ros2_detect_blue

This is a ROS2 package that detects "blue pixels" in an image published by a camera on the ```/camera``` topic. The methods used to detect the blue pixels are detailed below and can be changed using the runtime parameters.

> The package has been tested on ROS2 Humble.

After running the node, the following topics will be available:

* ```/blue_detected``` - A boolean value that is true if blue is detected in the image.

## Getting Started

### Prerequisites

* cv2

```bash
sudo apt-get install python3-opencv
```

* numpy

```bash
sudo apt-get install python3-numpy
```

## Running the Node

```bash
ros2 run ros2_detect_blue detect_blue --ros-args -p detection_method:=0
```

This will create a simple video feed as shown below. The video feed will show three windows:
* The original image from the camera with text indicating if blue is detected.
* The image mask
* The image after the mask has been applied

Table below shows the available detection methods and their corresponding algorithm. An in depth description of the algorithm can be found in subsequent sections.

| Method | Algorithm | Description |
| --- | --- | --- |
| 0 | simple_HSV_theshhold| Converts the RGB image to an HSV image and return true if any blue is found within a certain range. |
| 1 | simple_HSV_theshhold with K-mean for sky removal | Applies K-mean clustering to the simple_HSV_theshhold image to remove the sky. This is done by calculating the centroids of each cluther and removing the clusters with centroids in the upper 10% of the image. |
| 2 | simple_HSV_theshhold with K-mean for sky removal and coverage threshold | Calculate the percentage of each cluster and remove the clusters with a coverage percentage below a certain threshold. Only blue clusters are returned |

### Simple HSV Threshold

This method converts the RGB image to an HSV image and return true if any blue is found within a certain range. The range is defined by the ```lower_blue``` and ```upper_blue``` parameters. The HSV colour space is more suitable for colour detection than the RGB colour space because it separates the colour information from the intensity information. Therefore, the HSV colour space is more robust to changes in lighting conditions.

Considerations:
- The ```lower_blue``` and ```upper_blue``` parameters are highly depentent on the calibration of the camera. The values used in this package are for the camera used in the project. The values may need to be changed for other cameras.
- There may be false positives if the camera is pointed at a blue object that is not the sky.
- There may be edge cases where 'weird' colours are detected as blue.

### Simple HSV Threshold with K-mean for Sky Removal

This method employs the same thresholding method as the simple HSV threshold method. However, it also applies K-mean clustering to the simple HSV threshold image to remove the sky. This is done by calculating the centroids of each cluther and removing the clusters with centroids in the upper 10% of the image.

Considerations:
- The ```lower_blue``` and ```upper_blue``` parameters are highly depentent on the calibration of the camera. The values used in this package are for the camera used in the project. The values may need to be changed for other cameras.
- Objects that enter the frame from the top of the image may be falsely classified as sky and removed.
- K-mean clustering is computationally expensive. Therefore, this method may not be suitable for real-time applications.
- K-means is not guaranteed to correctly segment objects in the image and is therfore not a robust method for general applications (scenarios where K-means can be purposefully tuned for one specific use-case).

### Simple HSV Threshold with K-mean for Sky Removal and Coverage Threshold

This employs both the simple HSV threshold and K-mean methods. However, it also calculates the percentage of each cluster and removes the clusters with a coverage percentage below a certain threshold. Only blue clusters are returned. 

> By defaault, the threshold is set to 0.25 (25%) of the image. This means that any cluster that cover more than 25% of the image will be detected.

Considerations:
- The ```lower_blue``` and ```upper_blue``` parameters are highly depentent on the calibration of the camera. The values used in this package are for the camera used in the project. The values may need to be changed for other cameras.
- Objects that enter the frame from the top of the image may be falsely classified as sky and removed.
- K-mean clustering is computationally expensive. Therefore, this method may not be suitable for real-time applications.
- K-means is not guaranteed to correctly segment objects in the image. Consequently, single objects may be falsely classified as multiple clusters and can be removed as they do not meet the coverage threshold. This is especially true for objects that are close to each other or are close to the camera.
