# ros2_detect_blue

This is a ROS2 package that detects "blue pixels" in an image published by a camera on the ```/camera``` topic. The methods used to detect the blue pixels are detailed below and can be changed using the runtime parameters.

> The package has been tested on ROS2 Humble.

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
ros2 run ros2_detect_blue detect_blue
```