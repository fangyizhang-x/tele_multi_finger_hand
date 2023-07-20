## Demo code: Control RH8D Robot Hand (hand) with 2D USB Camera

##### demo code path: /seed_robotics/camera_control_hand_demo

##### MediaPipe installation: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python

## ---------Steps as follows---------

## step 1: configure and run `RH8D Robot Hand (hand)`

### 1.1 configure USB ports

#### 1.1.1 check the USB port number each time when plugging in the USB of PC.

open the terminal and run the code `ll /dev/ | grep ttyUSB` to check USB ports:

```
crw-rw----   1 root    dialout 188,   0 Apr  4 16:42 ttyUSB0
crw-rw----   1 root    dialout 188,   1 Apr  4 16:42 ttyUSB1
```

#### 1.1.2 use `vim` to open the file, then change the `value` in the file from `16` to `1`. One usb port for seed hand control and another one for sensors.

```
sudo vim /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
sudo vim /sys/bus/usb-serial/devices/ttyUSB1/latency_timer
```

**Note:**

- If your PC already has other USB devices plugged in, the port numbers may not be 0 and 1. So, you should change the port number to a specific number.
- Besides, you should change the port number to the specific number in file named `RH8D_R.yaml` inside `"/seed_robotics/config"` folder.
- if error`vim: command not found` happens , you can install `vim` via command `sudo apt install vim`

For example: if your USB ports are USB3 and USB4:

- change the port number inside `/seed_robotics/config/RH8D_R.launch` and `/sensor_pkg/config/sensors_right.yaml` files respectively: `port: "/dev/ttyUSB3"` and `port: "/dev/ttyUSB4`"

### 1.2 run the `launch` file to start the hand

`roslaunch seed_robotics RH8D_R.launch`

## step 2: run the `mediapipe_v4_30Mar.py` to publish a topic

plug your USB camera into your computer, and then publish the rostopic named `mediapipe_hand_topic`.
**code show as below (MediaPipe is a custom message which is inside `/seed_robotics/msg` folder):**
`rospy.Publisher('mediapipe_hand_topic', MediaPipe, queue_size=10)`

## step 3: run the `draw_seed_hand.py` to draw the graphic

## step 4: run the `camera_control_hand_demo_24Mar.py`

- subscribe a topoic named `mediapipe_hand_topic` to ge the control commands from step 2.
  `rospy.Subscriber('mediapipe_hand_topic', MediaPipe, cameraCallback)`
- then publish a topic named `R_speed_position` to control the joint speed and position.
  `rospy.Publisher('R_speed_position', JointListSetSpeedPos, queue_size = 10)`
