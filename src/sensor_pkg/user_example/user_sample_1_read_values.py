#!/usr/bin/env python3

# PLEASE NOTE : This samples assume you are using a RIGHT Hand
# To use a left hand you must replace the Topic prefix, i.e replace R_sensor_user_command by L_sensor_user_command
# Also change R_AllSensors to L_AllSensors
# User sample code that calibrates the sensor via a ROS message, recovers X Y Z sensor values and display them

import rospy
import std_msgs.msg
import serial, sys
from sensor_pkg.msg import *
import time
import math
# Callback function definition
def callback(sensor_data):

    # Initialize a list of lone_sensor messages to store the data that will be read
    sensor_values = [lone_sensor() for i in range(sensor_data.length)]

    for i, sensor in enumerate(sensor_data.data):
        if sensor.is_present == False:             # Check if the sensor is present
            sensor_values[i].id = None                          # if not : set id to None, sensor will be displayed as "Sensor None"
        else:
            sensor_values[i] = sensor                           # If sensor is present, then copy the informations in the lone_sensor message
        # Then print each sensor with its X Y Z coordinates
        print("\t Sensor-ID: {} \n".format(sensor_values[i].id))
        print("\t fx:{}, fy:{}, fz:{} \n".format(sensor_values[i].fx, sensor_values[i].fy, sensor_values[i].fz))
        sum = math.sqrt(sensor_values[i].fx**2 + sensor_values[i].fy**2 + sensor_values[i].fz**2)
        print("\t sum:{} \n".format(sum))
        # print("\t Sensor-abs: {} \n".format(sensor_values[i].abs))
        # print("\t Sensor-yaw: {} \n".format(sensor_values[i].yaw))
        # print("\t Sensor-pitch: {} \n".format(sensor_values[i].pitch))
        # print("\t Sensor-is-present: {} \n".format(sensor_values[i].is_present))
        # print("\t Sensor-is-3D: {} \n".format(sensor_values[i].is_3D))
k=4
def callback1(sensor_data):

    # Initialize a list of lone_sensor messages to store the data that will be read
    sensor_values = [lone_sensor() for i in range(sensor_data.length)]

    if sensor_data.data[1].is_present == False:             # Check if the sensor is present
        return # if not : set id to None, sensor will be displayed as "Sensor None"
    else:
        sensor_values = sensor_data.data                       # If sensor is present, then copy the informations in the lone_sensor message
        sum = math.sqrt(sensor_values[k].fx**2 + sensor_values[k].fy**2 + sensor_values[k].fz**2)
        print("\t fx:{}, fy:{}, fz:{}, sum:{}\n".format(sensor_values[k].fx, sensor_values[k].fy, sensor_values[k].fz, sum))
    

# Initialize ros node
rospy.init_node('user_listener_1', anonymous = True)

# Initialize a publisher to sensor_user_command to be able to send command to the sensors
pub = rospy.Publisher('R_sensor_user_command', sensor_user_command, queue_size = 10)

# Defining a sensor_user_command message to fill it with the needed Command
command = sensor_user_command()

# Setting the 'calibrate' field to True
command.calibrate = True

# Let 1s to ROS for the Node's initialization before publishing
time.sleep(1)

# Publish the command to the sensor_user_command topic.
pub.publish(command)

# Subscribe to the AllSensors Topic, in which informations read about sensors are published
rospy.Subscriber("R_AllSensors", AllSensors, callback1)

# Keeps the node listening until you stop the script manually
rospy.spin()
