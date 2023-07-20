#!/usr/bin/env python3
import rospy
from seed_robotics.msg import *
from sensor_pkg.msg import *
import time

# Initialize global variables to continuously store to incoming information.
joint_name = [
    "r_th_adduction",
    "r_th_flexion",
    "r_ix_flexion",
    "r_middle_flexion",
    "r_ring_ltl_flexion",
]
joint_position = [0, 0, 0, 0, 0]


# Define a function to fill the message 'final_message' to send based on lists of names, target position values and target speed values
def buildSpeedPosMsg(names, target_positions):
    # Initialize a list of JointSetSpeedPos messages, the length of the number of joints we want to send instruction to
    joint_list_msg = [JointSetSpeedPos() for i in range(len(names))]
    # Fill up that list with the informations about name, position and speed that are listed above
    for name, position, joint in zip(names, target_positions, joint_list_msg):
        joint.name = name
        joint.target_pos = position
        joint.target_speed = 10

    # Declare, fill up and return an instance of JointListSetSpeedPos message, ready to be sent
    final_message = JointListSetSpeedPos()
    final_message.joints = joint_list_msg
    pub.publish(final_message)
    # print(final_message)


# Process data that come from the Camera.
def cameraCallback(msgs):
    joint_position[0] = msgs.thumb_adduction
    joint_position[1] = msgs.thumb_flexion
    joint_position[2] = msgs.index
    joint_position[3] = msgs.middle
    joint_position[4] = msgs.ring
    buildSpeedPosMsg(joint_name, joint_position)


# Initialize a ROS Node
rospy.init_node("camera_control_hand_node", anonymous=False)

# Subscribe to the 'mediapipe_hand_topic' Topic to receive messages that will be processed by the cameraCallback function.
# Note that the Topic name MUST be 'mediapipe_hand_topic' so that the data can be processed.
rospy.Subscriber("mediapipe_hand_topic", MediaPipe, cameraCallback)

# Initialize a Publisher to the 'R_speed_position' Topic. This MUST be the name of the topic so that the data can be processed.
# The publisher will publish JointListSetSpeedPos messages.
pub = rospy.Publisher("R_speed_position", JointListSetSpeedPos, queue_size=10)

# Sleeping for 1sec to let ROS initialize
time.sleep(1)

# Main Loop-------------------------------------------------------------
while not rospy.is_shutdown():
    print(">>> send joint position to hand: ", joint_position)
    time.sleep(0.5)
# end of Main loop------------------------------------------------------
