#!/usr/bin/env python3

# NOTICE : This sample is the same as the user_sample 7, but with one more threshold than the current, which is the threshold force value mesured by the fingertips sensors

# This sample is done for a RH8D Right Hand
# You can adapt it to a right Hand by changing the 'l' by a 'r' on joint names, and changing the R_ prefixes to ROS Topics by L_ prefixes
# User sample code High Level Logic : When an object is close to the Hand, the fingers close to grab the object. Once it's grabbed, the finger open themselves 5seconds later
# Several things are done here :
# - Get real-time data for each joint
# - Get real-time data from the IR sensor in the palm of the hand
# - Get real-time data from the fingertips pressure sensors
# - Send the instruction to close the index, the ring and the little fingers
# - Send the instruction to close the thumb
# - Continuously checking if any joint is too stressed, i.e. if its current goes above the current limit hardcoded to 300mA
# - Continuously checking if any fingertip sensor goes past the hardcoded force limit set to 1000mN
# - If a joint is too stressed, send an instruction to set its target position to its present position
# - When every joint have their present position equal to their target position -> open the finger and let the object go


# sudo vim /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
# roslaunch seed_robotics RH8D_R.launch


import rospy
import std_msgs.msg
from seed_robotics.msg import *
from sensor_pkg.msg import *
import time
import csv
from armer import *

# Hardcoded value of the current limit, if a joint goes above that limit it must stop forcing
# CURRENT_LIMIT = 300 #mAmp
# Hardcoded value of the fingertip sensor limit, if a joint goes above that limit it must stop forcing
SENSOR_FORCE_TOLERANCE = 50 #mN
SENSOR_FORCE_TARGET = 200 #mN
TIME_DELAY       = 0.02 #seconds
POSITION_CHANGE  = 1

ACCEPTABLE_HIGH = SENSOR_FORCE_TARGET + SENSOR_FORCE_TOLERANCE
ACCEPTABLE_LOW  = SENSOR_FORCE_TARGET - SENSOR_FORCE_TOLERANCE
# Control class to be able to change the IR sensor value in the callback function
# Including 2 flags to control the main loop
class Control:
    def __init__(self):
        self.IR_sensor_value = 254
        self.start_flag      = False
        self.step2_flag      = False


# Initialize an instance of the Control class
control = Control()


# Initialize variables to contunuisly store to incoming informations
joint_list = [LoneJoint() for i in range (8)] # We will only need informations about 5 joints : r_th_adduction, r_th_flexion, r_ix_flexion, r_middle_flexion, r_ring_ltl_flexion
sensor_list = []                              # Init a list that will be filled with lone_sensor messages
# Initialize variables and structures to send messages
names_step_1            = ['r_th_adduction','r_ix_flexion','r_middle_flexion','r_ring_ltl_flexion']# Name of the joints we want to move first
target_positions_step_1 = [4095, 4095, 4095, 4095] # Maximum position value : closed position
target_speeds_step_1    = [50, 50, 50, 50]  # Speed 0 : Highest speed. Speed 50 : Low speed

names_step_2            = ['r_th_flexion']# Name of the joint to move on the 2nd step of the hand closing
target_positions_step_2 = [4095] # Maximum position value : closed position
target_speeds_step_2    = [50] # Speed 50 : Low speed

names_step_3            = ['r_th_adduction','r_th_flexion','r_ix_flexion','r_middle_flexion','r_ring_ltl_flexion'] # Names of all the joints to move. Step 3 will be to open each finger
target_positions_step_3 = [0, 0, 0, 0, 0] # Minimum position value : open position
target_speeds_step_3    = [50, 50, 50, 50, 50] # Speed 10 : very low speed

# Define a function to get joint's name from its fingertip sensor ID
# The mapping is the following :
# ID = 0 -> thumb flexion
# ID = 1 -> index flexion
# ID = 2 -> middle finger flexion
# ID = 3 or 4 -> ring and little finger flexion (3 is on the ring finger, 4 is on the little finger)
# Here, all joints are mapped to left hand joints, don't forget to check if you are using a right or a left hand
# Note that you have information about the hand polarity in the AllSensors messages
def getNameFromSensorID(id):
    if id == 0:
        return 'r_th_flexion'
    elif id == 1:
        return 'r_ix_flexion'
    elif id == 2:
        return 'r_middle_flexion'
    elif id == 3 or id == 4:
        return 'r_ring_ltl_flexion'
    else:
        rospy.logwarn("Couldn't match sensor ID %d with its joint, joint name set to 'None'" % id)
        return 'None'

# Define a function to fill the message 'final_message' to send based on lists of names, target position values and target speed values
def buildSpeedPosMsg(names,target_positions):
    # Initialize a list of JointSetSpeedPos messages, the length of the number of joints we want to send instruction to
    joint_list_msg = [JointSetSpeedPos() for i in range(len(names))]
    # Fill up that list with the informations about name, position and speed that are listed above
    for name, position, joint in zip(names, target_positions, joint_list_msg):
        joint.name = name
        joint.target_pos = position
        joint.target_speed = 50
    # Declare, fill up and return an instance of JointListSetSpeedPos message, ready to be sent
    final_message = JointListSetSpeedPos()
    final_message.joints = joint_list_msg
    pub.publish(final_message)
    # print(final_message)
    return

# Callback function called when a AllJoints message is published.
# Will fill up the joint_list with each LoneJoint messages that are received that interests us. (i.e thumb adduction, thumb flexion, index flexion and ring/little flexion)
def jointsCallback(joints_data):
    for joint in joints_data.joints:
        if joint.name == 'r_w_rotation':
            joint_list[0] = joint
        if joint.name == 'r_w_adduction':
            joint_list[1] = joint
        if joint.name == 'r_w_flexion':
            joint_list[2] = joint
        # control the fingers below
        if joint.name == 'r_th_adduction':
            joint_list[3] = joint
        if joint.name == 'r_th_flexion':
            joint_list[4] = joint
        if joint.name == 'r_ix_flexion':
            joint_list[5] = joint
        if joint.name == 'r_middle_flexion':
            joint_list[6] = joint
        if joint.name == 'r_ring_ltl_flexion':
            joint_list[7] = joint
        '''
        Number of joints : 8
        Joint name : r_w_rotation
        Joint ID : 31
        Joint name : r_w_adduction
        Joint ID : 32
        Joint name : r_w_flexion
        Joint ID : 33
        Joint name : r_th_adduction
        Joint ID : 34
        Joint name : r_th_flexion
        Joint ID : 35
        Joint name : r_ix_flexion
        Joint ID : 36
        Joint name : r_middle_flexion
        Joint ID : 37
        Joint name : r_ring_ltl_flexion
        Joint ID : 38
        '''

# Callback function called when a message about the main board is published
# Will update the value of the palm IR sensor
def mainBoardCallback(main_board_data):
    for board in main_board_data.boards :
        if board.id == 30:
            control.IR_sensor_value = board.palm_IR_sensor
    #print("IR Sensor value = %d" % IR_sensor_value)

# Callback function called when a AllSensor message is published
# Update the values in the sensor_list of lone_sensor messages
def sensorCallback(sensor_data):
    # If the list is empty : fill it
    if len(sensor_list) == 0:
        for sensor in sensor_data.data:
            sensor_list.append(sensor)
    # Else : update the list with the new values
    else:
        for index, sensor in enumerate(sensor_data.data):
            sensor_list[index] = sensor

# Initialize a ROS Node
rospy.init_node('Joint_listener', anonymous = False)
# Subscribe to the Joints Topic to receive AllJoints messages that will be processed by the jointsCallback function
# Note that the Topic name MUST be 'Joints'
rospy.Subscriber("R_Joints", AllJoints, jointsCallback)
# Subscribe to the Main_Boards Topic to receive AllMainBoards messages that will be processed by the mainBoardCallback function
# Note that the Topic name MUST be 'Main_Boards'
rospy.Subscriber('R_Main_Boards', AllMainBoards, mainBoardCallback)
# Subscribe to the AllSensors Topic to receive AllSensors messages that will be processed by the sensorCallback function
# Note that the Topic name MUST be 'AllSensors'
rospy.Subscriber('R_AllSensors',AllSensors,sensorCallback)
# Initialize a Publisher to the 'speed_position' Topic. This MUST be the name of the topic so that the data can be processed
# The publisher will publish JointListSetSpeedPos messages
pub = rospy.Publisher('R_speed_position', JointListSetSpeedPos, queue_size = 10)

# # Initialize an instance of the 'Frank Emika Panda' armer control class
gbc = GentleBenchmarkController()

# Define a function that takes a LoneJoint message instance in argument
# It will publish a message to that joint to set its target position to its present position
# The idea is to stop stressing the joint
def stopStressing(joint):
    # Getting the joint's present position
    target_pos = joint.present_position
    # Declare a list of 1 JointSetSpeedPos element
    joints = [JointSetSpeedPos()]
    # Fill the JointSetSpeedPos instance with joint's name, new target position and target_speed
    joints[0].name = joint.name
    joints[0].target_pos = target_pos
    joints[0].target_speed = -1         # If targert speed = -1, then the parameter will be ignored
    # Declare an instance of JointListSetSpeedPos that will be the message to send
    message = JointListSetSpeedPos()
    # Fill that message and publish it
    message.joints = joints
    pub.publish(message)

def decreaseStress(joint,pos_change):
    # Getting the joint's present position
    if joint.present_position < 201:
        print("Trying to decrease pos on joint %s that already has present position to %d" % (joint.name,joint.present_position))
        return
    target_pos = joint.present_position - pos_change
    # Declare a list of 1 JointSetSpeedPos element
    joints = [JointSetSpeedPos()]
    # Fill the JointSetSpeedPos instance with joint's name, new target position and target_speed
    joints[0].name = joint.name
    joints[0].target_pos = target_pos
    joints[0].target_speed = -1  # If targert speed = -1, then the parameter will be ignored
    # Declare an instance of JointListSetSpeedPos that will be the message to send
    message = JointListSetSpeedPos()
    # Fill that message and publish it
    message.joints = joints
    pub.publish(message)

def increaseStress(joint,pos_change):
    # Getting the joint's present position
    if joint.present_position > 3894:
        list_joints_too_far.append(joint.name)
        print("Trying to increase pos on joint %s that already has present position to %d" % (joint.name,joint.present_position))
        return
    target_pos = joint.present_position + pos_change
    # Declare a list of 1 JointSetSpeedPos element
    joints = [JointSetSpeedPos()]
    # Fill the JointSetSpeedPos instance with joint's name, new target position and target_speed
    joints[0].name = joint.name
    joints[0].target_pos = target_pos
    joints[0].target_speed = -1  # If targert speed = -1, then the parameter will be ignored
    # Declare an instance of JointListSetSpeedPos that will be the message to send
    message = JointListSetSpeedPos()
    # Fill that message and publish it
    message.joints = joints
    pub.publish(message)

def computeGain(abs_val):
    gain = int(abs(abs_val - SENSOR_FORCE_TARGET))
    if gain > 200:
        return 200
    else:
        return gain


def demoTestJoints():
    # Initialize variables and structures to send messages
    # Declaring a list to store the names of the joints we want to send command to
    joint_names = ['r_w_rotation','r_w_adduction','r_w_flexion','r_th_adduction','r_th_flexion','r_ix_flexion','r_middle_flexion','r_ring_ltl_flexion']
    # Declaring a list to be filled with the target positions read
    target_position_list = []

    # Open and read the csv file with the list of commands
    # with open('src/seed_robotics/user_samples/KEYFRAMES_IROS2019_RH8Donly.csv', 'r') as csv_file:
    with open('ttt.csv', 'r') as csv_file:
    # with open('KEYFRAMES_IROS2019_RH8Donly.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        # Go through each line in the csv file
        for row in reader:
            # Clear the target position list between each iteration
            target_position_list.clear()
            print(row)
            # If a line doesn't start with a 'K' ignore it
            if row[0] == 'K':
                for i in range(2,len(row)):
                    # Last element is the time to wait between each send
                    if i == 10:
                        wait_ms = row[i]
                    else:
                        # Every other one is a target position
                        target_position_list.append(int(row[i]))
                print(target_position_list)
                print('------------------------------------------------------------------------------')
                buildSpeedPosMsg(joint_names,target_position_list)
                time_sleep_s = int(wait_ms) * 0.001
                time.sleep(time_sleep_s)
        csv_file.close()



# Declaring a empty list to store future stressed joints
list_joints_too_far = []

# Sleeping for 1sec to let ROS initialize
time.sleep(1)
counter = 0 # count the times (check each sensor's value to see if the value is equal to the threshold), ACCEPTABLE_LOW  = SENSOR_FORCE_TARGET - SENSOR_FORCE_TOLERANCE= 150mN
outer = False # control the while loop for handshake
timer_waiting = 0 # control the speed of printing '>>> Waiting for handshake......'

# Main Loop---------------------------------------------------------------------------------------------------------------------------------------------------
while not rospy.is_shutdown():
#--------------------------------------------------------------
    if not outer:
        for i in range(3):
            time.sleep(1)
            print('>>> Waiting to start... ')
        print('>>> Move to standby position')
        gbc.move_to_somewhere('hand1') # move to handshake standby position
        # -----------------------------------------
        print('>>> To test each joint:')
        demoTestJoints() # test each joints of the hand
        print('>>> End of joint test.')
        print('>>> Move to handshake position')
        gbc.move_to_somewhere('hand2') # move to handshake position
#--------------------------------------------------------------
    while not outer: # while loop for handshake
        # Step 1: If start flag is false, then try to do the first step
        if control.start_flag is False:
            timer_waiting +=1
            time.sleep(0.001)
            if timer_waiting == 500:
                timer_waiting = 0
                print('>>> Waiting for handshake......') # print the message every 0.5 second
            # If the IR sensor value is below 20, then there is an object to grab -> do the first step
            if control.IR_sensor_value < 35:
                # Create the message to send that will close the index, ring and little finger. It will also rotate the thumb in prevision of the next step
                print('Step1: ')
                buildSpeedPosMsg(names_step_1,target_positions_step_1)
                # Setting start flag to True : first step is done
                control.start_flag = True
                # Sleep for 1sec to not interfere between messages
                time.sleep(1)

        # Step 2: If start flag is true and step2_flag is false, then it is time for step 2
        elif control.step2_flag is False:
            # Create the message to send that will close the thumb and publish it
            print('Step2: ')
            buildSpeedPosMsg(names_step_2,target_positions_step_2)
            # Setting step2_flag to True : step 2 is done
            control.step2_flag = True
            time.sleep(1)

        # Step 3: check each sensor's value
        if control.step2_flag is True:
            while counter < 2:
                print('Step3: adjustment')
                # Continuously check each sensor's value to see if the value is above the threshold
                for sensor in sensor_list:
                    # If the id is 3 or 4, the sensor is on the ring or little finger
                    # If so, we divide the value by 2 because only 1 joint corresponds to these 2 sensors, so we average the value
                    joint_name = getNameFromSensorID(sensor.id)
                    if sensor.id == 3:
                        abs_val = (sensor.abs + sensor_list[4].abs)/2
                    elif sensor.id == 4:
                        abs_val = (sensor.abs + sensor_list[3].abs)/2
                    else:
                        abs_val = sensor.abs

                    if abs_val > ACCEPTABLE_HIGH :
                    # If the value is above the threshold, get the corresonding joint's name
                        corresponding_joint = [joint for joint in joint_list if joint.name == joint_name]
                        if corresponding_joint:
                        # Compute the gain
                            gain = computeGain(abs_val)
                        # Set its target position to a lower value
                            decreaseStress(corresponding_joint[0],gain)
                            time.sleep(TIME_DELAY)
                    elif abs_val < ACCEPTABLE_LOW:
                    # If the value is below the threshold
                        if not (joint_name in list_joints_too_far):
                    # If that joint is not already at its minimum position
                    # Find its name
                            corresponding_joint = [joint for joint in joint_list if joint.name == joint_name]
                            if corresponding_joint:
                        # Compute the gain
                                gain = computeGain(abs_val)
                        # Increasae its target position
                                increaseStress(corresponding_joint[0],gain)
                                time.sleep(TIME_DELAY)
                    else: # abs_val == ACCEPTABLE_LOW (=150 mN)
                        counter += 1 # control the times of while loop (Step: 3): check each sensor's value to see if the value is equal to the threshold (150mN)
                        print('Counter: ', counter)
                        # If the value is inside our wanted pressure interval
                        corresponding_joint = [joint for joint in joint_list if joint.name == joint_name]
                        if corresponding_joint:
                            # Set its target position to its current position
                            stopStressing(corresponding_joint[0])
                            time.sleep(TIME_DELAY)
            outer = True
            print('>>> Exit Step3: adjustment (check each sensor\'s value).')
        
        if outer is True:
            print('>>> Shake hand: 1 second')
            print('>>> Shake hand: 2 second')
            time.sleep(1)
            print('>>> Complete the handshake.')
            buildSpeedPosMsg(names_step_3,target_positions_step_3) # send message to open the hand
            time.sleep(1)
            buildSpeedPosMsg(names_step_3,target_positions_step_3) # send message to open the hand
            gbc.move_to_somewhere('hand1') # return to standby position
    # end of the handshake task
    print('>>> The handshake task has been finished. [Ctrl+c] to kill the node.')
    time.sleep(1)
    break
#--------------------------------------------------------------
    