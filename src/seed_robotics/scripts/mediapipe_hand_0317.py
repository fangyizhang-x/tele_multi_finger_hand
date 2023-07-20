# code from:
# https: https://google.github.io/mediapipe/solutions/hands.html

# use ros topic to send positions of the hand

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import numpy as np
import rospy
from seed_robotics.msg import *
import math
import matplotlib.pyplot as plt
import time

def distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def calculate_line_radius(points):
    # Convert the list of points to a NumPy array
    points = np.array(points)

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the distance from eachmediapipe
    # Calculate the radius as the average distance from the centroid
    radius = np.mean(distance)

    return radius


def calculate_line_curvature(points):
    # Convert the list of points to a NumPy array
    points = np.array(points)

    # Calculate the tangent vectors at each point
    tangent_vectors = np.gradient(points, axis=0)

    # Calculate the curvature at each point
    norm_tangent_vectors = np.linalg.norm(tangent_vectors, axis=1)
    curvature = np.gradient(tangent_vectors / norm_tangent_vectors[:, np.newaxis], axis=0)
    curvature = np.linalg.norm(curvature, axis=1) / norm_tangent_vectors ** 3

    # Calculate the average curvature
    avg_curvature = np.mean(curvature)

    return int(avg_curvature)


def landmark2list(landmark):
    m = len(landmark)
    list_of_landmarks = []
    for i in range(m):
        curr_pos = [landmark[i].x, landmark[i].y, landmark[i].z]
        # curr_pos = [landmark[i].x, landmark[i].y]
        list_of_landmarks.append(curr_pos)
        
    return list_of_landmarks

def get2PointsDistance():
    print(11)

def cameraToRos(value, max, min, coefficient=3000):
    result = 0
    if value >= max:
        value = max
    elif value <= min:
        value = min
    
    result = coefficient-coefficient*(value - min)/(max - min)
    return int(result)
        

rospy.init_node('mediapipe_hand_publisher',anonymous=True)
pub_mediapipe = rospy.Publisher('mediapipe_hand_topic', MediaPipe, queue_size=10)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  
  while cap.isOpened() and not rospy.is_shutdown():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # print('>>> multi_hand_landmarks:', results.multi_hand_landmarks)
    # print('>>> multi_hand_world_landmarks:', results.multi_hand_world_landmarks)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
    # if results.multi_hand_world_landmarks:

        # draw the 21 points
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
        # calculate a list of 21 hand landmarks in world coordinates
        for hand_landmarks in results.multi_hand_world_landmarks:
            list_of_landmarks = landmark2list(hand_landmarks.landmark)
            time.sleep(0.001)

            # method 1: 2 points distance
            p1_2 = distance(list_of_landmarks[1], list_of_landmarks[2])
            p2_3 = distance(list_of_landmarks[2], list_of_landmarks[3])
            p3_4 = distance(list_of_landmarks[3], list_of_landmarks[4])
            p0_4 = distance(list_of_landmarks[0], list_of_landmarks[4])

            p3_5 = distance(list_of_landmarks[3], list_of_landmarks[5])
            p4_5 = distance(list_of_landmarks[4], list_of_landmarks[5])
            p4_17 = distance(list_of_landmarks[4], list_of_landmarks[17])

            p0_8 = distance(list_of_landmarks[0], list_of_landmarks[8])
            p0_12 = distance(list_of_landmarks[0], list_of_landmarks[12])
            p0_16 = distance(list_of_landmarks[0], list_of_landmarks[16])
            p0_20 = distance(list_of_landmarks[0], list_of_landmarks[20])



            thumb_adduction = distance(list_of_landmarks[17], list_of_landmarks[4])
            thumb_flexion = distance(list_of_landmarks[5], list_of_landmarks[4])
            index = distance(list_of_landmarks[5], list_of_landmarks[8])
            middle = distance(list_of_landmarks[9], list_of_landmarks[12])
            ring = distance(list_of_landmarks[13], list_of_landmarks[16])
            little = distance(list_of_landmarks[17], list_of_landmarks[20])

            # index = distance(list_of_landmarks[0], list_of_landmarks[8])
            # middle = distance(list_of_landmarks[0], list_of_landmarks[12])
            # ring = distance(list_of_landmarks[0], list_of_landmarks[16])
            # little = distance(list_of_landmarks[0], list_of_landmarks[20])

            # 2 points distance
            # thumb1 = distance(list_of_landmarks[1], list_of_landmarks[4])
            # index1 = distance(list_of_landmarks[5], list_of_landmarks[8])
            # middle1 = distance(list_of_landmarks[9], list_of_landmarks[12])
            # ring1 = distance(list_of_landmarks[13], list_of_landmarks[16])
            # little1 = distance(list_of_landmarks[17], list_of_landmarks[20])

            # method 2: curvature of a line
            thumb_curvature = calculate_line_curvature( list_of_landmarks[1:5])
            index_curvature = calculate_line_curvature( list_of_landmarks[5:9])
            middle_curvature = calculate_line_curvature( list_of_landmarks[9:13])
            ring_curvature = calculate_line_curvature( list_of_landmarks[13:17])
            little_curvature = calculate_line_curvature( list_of_landmarks[17:21])
            
            print('-----distance of 2 points--------------------------------------------------------------------')
            print('1. adduction: {:.3f}; flexion: {:.3f}; index: {:.3f}; middle: {:.3f}; ring: {:.3f}; little: {:.3f}'.format(thumb_adduction, thumb_flexion, index, middle, ring, little))
            print('**********************************************************************************************')
            print('1. 1-2: {:.3f}; 2-3: {:.3f}; 3-4: {:.3f}; 0-4: {:.3f}'.format(p1_2, p2_3, p3_4, p0_4))
            print('**********************************************************************************************')
            print('1. 3-5: {:.3f}; 4-5: {:.3f}; 4-17: {:.3f}; ---: {:.3f}'.format(p3_5, p4_5, p4_17, p4_5, p4_5))
            print('**********************************************************************************************')
            print('1. 8-0: {:.3f}; 12-0: {:.3f}; 16-0: {:.3f}; 20-0: {:.3f}'.format(p0_8, p0_12, p0_16, p0_20))
            print('-----curvature of a line----------------------------------------------------------------------')
            print('2. thumb: {}; index: {}; middle: {}; ring: {}; little: {}'.format(thumb_curvature, index_curvature, middle_curvature, ring_curvature, little_curvature))

            # print('2. thumb: {:.3f}; index: {:.3f}; middle: {:.3f}; ring: {:.3f}; little: {:.3f}'.format(thumb1, index1, middle1, ring1, little1))
            
            # calculate the ROS topic data
            ros_thumb_adduction = cameraToRos(p4_17, 0.11, 0.09, 4000)
            ros_thumb_flexion = cameraToRos(p3_5, 0.056, 0.04, 4000)
            if ros_thumb_flexion >999:
                ros_thumb_adduction = 0
            if ros_thumb_adduction >500:
                ros_thumb_flexion = 0
            ros_index = cameraToRos(p0_8, 0.156, 0.057)
            ros_middle = cameraToRos(p0_12, 0.158, 0.05)
            ros_ring = cameraToRos(p0_16, 0.15, 0.05)
            ros_little = cameraToRos(p0_20, 0.125, 0.06)
            print('---------------------------------------------------------------------------------------------')
            print('ROS >>>')
            print('adduction: {}; flexion: {}; index: {}; middle: {}; ring: {}; little: {}'.format(ros_thumb_adduction, ros_thumb_flexion, ros_index, ros_middle, ros_ring, ros_little))

             # rostopic--publish the message
            hand_landmarks_msgs = MediaPipe()
            hand_landmarks_msgs.thumb_adduction = ros_thumb_adduction
            hand_landmarks_msgs.thumb_flexion = ros_thumb_flexion
            hand_landmarks_msgs.index = ros_index
            hand_landmarks_msgs.middle = ros_middle
            hand_landmarks_msgs.ring = ros_ring
            hand_landmarks_msgs.little = ros_little
            
            # publish messages
            pub_mediapipe.publish(hand_landmarks_msgs)
            
           
        # draw bar chart via pyplot 
        # x_lable = ['adduction', 'flexion', 'index', 'middle', 'ring', 'little']
        # x_lable = ['adduction', 'flexion', 'index', 'middle', 'ring', 'little']
        # # Clear the graph
        # plt.cla()
        # # Iteration: open
        # plt.ion()
        # # Create a bar chart
        # plt.bar(x_lable, [thumb_adduction, thumb_flexion, index, middle, ring, little], width=0.2)
        # plt.title('Finger position change bar chart')
        # # Show the plot
        # plt.show()
        # plt.pause(0.001)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()