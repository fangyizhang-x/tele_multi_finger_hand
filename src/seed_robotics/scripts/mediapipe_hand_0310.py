# code from:
# https: https://google.github.io/mediapipe/solutions/hands.html


import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import numpy as np
import rospy
from seed_robotics.msg import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


rospy.init_node('mediapipe_hand_publisher',anonymous=True)
pub_mediapipe = rospy.Publisher('mediapipe_hand_topic', MediaPipe, queue_size=10)


def draw3d():
    # Sample data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    z = [3, 6, 9, 12, 15]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add points to the plot
    ax.scatter(x, y, z, c='r', marker='o')

    # Set labels and title
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Scatter Plot')

    # Show the plot
    plt.show()



def calculate_line_radius(points):
    # Convert the list of points to a NumPy array
    points = np.array(points)

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the distance from each point to the centroid
    distances = np.linalg.norm(points - centroid, axis=1)

    # Calculate the radius as the average distance from the centroid
    radius = np.mean(distances)

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

    return avg_curvature


def landmark2list(landmark):
    m = len(landmark)
    list_of_landmarks = []
    for i in range(m):
        curr_pos = [landmark[i].x, landmark[i].y, landmark[i].z]
        # curr_pos = [landmark[i].x, landmark[i].y]
        list_of_landmarks.append(curr_pos)
        
    return list_of_landmarks


# while not rospy.is_shutdown():
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
        # print(results.multi_hand_landmarks[0].x)
        # thum_radius = calculate_line_radius( results.multi_hand_landmarks[2:5])
        # print(">>>>>>>>>  Thumb Radius: ", thum_radius)
        for hand_landmarks in results.multi_hand_landmarks:
        # for hand_landmarks in results.multi_hand_world_landmarks:
            # print(hand_landmarks.landmark[0].x)
            # print(hand_landmarks.landmark[0].tolist())
            list_of_landmarks = landmark2list(hand_landmarks.landmark)
            # thum_radius = calculate_line_radius( list_of_landmarks[:5])
            ls = []
            ls.append(list_of_landmarks[17])
            ls.append(list_of_landmarks[0])
            ls.append(list_of_landmarks[1])
            ls.append(list_of_landmarks[2])
            thumb_adduction = calculate_line_curvature(ls)
            # thumb_adduction = calculate_line_curvature(list_of_landmarks[0:5])
            thumb_curvature = calculate_line_curvature( list_of_landmarks[1:5])
            index_curvature = calculate_line_curvature( list_of_landmarks[5:9])
            middle_curvature = calculate_line_curvature( list_of_landmarks[9:13])
            ring_curvature = calculate_line_curvature( list_of_landmarks[13:17])
            little_curvature = calculate_line_curvature( list_of_landmarks[17:21])
           
           # print(len(hand_landmarks.landmark))
            # print(len(hand_landmarks))
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
            # rostopic--publish the message
            hand_landmarks_msgs = MediaPipe()
            hand_landmarks_msgs.thumb_curvature = int(thumb_adduction)
            hand_landmarks_msgs.index_curvature = int(index_curvature)
            hand_landmarks_msgs.middle_curvature = int(middle_curvature)
            hand_landmarks_msgs.ring_ltl_curvature = int((ring_curvature+little_curvature)/2)
            
            # publish messages
            pub_mediapipe.publish(hand_landmarks_msgs)
            
            # print(">>>>>>>>>  Thumb Radius: ", thum_radius)
            print(">>> thu_adduction:{}, thumb_flex:{}, index:{}, middle:{}, rl:{}".format(int(thumb_adduction), int(thumb_curvature), int(index_curvature), int(middle_curvature), int((ring_curvature+little_curvature)/2)))
    draw3d()
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()