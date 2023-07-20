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

def distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# rospy.init_node('mediapipe_hand_publisher',anonymous=True)
# pub_mediapipe = rospy.Publisher('mediapipe_hand_topic', mediapipe, queue_size=10)

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

def angle_line_plane(line_point1, line_point2, plane_point1, plane_point2, plane_point3):
    
    # calculate the normal vector of the plane
    plane_vector1 = [plane_point2[i]-plane_point1[i] for i in range(3)]
    plane_vector2 = [plane_point3[i]-plane_point1[i] for i in range(3)]
    normal_vector = [
        plane_vector1[1]*plane_vector2[2] - plane_vector1[2]*plane_vector2[1],
        plane_vector1[2]*plane_vector2[0] - plane_vector1[0]*plane_vector2[2],
        plane_vector1[0]*plane_vector2[1] - plane_vector1[1]*plane_vector2[0]
    ]

    # calculate the vector of the line
    line_vector = [line_point2[i]-line_point1[i] for i in range(3)]

    # calculate the angle between the line and the plane
    dot_product = sum([line_vector[i]*normal_vector[i] for i in range(3)])
    line_length = math.sqrt(sum([line_vector[i]**2 for i in range(3)]))
    plane_length = math.sqrt(sum([normal_vector[i]**2 for i in range(3)]))
    cosine = dot_product / (line_length * plane_length)
    angle = math.acos(cosine)

    return angle

def get_perpendicular_point(a, b, c):
    """
    Calculate the line perpendicular to the plane defined by points 'a', 'b', and 'c', 
    which passes through the point 'a'.
    
    Args:
    a, b, c: numpy arrays with shape (3,) representing the 3D coordinates of points
    
    Returns:
    l: a tuple of two numpy arrays with shape (3,) representing a point and a direction 
       vector of the line
    """
    # Calculate the normal vector of the plane B
    normal = np.cross(np.array(b) - np.array(a), np.array(c) - np.array(a))
    
    # Calculate the direction vector of the line l
    direction = normal / np.linalg.norm(normal)
    
    return direction.tolist()

def angle_between_lines(a, b, c):
    """
    Calculates the angle between two lines.
    
    Arguments:
    a -- the first point of the first line as a tuple (x, y, z)
    b -- the second point of the first line and the first point of the second line as a tuple (x, y, z)
    c -- the second point of the second line as a tuple (x, y, z)
    
    Returns:
    The angle between the two lines in radians.
    """
    # Calculate the vectors of the two lines
    vec1 = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    vec2 = (c[0]-b[0], c[1]-b[1], c[2]-b[2])
    
    # Calculate the dot product of the vectors
    dot_product = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
    
    # Calculate the magnitudes of the vectors
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2 + vec1[2]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2 + vec2[2]**2)
    
    # Calculate the angle between the two vectors using the dot product and magnitudes
    angle = math.acos(dot_product / (mag1 * mag2))
    
    return angle


def angle_between_planes(a, b, c, d):
    # calculate normal vectors for planes A and B
    v1 = np.cross(np.array(b)-np.array(a), np.array(c)-np.array(a))
    v2 = np.cross(np.array(b)-np.array(a), np.array(d)-np.array(a))
    v1 /= np.linalg.norm(v1)  # normalize v1
    v2 /= np.linalg.norm(v2)  # normalize v2

    # calculate the cosine of the angle between the normal vectors
    cos_angle = np.dot(v1, v2)

    # convert cosine to angle in degrees
    # angle = np.degrees(np.arccos(cos_angle))
    angle = np.arccos(cos_angle)

    return angle

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


def cameraToRosDirectRatio(value, max, min, coefficient=4000):
    result = 0
    if value >= max:
        value = max
    elif value <= min:
        value = min
    
    result = coefficient*(value - min)/(max - min)
    return int(result)

def cameraToRosIverseRatio(value, max, min, coefficient=4000):
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
  cameraToRosDirectRatio
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

            # raw data: from camera
            perpendicular_point = get_perpendicular_point(list_of_landmarks[0], list_of_landmarks[5], list_of_landmarks[17])

            thumb_adduction = angle_between_planes(list_of_landmarks[0], list_of_landmarks[5], perpendicular_point, list_of_landmarks[2])
            thumb_flexion = angle_between_lines(list_of_landmarks[1], list_of_landmarks[2], list_of_landmarks[3])
            index = angle_line_plane(list_of_landmarks[6], list_of_landmarks[5], list_of_landmarks[5], list_of_landmarks[17], list_of_landmarks[0])
            middle = angle_line_plane(list_of_landmarks[10], list_of_landmarks[9], list_of_landmarks[5], list_of_landmarks[17], list_of_landmarks[0])
            ring = angle_line_plane(list_of_landmarks[14], list_of_landmarks[13], list_of_landmarks[5], list_of_landmarks[17], list_of_landmarks[0])
            little = angle_line_plane(list_of_landmarks[18], list_of_landmarks[17], list_of_landmarks[5], list_of_landmarks[17], list_of_landmarks[0])
            print('1. adduction: {:.3f}; flexion: {:.3f}; index: {:.3f}; middle: {:.3f}; ring: {:.3f}; little: {:.3f}'.format(thumb_adduction, thumb_flexion, index, middle, ring, little))


             # ROS data: calculate the ROS topic data
            ros_thumb_adduction = cameraToRosIverseRatio( thumb_adduction, 1.0, 0.65)
            ros_thumb_flexion = cameraToRosDirectRatio(thumb_flexion, 0.45, 0.17)
            ros_index = cameraToRosDirectRatio(index, 2.6, 1.56)
            ros_middle = cameraToRosDirectRatio(middle, 2.7, 1.38)
            ros_ring = cameraToRosDirectRatio(ring, 2.8, 1.4)
            ros_little = cameraToRosDirectRatio(little, 2.4, 1.35)
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


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()