# code from:
# https: https://google.github.io/mediapipe/solutions/hands.html

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
# IMAGE_FILES = []
# with mp_hands.Hands(
#     static_image_mode=False, # If set to false, the solution treats the input images as a video stream.
#     max_num_hands=1, # Maximum number of hands to detect. Default to 2.
#     model_complexity=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:
#   for idx, file in enumerate(IMAGE_FILES):
#     # Read an image, flip it around y-axis for correct handedness output (see above).
#     image = cv2.flip(cv2.imread(file), 1)
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Print handedness and draw hand landmarks on the image.
#     print('Handedness:', results.multi_handedness)
#     if not results.multi_hand_landmarks:
#       continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       print('hand_landmarks:', hand_landmarks)
#       print(
#           f'Index finger tip coordinates: (',
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#       )
#       mp_drawing.draw_landmarks(
#           annotated_image,
#           hand_landmarks,
#           mp_hands.HAND_CONNECTIONS,
#           mp_drawing_styles.get_default_hand_landmarks_style(),
#           mp_drawing_styles.get_default_hand_connections_style())
#     cv2.imwrite(
#         '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    
#     # Draw hand world landmarks.
#     if not results.multi_hand_world_landmarks:
#       continue
#     for hand_world_landmarks in results.multi_hand_world_landmarks:
#       mp_drawing.plot_landmarks(
#         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)



def landmark2list(landmark):
    m = len(landmark)
    list_of_landmarks = []
    for i in range(m):
        curr_pos = [landmark[i].x, landmark[i].y, landmark[i].z]
        # curr_pos = [landmark[i].x, landmark[i].y]
        list_of_landmarks.append(curr_pos)
        
    return list_of_landmarks

def f3d(landmark):
    m = len(landmark)
    x = []
    y = []
    z = []
    for i in range(m):
        x.append(landmark[i].x)
        y.append(landmark[i].y)
        z.append(landmark[i].z)      
    return x, y, z

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  
  while cap.isOpened():
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
    """Processes an RGB image and returns the hand landmarks and handedness of each detected hand.

    Args:
      image: An RGB image represented as a numpy ndarray.

    Raises:
      RuntimeError: If the underlying graph throws any error.
      ValueError: If the input image is not three channel RGB.

    Returns:
      A NamedTuple object with the following fields:
        1) a "multi_hand_landmarks" field that contains the hand landmarks on
           each detected hand.
        2) a "multi_hand_world_landmarks" field that contains the hand landmarks
           on each detected hand in real-world 3D coordinates that are in meters
           with the origin at the hand's approximate geometric center.
        3) a "multi_handedness" field that contains the handedness (left v.s.
           right hand) of the detected hand.
    """
    print('-----------------------------------------')
    print(results.multi_hand_landmarks)
    print('results.multi_hand_landmarks:',type(results.multi_hand_landmarks))

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        print('********************************************')
        print(hand_landmarks)
        print('hand_landmarks',type(hand_landmarks))
        print('/////////////////////////////////')
        print(hand_landmarks.landmark[0])
        print('....................................')
        print(hand_landmarks.landmark[20].x)
        print('hand_landmarks.landmark: ',type(hand_landmarks.landmark))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        ls_of_landmarks = f3d(hand_landmarks.landmark)
        print('x=',ls_of_landmarks[0])
        print('y=',ls_of_landmarks[1])
        print('z=',ls_of_landmarks[2])
        print(type(ls_of_landmarks))
        print('8888888888888888888888888888888888')



        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Print handedness
        # print('1. multi_hand_world_landmarks:', results.multi_hand_world_landmarks)
        # print('Handedness:', results.multi_handedness)
        # Print info of 
        # print('2. multi_hand_landmarks:', hand_landmarks)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()