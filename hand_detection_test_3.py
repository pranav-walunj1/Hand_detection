# Calculating angles between all possible joints of a hand as mentioned in 'joints' 2D array

import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe components for hand detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define joint combinations for which angles will be calculated
joints = np.array([
    [0, 1, 2],  # Thumb joints
    [1, 2, 3],
    [2, 3, 4],
    [0, 5, 6],  # Index finger joints
    [5, 6, 7],
    [6, 7, 8],
    [9, 10, 11],  # Middle finger joints
    [10, 11, 12],
    [13, 14, 15],  # Ring finger joints
    [14, 15, 16],
    [0, 17, 18],  # Pinky finger joints
    [17, 18, 19],
    [18, 19, 20]
])


def calculate_angle_between_lines(A, B, C):
    """
    Calculate the angle between three points A, B, and C.
    B is the vertex point where the angle is calculated.

    Parameters:
        A, B, C: Lists or arrays containing the (x, y, z) coordinates of the points.

    Returns:
        Angle in degrees between the lines AB and BC.
    """
    A = np.array(A)  # Convert point A to a numpy array
    B = np.array(B)  # Convert point B to a numpy array
    C = np.array(C)  # Convert point C to a numpy array
    
    # Calculate vectors AB and BC
    AB = A - B  # Vector from A to B
    BC = C - B  # Vector from B to C
    
    # Calculate the dot product of AB and BC
    dot_product = AB @ BC
    
    # Calculate the magnitudes of AB and BC
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_AB * magnitude_BC)
    
    # Clamp cos_theta to the range [-1, 1] to avoid domain errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(cos_theta)
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees


# Initialize video capture from the first camera
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()
varrr = 0  # This variable is not used in the code but is initialized

while True:
    # Read a frame from the video capture
    data, image = cap.read()
    
    # Convert the image to RGB and process with MediaPipe
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Iterate through the landmarks and print their coordinates
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape  # Get the shape of the image
                cx, cy = int(landmark.x * w), int(landmark.y * h)  # Convert to pixel coordinates
                print(f'Landmark {idx}: x={cx}, y={cy}, z={landmark.z}')
                
                # Optionally, draw circles on the landmarks
                cv2.circle(image, (cx, cy), 3, (255, 255, 0), cv2.FILLED)
            
            # Get the number of joint combinations
            rows, cols = joints.shape
            
            # Calculate joint angles for all possible joints
            for i in range(rows):
                # Get points from the joints array
                finger_joint_1 = hand_landmarks.landmark[joints[i][0]]
                finger_joint_2 = hand_landmarks.landmark[joints[i][1]]
                finger_joint_3 = hand_landmarks.landmark[joints[i][2]]
                
                # Convert to pixel coordinates
                joint1 = [finger_joint_1.x, finger_joint_1.y, finger_joint_1.z]
                joint2 = [finger_joint_2.x, finger_joint_2.y, finger_joint_2.z]
                joint3 = [finger_joint_3.x, finger_joint_3.y, finger_joint_3.z]
                
                # Calculate the angle
                angle = calculate_angle_between_lines(joint1, joint2, joint3)
                print(f'Angle at Finger Joints {joints[i][0]},{joints[i][1]},{joints[i][2]} is : {angle:.2f} degrees')
    
    # Display the image with landmarks
    cv2.imshow('HandTracker', image)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
