import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

joint1 = -1
joint2 = -1
joint3 = -1
joints = [5,6,7]


def calculate_angle_between_lines(A, B, C):
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


cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()
varrr = 0
while True:
    data, image = cap.read()
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            #print('hand_landmarks are :', hand_landmarks)
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            if varrr == 0:
                # Iterate through the landmarks and print their coordinates
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape  # get the shape of the image
                    cx, cy = int(landmark.x * w), int(landmark.y * h)  # convert to pixel coordinates
                    print(f'Landmark {idx}: x={cx}, y={cy}, z={landmark.z}')

                    # Example for the index finger
                    index_finger_joint_1 = hand_landmarks.landmark[joints[0]]
                    index_finger_joint_2 = hand_landmarks.landmark[joints[1]]
                    index_finger_joint_3 = hand_landmarks.landmark[joints[2]]

                    
                    
                        # Optionally, you can draw circles on the landmarks
                    cv2.circle(image, (cx, cy), 3, (255, 255, 0), cv2.FILLED)
                    #varrr = 1
                # Convert to pixel coordinates
                joint1 = [index_finger_joint_1.x , index_finger_joint_1.y , index_finger_joint_1.z]
                joint2 = [index_finger_joint_2.x , index_finger_joint_2.y , index_finger_joint_2.z]
                joint3 = [index_finger_joint_3.x , index_finger_joint_3.y , index_finger_joint_3.z]
                # Calculate the angle
                angle = calculate_angle_between_lines(joint1, joint2, joint3)
                print(f'Angle at Index Finger Joint{joints[0]},{joints[1]},{joints[2]}: {angle:.2f} degrees')
    cv2.imshow('HandTracker', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

