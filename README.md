Hand Tracking with MediaPipe and OpenCV - (Currently it is at the testing phase)

- Features
  -  Real-time hand tracking using MediaPipe.
  -  Angle calculation between specified joints of the finger.
  -  Visualization of hand landmarks and angles on the video feed.
  
- Imports: 
  Import necessary libraries for computer vision (cv2), hand tracking (mediapipe), and numerical operations (numpy).

- calculate_angle_between_lines(A, B, C): Computes the angle between three points using vector mathematics.

- Main Loop:
  - Captures video from the webcam.
  - Processes each frame to detect hand landmarks.
  - Calculates and prints the angle between specific joints of the index finger.
  - Visualizes hand landmarks and angles on the video feed.

  - Files :
    - hand_detection_test.py :  only detects hands
    - hand_detection_test_1.py : calculates one angle between given 3 joints(A,B,C), angle formed at B, with vectos AB and BC
    - hand_detection_test_2-py : calculates all possible angles based on given joint combinations in 'joints' 2D array