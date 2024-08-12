Hand Tracking with MediaPipe and OpenCV - for one test angle, i.e. 3 handlandmark points : (Currently it is at the testing mode)

- Features
  -  Real-time hand tracking using MediaPipe.
  -  Angle calculation between specified joints of the index finger.
  -  Visualization of hand landmarks and angles on the video feed.
  
- Imports: 
  Import necessary libraries for computer vision (cv2), hand tracking (mediapipe), and numerical operations (numpy).

- calculate_angle_between_lines(A, B, C): Computes the angle between three points using vector mathematics.

- Main Loop:
  - Captures video from the webcam.
  - Processes each frame to detect hand landmarks.
  - Calculates and prints the angle between specific joints of the index finger.
  - Visualizes hand landmarks and angles on the video feed.

