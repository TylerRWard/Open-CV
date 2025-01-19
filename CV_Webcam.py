import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Opens webcame. On other devices its 0 instead of 1
cap = cv2.VideoCapture(1)

with mp_holistic.Holistic(min_detection_confidence = .5,  min_tracking_confidence=.5)as holistic:
# Loops through
    while cap.isOpened():
        ret, frame = cap.read()

        # recolor feed so it can do the detection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # make Detecations
        results = holistic.process(image)

        # recolor so we can see it nice
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmark
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                  #Color
                                  mp_drawing.DrawingSpec(color=(0,225,0), thickness = 2, circle_radius = 1),
                                  mp_drawing.DrawingSpec(color=(0,0,225), thickness = 2, circle_radius = 1)
                                  )
        
        # Right Hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  #Color
                                  mp_drawing.DrawingSpec(color=(0,0,225), thickness = 2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color=(255,0,0), thickness = 2, circle_radius = 2)
                                  )

        # Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  #Color
                                  mp_drawing.DrawingSpec(color=(0,0,225), thickness = 2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color=(0,0,225), thickness = 2, circle_radius = 2)
                                  )
        
        # Pose Connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  #Color
                                  mp_drawing.DrawingSpec(color=(0,0,225), thickness = 2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color=(0,0,225), thickness = 2, circle_radius = 2)
                                  )
        
        # Displays drawing
        cv2.imshow('Raw Webcam Feed', image)

        # Quit
        key = cv2.waitKey(500) & 0xFF
        if key == ord('q'):
            break

# Close webcam and gets rid of dependencies 
cap.release()
cv2.destroyAllWindows()





