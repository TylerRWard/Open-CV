import cv2
import mediapipe as mp

# Initialize MediaPipe and OpenCV
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.pose

# Create Pose instance with desired confidence values
with mp_holistic.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    # Open video file
    cap = cv2.VideoCapture("Homeplate.mp4")
    
    # Get video properties for saving the output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter object to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI, change to 'MP4V' for .mp4 output
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))
    
    # Loop through frames of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Show the original frame (BGR) to check the input color
        cv2.imshow('Original Frame', frame)

        # Convert the image to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
        image_rgb.flags.writeable = False

        # Make pose detection
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        # Convert back to BGR (since OpenCV works with BGR)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks on the frame (in RGB space)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 225), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 0, 225), thickness=2, circle_radius=2))


        # List of body part names corresponding to each landmark index
        landmark_names = [
            "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder", 
            "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", 
            "Left Index", "Right Index", "Left Thumb", "Right Thumb", "Left Hip", "Right Hip", 
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Left Heel", "Right Heel", 
            "Left Foot Index", "Right Foot Index", "Left Leg", "Right Leg", "Left Arm", "Right Arm", 
            "Left Shoulder to Hip Line", "Right Shoulder to Hip Line"
        ]

        # Loop through the landmarks and print the index and corresponding body part name
        for index, landmark in enumerate(results.pose_landmarks.landmark):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            z = landmark.z  # z is a relative depth value
            visibility = landmark.visibility

            # Print the body part name, index, and its coordinates
            print(f"{landmark_names[index]} (Index {index}): X: {x}, Y: {y}, Z: {z}, Visibility: {visibility}")


        # Write the processed frame to the output video (in BGR format)
        out.write(image_bgr)

        # Optionally, display the frame (for debugging purposes)
        cv2.imshow('Processed Frame', image_bgr)

        # Quit on 'q' key press (for real-time debugging only)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
