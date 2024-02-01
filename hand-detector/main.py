import cv2
import mediapipe as mp

# Function to ask for camera access
def request_camera_access():
    cap = cv2.VideoCapture(0)
    success, _ = cap.read()
    cap.release()
    return success

# Provide the path to your video file, or set to None for camera access
video_path = None  # Set the path to video or to None for camera access

# Check if a video file path is provided
if video_path is None or not cv2.VideoCapture(video_path).isOpened():
    # If no video file path is provided or the file cannot be opened, try camera access
    if not request_camera_access():
        print("Camera access denied. Exiting.")
        exit()

# Initialize video capture based on the user's choice
if video_path is not None:
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(0)

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize Mediapipe Hand Tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding box around the moving object (hand)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Hand detection using Mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                # Hand is detected, display text
                cv2.putText(frame, 'Hand Moving', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Detector', frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Adjust the waitKey time according to the video frame rate
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
