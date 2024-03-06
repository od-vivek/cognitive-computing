import cv2
import mediapipe as mp

# Function to ask for camera access
def request_camera_access():
    cap = cv2.VideoCapture(0)
    success, _ = cap.read()
    cap.release()
    return success