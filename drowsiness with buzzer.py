"""
drowsiness_alert_with_buzzer.py

Requirements:
    pip install opencv-python mediapipe pyttsx3 numpy

Usage:
    python drowsiness_alert_with_buzzer.py
Press 'q' to quit.

What it does:
 - Uses MediaPipe Face Mesh to get facial landmarks.
 - Computes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
 - If eyes closed for consecutive frames or MAR exceeds threshold, plays buzzer + voice alert.
"""

import time
import math
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import winsound  # for buzzer sound on Windows

# ------------------------ Parameters ------------------------
EAR_THRESHOLD = 0.23
EAR_CONSEC_FRAMES = 40
MAR_THRESHOLD = 0.6
ALERT_COOLDOWN = 5.0
# ------------------------------------------------------------

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# pyttsx3 setup
tts = pyttsx3.init()
tts.setProperty("rate", 150)

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

# ------------------------ Functions ------------------------
def euclidean(a, b):
    return math.dist(a, b)

def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    pts = [(landmarks[idx].x * image_w, landmarks[idx].y * image_h) for idx in eye_indices]
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, mouth_indices, image_w, image_h):
    up = landmarks[mouth_indices[0]]
    down = landmarks[mouth_indices[1]]
    left = landmarks[mouth_indices[2]]
    right = landmarks[mouth_indices[3]]
    up_xy = (up.x * image_w, up.y * image_h)
    down_xy = (down.x * image_w, down.y * image_h)
    left_xy = (left.x * image_w, left.y * image_h)
    right_xy = (right.x * image_w, right.y * image_h)
    vertical = euclidean(up_xy, down_xy)
    horizontal = euclidean(left_xy, right_xy)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

def play_buzzer():
    try:
        frequency = 2500  # Hz
        duration = 2000   # milliseconds
        winsound.Beep(frequency, duration)
    except Exception as e:
        print("Buzzer error:", e)

def speak_alert(text="Wake up! Please stay alert.please take a break."):
    try:
        play_buzzer()  # buzzer before voice
        tts.say(text)
        tts.runAndWait()
    except Exception as e:
        print("TTS error:", e)

# ------------------------ Main ------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    closed_frames = 0
    last_alert_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        alert_flag = False
        if results.multi_face_landmarks:
            mesh_points = results.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(mesh_points, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(mesh_points, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mesh_points, MOUTH, w, h)

            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames >= EAR_CONSEC_FRAMES:
                alert_flag = True
                reason = f"Eyes closed for {closed_frames} frames"
            elif mar > MAR_THRESHOLD:
                alert_flag = True
                reason = "Yawning detected"
            else:
                reason = None

            if alert_flag:
                now = time.time()
                if now - last_alert_time > ALERT_COOLDOWN:
                    speak_alert("Warning! You appear drowsy. Please take a break.")
                    last_alert_time = now
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
                cv2.putText(frame, "DROWSINESS ALERT!", (int(w * 0.1), int(h * 0.5)),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
                if reason:
                    cv2.putText(frame, reason, (int(w * 0.1), int(h * 0.6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            closed_frames = 0

        cv2.imshow("Drowsiness Monitor (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
