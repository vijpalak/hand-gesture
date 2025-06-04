import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import numpy as np
import time
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Setup Audio
def setup_audio():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    vol_range = volume.GetVolumeRange()
    return volume, vol_range[0], vol_range[1]

# Setup Hand Tracking
def setup_hand_tracking():
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    draw = mp.solutions.drawing_utils
    return hands, draw

# Gesture Detection
def detect_gesture(hand_landmarks):
    fingers = []
    tips_ids = [4, 8, 12, 16, 20]

    for tip in tips_ids:
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[tip - 2].y
        fingers.append(tip_y < pip_y)

    # Thumbs Up
    if fingers == [True, False, False, False, False] and hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y:
        return "thumbs_up"

    # Thumbs Down
    if fingers == [True, False, False, False, False] and hand_landmarks.landmark[4].y > hand_landmarks.landmark[3].y:
        return "thumbs_down"

    # Volume Control (Thumb and Index raised only)
    if fingers == [True, True, False, False, False]:
        return "volume_adjust"

    return None

# Volume Adjustment
def get_thumb_index_distance(frame, hand_landmarks):
    h, w, _ = frame.shape
    x1 = int(hand_landmarks.landmark[4].x * w)
    y1 = int(hand_landmarks.landmark[4].y * h)
    x2 = int(hand_landmarks.landmark[8].x * w)
    y2 = int(hand_landmarks.landmark[8].y * h)

    cv2.circle(frame, (x1, y1), 7, (255, 0, 255), -1)
    cv2.circle(frame, (x2, y2), 7, (255, 0, 255), -1)
    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return hypot(x2 - x1, y2 - y1)

def main():
    hands, draw = setup_hand_tracking()
    volume, min_vol, max_vol = setup_audio()
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Hand Gesture Control", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Hand Gesture Control", cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow("Hand Gesture Control", 960, 720)

    last_gesture = None
    last_gesture_time = time.time()
    gesture_delay = 2  # seconds
    youtube_opened = False

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                gesture = detect_gesture(hand_landmarks)
                current_time = time.time()

                if gesture and (current_time - last_gesture_time > gesture_delay):
                    if gesture == "thumbs_down":
                        webbrowser.open("https://www.youtube.com")
                        print("Opening YouTube (Thumbs Down)")
                        last_gesture_time = current_time

                    elif gesture == "thumbs_up":
                        pyautogui.press("playpause")
                        print("Play/Pause Video (Thumbs Up)")
                        last_gesture_time = current_time

                    elif gesture == "volume_adjust":
                        distance = get_thumb_index_distance(frame, hand_landmarks)
                        vol = np.interp(distance, [20, 200], [min_vol, max_vol])
                        volume.SetMasterVolumeLevel(vol, None)
                        print("Adjusting Volume")
                        last_gesture_time = current_time

                elif gesture == "volume_adjust":
                    distance = get_thumb_index_distance(frame, hand_landmarks)
                    vol = np.interp(distance, [20, 200], [min_vol, max_vol])
                    volume.SetMasterVolumeLevel(vol, None)
                    print("Adjusting Volume")

            cv2.imshow("Hand Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
