import cv2
import mediapipe as mp
import numpy as np
import os
import h5py
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

Data_loc = '../Data/Data_vids/OnOff_gesture_data'
output_file = '../Data/fist_gesture_features.h5'

number_of_gestures = 2
video_number = 100

WAIT_TIME = 2

def detect_fist_state(landmarks):
    thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x, landmarks[mp_hands.HandLandmark.THUMB_TIP].y])
    index_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
    pinky_tip = np.array([landmarks[mp_hands.HandLandmark.PINKY_TIP].x, landmarks[mp_hands.HandLandmark.PINKY_TIP].y])

    wrist = np.array([landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y])

    distance_index_pinky = np.linalg.norm(index_tip - pinky_tip)
    finger_tips = [
        index_tip,
        np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y]),
        np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y]),
        pinky_tip
    ]
    avg_distance = np.mean([np.linalg.norm(thumb_tip - fingertip) for fingertip in finger_tips])
    distance_wrist_fingertips = np.mean([np.linalg.norm(wrist - fingertip) for fingertip in finger_tips])

    is_fist_open = (distance_index_pinky > 0.05) and (avg_distance > 0.1) and (distance_wrist_fingertips > 0.15)
    return int(is_fist_open)

def process_videos():
    features = []
    total_videos = number_of_gestures * video_number
    processed_videos = 0

    for gesture in range(1, number_of_gestures + 1):
        gesture_dir = os.path.join(Data_loc, str(gesture))

        for video_num in range(video_number):
            video_path = os.path.join(gesture_dir, f'{gesture}_clip_{video_num + 1}.mp4')
            cap = cv2.VideoCapture(video_path)

            prev_fist_state = None
            state_change_time = None
            fist_state = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = hand_landmarks.landmark
                        current_fist_state = detect_fist_state(landmarks)

                        if prev_fist_state is not None:
                            if current_fist_state != prev_fist_state:
                                if state_change_time is None:
                                    state_change_time = time.time()
                                elif time.time() - state_change_time >= WAIT_TIME:
                                    fist_state = current_fist_state
                                    state_change_time = None
                        else:
                            if current_fist_state != prev_fist_state:
                                state_change_time = time.time()

                        if fist_state is not None:
                            features.append([gesture, fist_state])
                            fist_state = None

                        prev_fist_state = current_fist_state

            cap.release()
            processed_videos += 1
            percentage_complete = (processed_videos / total_videos) * 100
            print(f'Progress: {percentage_complete:.2f}% ({processed_videos}/{total_videos} videos processed)')

    features_np = np.array(features, dtype=int)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('fist_gesture_features', data=features_np, compression='gzip')

process_videos()
