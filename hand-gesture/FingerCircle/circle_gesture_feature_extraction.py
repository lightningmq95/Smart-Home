import cv2
import mediapipe as mp
import numpy as np
import os
import h5py

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

Data_loc = '../Data/Data_vids/Circle_gesture_data'
output_file = '../Data/finger_circle_features.h5'
number_of_gestures = 2
video_number = 100
fps = 60

features = []
prev_angle = None
float_value = 0.0

def detect_finger_circle_gesture(landmarks, gesture_id):
    global prev_angle, float_value

    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    wrist = landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y

    current_angle = np.arctan2(index_tip[1] - wrist[1], index_tip[0] - wrist[0])

    angle_diff = 0
    if prev_angle is not None:
        angle_diff = np.degrees(current_angle - prev_angle)
        if angle_diff > 10:
            float_value += 0.1
        elif angle_diff < -10:
            float_value -= 0.1

    prev_angle = current_angle

    return [gesture_id, float_value, angle_diff]

def process_videos():
    global prev_angle, float_value

    prev_angle = None
    float_value = 0.0

    total_videos = number_of_gestures * video_number
    processed_videos = 0

    for gesture in range(number_of_gestures):
        gesture_dir = os.path.join(Data_loc, str(gesture))

        for video_num in range(video_number):
            video_path = os.path.join(gesture_dir, f'{gesture}_clip_{video_num + 1}.mp4')
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = hand_landmarks.landmark
                        feature_row = detect_finger_circle_gesture(landmarks, gesture)
                        features.append(feature_row)

            cap.release()
            processed_videos += 1
            percentage_complete = (processed_videos / total_videos) * 100
            print(f'Progress: {percentage_complete:.2f}% ({processed_videos}/{total_videos} videos processed)')

    features_np = np.array(features, dtype=float)

    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('features', data=features_np, dtype='f')
        hf.create_dataset('columns', data=np.array(['Gesture', 'Float_Value', 'Angle_Diff'], dtype='S'))

process_videos()
