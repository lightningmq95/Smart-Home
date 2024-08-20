import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import pandas as pd
import h5py
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

Data_loc = '../Data/Data_vids/Circle_gesture_data'
pickle_file = '../Data/finger_circle_features.pkl'
csv_file = '../Data/finger_circle_features.csv'
h5py_file = './finger_circle_features.h5'
number_of_gestures = 2
video_number = 175

def detect_finger_circle_gesture(landmarks, start_time):
    index_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
    wrist = np.array([landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y])

    current_time = time.time()
    time_elapsed = current_time - start_time


    float_value = min(time_elapsed / 10.0, 1.0)

    return float_value

def process_videos():
    features = []
    labels = []
    total_videos = number_of_gestures * video_number
    processed_videos = 0

    max_features_length = 0
    start_time = time.time()

    for gesture in range(number_of_gestures):
        gesture_dir = os.path.join(Data_loc, str(gesture))

        for video_num in range(video_number):
            video_path = os.path.join(gesture_dir, f'{gesture}_clip_{video_num + 1}.mp4')
            print(f"Trying to open video file: {video_path}")
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error: Unable to open video file {video_path}.")
                continue

            video_features = []
            start_time_video = time.time()

            print(f"Processing video {video_path}")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        landmarks = hand_landmarks.landmark

                        if landmarks:
                            current_float_value = detect_finger_circle_gesture(landmarks, start_time_video)
                            video_features.append(current_float_value)
                            print(f'Feature detected: {current_float_value}')

                cv2.imshow('Hand Gesture Detection', frame)
                cv2.waitKey(1)

            cap.release()

            max_features_length = max(max_features_length, len(video_features))

            features.append(video_features)
            labels.append(gesture)

            if video_features:
                num_features = len(video_features)
                print(f'Features for video {video_num + 1} of gesture {gesture}: {video_features}')
                print(f'Video {video_path} processed. Features found: {num_features}')
            else:
                print(f'No features detected for video {video_path}. Check if the feature extraction logic is correct.')

            processed_videos += 1
            percentage_complete = (processed_videos / total_videos) * 100
            print(f'Progress: {percentage_complete:.2f}% ({processed_videos}/{total_videos} videos processed)')

    cv2.destroyAllWindows()


    features_padded = []
    for feature in features:
        if len(feature) < max_features_length:
            feature.extend([0] * (max_features_length - len(feature)))
        elif len(feature) > max_features_length:
            feature = feature[:max_features_length]
        features_padded.append(feature)

    features_np = np.array(features_padded, dtype=float)
    labels_np = np.array(labels, dtype=int)


    with open(pickle_file, 'wb') as f:
        pickle.dump((features_np, labels_np), f)


    df = pd.DataFrame(features_np)
    df['labels'] = labels_np
    df.to_csv(csv_file, index=False, header=False)


    with h5py.File(h5py_file, 'w') as hf:
        hf.create_dataset('features', data=features_np)
        hf.create_dataset('labels', data=labels_np)

    print(f"Features and labels saved to {pickle_file}, {csv_file}, and {h5py_file}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")

process_videos()
