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

Data_loc = '../Data/Data_vids/Circles'

pickle_file = '../Data/Circle_features/circle_gesture_features.pkl'
csv_file = '../Data/Circle_features/circle_gesture_features.csv'
h5py_file = '../Data/Circle_features/circle_gesture_features.h5'

number_of_gestures = 2
video_number = 175
max_action_duration = 3.0

def extract_features(landmarks):
    index_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
    wrist = np.array([landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y])

    distance_index_wrist = np.linalg.norm(index_tip - wrist)
    return [index_tip[0], index_tip[1], distance_index_wrist]

def calculate_action_duration(start_time, end_time, max_duration=max_action_duration):
    duration = end_time - start_time
    if duration > max_duration:
        return 1.0
    return duration / max_duration

def process_videos():
    features = []
    labels = []
    total_videos = number_of_gestures * video_number
    processed_videos = 0
    start_time = time.time()

    for gesture in range(number_of_gestures):
        gesture_dir = os.path.join(Data_loc, str(gesture))

        for video_num in range(video_number):
            video_path = os.path.join(gesture_dir, f'{gesture}_clip_{video_num + 1}.mp4')
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error: Unable to open video file {video_path}.")
                continue

            video_features = []
            start_action_time = None
            end_action_time = None
            is_action_started = False
            final_label = gesture

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
                            feature_vector = extract_features(landmarks)
                            video_features.append(feature_vector)

                            if feature_vector[2] > 0.1:  # Example threshold to detect motion
                                if not is_action_started:
                                    start_action_time = time.time()
                                    is_action_started = True
                            else:
                                if is_action_started:
                                    end_action_time = time.time()
                                    is_action_started = False
                                    break

                cv2.imshow('Hand Gesture Detection', frame)
                cv2.waitKey(1)

            cap.release()

            if start_action_time and end_action_time:
                action_duration = calculate_action_duration(start_action_time, end_action_time)
            else:
                action_duration = 0

            if video_features:
                features.append([action_duration])
                labels.append(final_label)

                num_features = len(video_features)
                print(f'Features for video {video_num + 1} of gesture {gesture}: {num_features} frames processed.')
            else:
                print(f'No features detected for video {video_path}. Check if the gesture is performed properly.')

            processed_videos += 1
            percentage_complete = (processed_videos / total_videos) * 100
            print(f'Progress: {percentage_complete:.2f}% ({processed_videos}/{total_videos} videos processed)')

    cv2.destroyAllWindows()

    end_time = time.time()
    execution_time = end_time - start_time

    if features:
        features_np = np.array(features, dtype=float)
        labels_np = np.array(labels, dtype=int)

        with open(pickle_file, 'wb') as f:
            pickle.dump((features_np, labels_np), f)

        df = pd.DataFrame(features_np, columns=['action_duration'])
        df['labels'] = labels_np
        df.to_csv(csv_file, index=False)

        with h5py.File(h5py_file, 'w') as hf:
            hf.create_dataset('features', data=features_np)
            hf.create_dataset('labels', data=labels_np)

        print(f"Features and labels saved to {pickle_file}, {csv_file}, and {h5py_file}")
    else:
        print("No features were collected. Please check the detection and feature collection logic.")

    print(f"Total execution time: {execution_time:.2f} seconds")

process_videos()
