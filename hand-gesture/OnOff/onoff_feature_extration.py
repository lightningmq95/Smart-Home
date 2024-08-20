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

Data_loc = '../Data/Data_vids/OnOff'

pickle_file = '../Data/OnOff_features/fist_gesture_features.pkl'
csv_file = '../Data/OnOff_features/fist_gesture_features.csv'
h5py_file = '../Data/OnOff_features/fist_gesture_features.h5'

number_of_gestures = 2
video_number = 175

def extract_features(landmarks):
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

    return [distance_index_pinky, avg_distance, distance_wrist_fingertips]

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
            final_fist_state = gesture

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

                cv2.imshow('Hand Gesture Detection', frame)
                cv2.waitKey(1)

            cap.release()

            if video_features:
                features.append(video_features)
                labels.append(final_fist_state)

                num_features = len(video_features)
                print(f'Features for video {video_num + 1} of gesture {gesture}: {num_features} frames processed.')
            else:
                print(f'No features detected for video {video_path}. Check if the fist state changes appropriately.')

            processed_videos += 1
            percentage_complete = (processed_videos / total_videos) * 100
            print(f'Progress: {percentage_complete:.2f}% ({processed_videos}/{total_videos} videos processed)')

    cv2.destroyAllWindows()

    end_time = time.time()
    execution_time = end_time - start_time

    if features:
        with open(pickle_file, 'wb') as f:
            pickle.dump((features, labels), f)

        df = pd.DataFrame({'features': features, 'labels': labels})
        df.to_csv(csv_file, index=False)

        with h5py.File(h5py_file, 'w') as hf:
            hf.create_dataset('features', data=np.array(features, dtype=object), dtype=h5py.special_dtype(vlen=np.dtype('float32')))
            hf.create_dataset('labels', data=np.array(labels, dtype=int))

        print(f"Features and labels saved to {pickle_file}, {csv_file}, and {h5py_file}")
    else:
        print("No features were collected. Please check the detection and feature collection logic.")

    print(f"Total execution time: {execution_time:.2f} seconds")

process_videos()
