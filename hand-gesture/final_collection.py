import os
import cv2
import time
import numpy as np

def get_gesture_and_type():
    print("Available gestures:")
    print("1. OnOff")
    print("2. Circle")

    gesture_type = int(input("Enter the gesture type (1 for OnOff, 2 for Circle): "))

    if gesture_type == 1:
        print("Available OnOff gestures:")
        gestures = [0, 1]

    elif gesture_type == 2:
        print("Available Circle gestures:")
        gestures = [0, 1]

    else:
        print("Invalid gesture type selected. Exiting.")
        exit()

    selected_gesture = int(input("Enter the gesture number you want to record: "))
    if selected_gesture not in gestures:
        print("Invalid gesture selected. Exiting.")
        exit()

    return gesture_type, selected_gesture

def setup_directories(gesture_type, selected_gesture):
    if gesture_type == 1:
        Data_loc = './Data/Data_vids/OnOff'
        waiting_time = 2
        video_length = 1
        fps = 60

    elif gesture_type == 2:
        Data_loc = './Data/Data_vids/Circles'
        waiting_time = 3
        video_length = 2
        fps = 60

    frame_count = video_length * fps
    gesture_dir = os.path.join(Data_loc, str(selected_gesture))

    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

    existing_files = [f for f in os.listdir(gesture_dir) if f.endswith('.mp4')]
    existing_numbers = sorted(int(f.split('_')[2].split('.')[0]) for f in existing_files)
    missing_numbers = [n for n in range(1, max(existing_numbers, default=0) + 1) if n not in existing_numbers]

    return Data_loc, gesture_dir, existing_files, missing_numbers, frame_count, fps, waiting_time, existing_numbers

def record_video(gesture_type, selected_gesture, video_num, gesture_dir, frame_count, fps, waiting_time):
    print(f'Prepare to start recording video {video_num} for gesture {selected_gesture}')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture video.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        info_frame = np.zeros((200, 640, 3), dtype=np.uint8)
        cv2.putText(info_frame, f'Gesture: {selected_gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(info_frame, f'Video: {video_num}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(info_frame, 'Press SPACE to start recording', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('Info', info_frame)
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            break

    for i in range(waiting_time, 0, -1):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        info_frame = np.zeros((200, 640, 3), dtype=np.uint8)
        cv2.putText(info_frame, f'Recording starts in {i} seconds', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Info', info_frame)
        cv2.imshow('Frame', frame)
        cv2.waitKey(1000)

    recording_window_name = 'Recording'
    cv2.namedWindow(recording_window_name, cv2.WINDOW_AUTOSIZE)

    print(f'Recording video {video_num} for gesture {selected_gesture}')

    video_writer = cv2.VideoWriter(
        os.path.join(gesture_dir, f'{selected_gesture}_clip_{video_num}.mp4'),
        cv2.VideoWriter_fourcc(*'H264'), fps, (int(cap.get(3)), int(cap.get(4)))
    )

    for frame_idx in range(frame_count):
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        video_writer.write(frame)

        remaining_time = (frame_count - frame_idx) / fps
        info_frame = np.zeros((200, 640, 3), dtype=np.uint8)
        cv2.putText(info_frame, f'Recording... Time left: {remaining_time:.2f}s', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Info', info_frame)
        cv2.imshow('Frame', frame)
        cv2.imshow(recording_window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_writer.release()
    print(f'Video {video_num} for gesture {selected_gesture} completed')

    cap.release()
    cv2.destroyWindow(recording_window_name)

def main():
    gesture_type, selected_gesture = get_gesture_and_type()
    Data_loc, gesture_dir, existing_files, missing_numbers, frame_count, fps, waiting_time, existing_numbers = setup_directories(gesture_type, selected_gesture)

    start_video_num = max(existing_numbers, default=0) + 1

    for video_num in range(start_video_num, 176):
        record_video(gesture_type, selected_gesture, video_num, gesture_dir, frame_count, fps, waiting_time)

    _, _, _, missing_numbers, _, _, _, _ = setup_directories(gesture_type, selected_gesture)

    if missing_numbers:
        print(f"Missing videos detected: {missing_numbers}")
        for missing_num in missing_numbers:
            record_video(gesture_type, selected_gesture, missing_num, gesture_dir, frame_count, fps, waiting_time)
    else:
        print(f"All videos for gesture {selected_gesture} are complete.")

if __name__ == "__main__":
    main()
