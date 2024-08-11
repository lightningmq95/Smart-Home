import os
import cv2
import time

Data_loc = './Data/Data_vids'

number_of_gestures = 3
video_number = 5

waiting_time = 5

video_length = 30
fps = 60
frame_count = video_length * fps

if not os.path.exists(Data_loc):
    os.makedirs(Data_loc)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

for gesture in range(number_of_gestures):

    gesture_dir = os.path.join(Data_loc, str(gesture))

    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

    for video_num in range(video_number):

        print(f'Prepare to start recording video {video_num + 1} for gesture {gesture}')

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture video.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

            cv2.putText(frame, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f'Video: {video_num + 1}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, 'Press SPACE to start recording', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)

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

            cv2.putText(frame, f'Recording starts in {i} seconds', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(1000)

        print(f'Recording video {video_num + 1} for gesture {gesture}')

        video_writer = cv2.VideoWriter(os.path.join(gesture_dir, f'{gesture}_clip_{video_num + 1}.mp4'), cv2.VideoWriter_fourcc(*'H264'), fps,(int(cap.get(3)), int(cap.get(4))))

        for _ in range(frame_count):

            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

            video_writer.write(frame)

            cv2.imshow('Recording', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_writer.release()

        print(f'Video {video_num + 1} for gesture {gesture} completed')

cap.release()
cv2.destroyAllWindows()
