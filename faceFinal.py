from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import os
import cv2
import time

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

# Function to capture image from webcam
def capImage():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Display the webcam feed in an OpenCV window for 5 seconds
    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            exit()
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture a single frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        exit()

    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Convert the captured frame to a PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return img

def face_match(data_path): 
    # getting embedding matrix of the given img
    img = capImage()
    face, prob = mtcnn(img, return_prob=True)  # returns cropped face and probability
    if face is None or prob <= 0.90:
        return ("No face detected", None, img)

    emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false

    saved_data = torch.load(data_path)  # loading data.pt file
    embedding_list = saved_data[0]  # getting embedding data
    name_list = saved_data[1]  # getting list of names

    dist_list = []  # list of matched distances, minimum distance is used to identify the person

    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    idx_min = dist_list.index(min(dist_list))
    min_dist = min(dist_list)
    
    if min_dist > 0.8:
        return ("Unknown", min_dist, img)
    else:
        return (name_list[idx_min], min_dist, img)

# Function to register a new user
def register_user(name):
    user_folder = os.path.join('database', name)
    os.makedirs(user_folder, exist_ok=True)
    
    instructions = ["SHOW FRONT FACE", "SHOW RIGHT SIDE", "SHOW LEFT SIDE"]
    
    for i in range(3):
        print(f"Capturing photo {i+1}...")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        # Display the webcam feed in an OpenCV window for 5 seconds with instructions
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                exit()
            
            # Add instruction text to the frame
            cv2.putText(frame, instructions[i], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Capture a single frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            exit()

        # Release the webcam and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        # Convert the captured frame to a PIL image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.save(os.path.join(user_folder, f'{name}_{i+1}.jpg'))        
        # time.sleep(5)

# Function to train the model with the new dataset
def train_model():
    dataset = datasets.ImageFolder('database')
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    
    loader = DataLoader(dataset, collate_fn=lambda x: x[0])
    
    embedding_list = []
    name_list = []
    
    for img, idx in loader:
        face, prob = mtcnn(img, return_prob=True)
        if face is not None and prob > 0.90:
            emb = resnet(face.unsqueeze(0)).detach()
            embedding_list.append(emb)
            name_list.append(idx_to_class[idx])
    
    data = [embedding_list, name_list]
    torch.save(data, 'data.pt')

# Match face using webcam image
result = face_match('data.pt')
# # Draw bounding box on the image
# img = result[2]
# boxes, _ = mtcnn.detect(img)
# if boxes is not None:
#     draw = ImageDraw.Draw(img)
#     for box in boxes:
#         draw.rectangle(box.tolist(), outline=(255, 0, 0), width=2)
# img.show()

if result[0] == "Unknown":
    print('Face is unknown with distance:', result[1])
    register = input("Do you want to register? (yes/no): ")
    if register.lower() == 'yes':
        name = input("Enter your name: ")
        register_user(name)
        train_model()
        result = face_match('data.pt')
        if result[0] == "Unknown":
            print('Face is still unknown with distance:', result[1])
        else:
            print('Face matched with:', result[0], 'With distance:', result[1])
elif result[0] == "No face detected":
    print('No face detected in the captured image.')
else:
    print('Face matched with:', result[0], 'With distance:', result[1])

# # Draw bounding box on the image
# img = result[2]
# boxes, _ = mtcnn.detect(img)
# if boxes is not None:
#     draw = ImageDraw.Draw(img)
#     for box in boxes:
#         draw.rectangle(box.tolist(), outline=(255, 0, 0), width=2)

# # Display the image with bounding box
# img.show()