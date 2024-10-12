from typing import List
from facenet_pytorch import MTCNN, InceptionResnetV1
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import json
import os
import numpy as np
import cv2
import torch
import bcrypt

app = FastAPI()
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embedding conversion


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Path to the credentials file
CREDENTIALS_FILE = 'credentials.json'

# Function to load credentials
def load_credentials():
    if not os.path.exists(CREDENTIALS_FILE):
        return {}
    with open(CREDENTIALS_FILE, 'r') as f:
        return json.load(f)
    
# Function to save credentials
def save_credentials(credentials):
    with open(CREDENTIALS_FILE, 'w') as f:
        json.dump(credentials, f, indent=4)

# Function to hash a password
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Function to check login credentials
def check_login(username, password):
    credentials = load_credentials()
    if username in credentials:
        hashed_password = credentials[username]
        return bcrypt.checkpw(password.encode(), hashed_password.encode())
    return False

# Function to change password
def change_password(username, old_password, new_password):
    if check_login(username, old_password):
        credentials = load_credentials()
        credentials[username] = hash_password(new_password)
        save_credentials(credentials)
        return True
    return False

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

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if check_login(username, password):
        return {"message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/change_password")
async def change_password_api(username: str = Form(...), old_password: str = Form(...), new_password: str = Form(...)):
    if change_password(username, old_password, new_password):
        return {"message": "Password changed successfully"}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/register_user")
async def register_user(
    name: str = Form(...),
    role: str = Form(...),
    images: List[UploadFile] = File(...)
):
    print(len(images))
    user_folder = os.path.join('database', name)
    os.makedirs(user_folder, exist_ok=True)
    
    # Get the current number of images in the directory
    existing_images = [f for f in os.listdir(user_folder) if f.endswith('.jpg')]
    start_index = len(existing_images) + 1

    for i, image in enumerate(images):
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert the captured frame to a PIL image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.save(os.path.join(user_folder, f'{name}_{start_index+i}.jpg'))

    # Ensure the file exists and initialize it with an empty list if it doesn't
    if not os.path.exists('users.json'):
        with open('users.json', 'w') as f:
            json.dump([], f)

    # Read the existing data
    with open('users.json', 'r') as f:
        users = json.load(f)

    # Check if the user already exists
    if not any(user['name'] == name for user in users):
        # Append the new user info to the list
        user_info = {"name": name, "role": role}
        users.append(user_info)

        # Write the updated list back to the file
        with open('users.json', 'w') as f:
            json.dump(users, f, indent=4)
    train_model()
    return {"message": "User registered successfully"}

@app.post("/face_match")
async def face_match(image: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Convert the frame to a PIL image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Get the embedding matrix of the given image
        face, prob = mtcnn(img, return_prob=True)
        if face is None or prob <= 0.90:
            return {"name": "No face detected", "distance": None}

        emb = resnet(face.unsqueeze(0)).detach()
        
        # Load saved embeddings and names
        saved_data = torch.load('data.pt')
        embedding_list = saved_data[0]
        name_list = saved_data[1]

        dist_list = []  # list of matched distances, minimum distance is used to identify the person

        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)

        idx_min = dist_list.index(min(dist_list))
        min_dist = min(dist_list)
        name = name_list[idx_min]

        # Retrieve the role from the users.json file
        with open('users.json', 'r') as f:
            users = json.load(f)
        role = next((user['role'] for user in users if user['name'] == name), "Unknown")

        if min_dist > 0.8:
            return {"name": "Unknown", "distance": min_dist}
        else:
            return {"name": name, "role": role, "distance": min_dist}
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))