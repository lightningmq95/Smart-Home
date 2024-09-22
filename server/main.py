# from typing import Union, List

# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware

# from pydantic import BaseModel
# from PIL import Image
# import json
# import os
# import numpy as np
# import cv2

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=['*'],
#     allow_credentials=True,
#     allow_methods=['*'],
#     allow_headers=['*'],
# )

# class UserRegistration(BaseModel):
#     name: str
#     role: str
#     images: List[UploadFile]

# @app.post("/register_user")
# async def register_user(user: UserRegistration):
#     user_folder = os.path.join('database', user.name)
#     os.makedirs(user_folder, exist_ok=True)
    
#     for i, image in enumerate(user.images):
#         contents = await image.read()
#         nparr = np.frombuffer(contents, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         # Convert the captured frame to a PIL image
#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         img.save(os.path.join(user_folder, f'{user.name}_{i+1}.jpg'))

#     # Save user info in JSON file
#     user_info = {"name": user.name, "role": user.role}
#     with open('users.json', 'a') as f:
#         f.write(json.dumps(user_info) + '\n')

#     return {"message": "User registered successfully"}

from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import json
import os
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

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

    return {"message": "User registered successfully"}