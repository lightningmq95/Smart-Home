import json
import bcrypt
import os

# Path to the credentials file
CREDENTIALS_FILE = 'credentials.json'

# Function to hash a password
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Function to initialize credentials
def initialize_credentials():
    # Default credentials
    credentials = {
        "admin": hash_password("admin")
    }

    # Ensure the file exists and initialize it with default credentials
    with open(CREDENTIALS_FILE, 'w') as f:
        json.dump(credentials, f, indent=4)

    print("Credentials initialized successfully.")

if __name__ == "__main__":
    initialize_credentials()