import cv2
import requests
import numpy as np
from PIL import Image
import io
import google.generativeai as genai
from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Back, Style
import colorama
import os
import time
import fuzzy
import Levenshtein
import pyttsx3
import pyaudio
import webrtcvad

urlFacematch = "http://127.0.0.1:8000/face_match"
urlLedOn = "http://localhost:8180/LED=1"
urlLedOff = "http://localhost:8180/LED=0"
urlMotorOn = "http://localhost:8180/MOTOR=1"
urlMotorOff = "http://localhost:8180/MOTOR=0"
urlMotorFast = "http://localhost:8180/MOTOR=FASTER"
urlMotorSlow = "http://localhost:8180/MOTOR=SLOWER"

# Gemini API configuration
API_KEY = "AIzaSyBJ2cMXQFnuyR5wbj5STTBWF124i91mxeI"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

soundex = fuzzy.Soundex(4)

commands = ["lights on", "lights off", "fan faster", "fan slower", "fan on", "fan off"]

current_motor_speed = 0.0

command_urls = {
    "lights on": urlLedOn,
    "lights off": urlLedOff,
    "fan faster": urlMotorFast,
    "fan slower": urlMotorSlow,
    "fan on": urlMotorOn,
    "fan off": urlMotorOff
}

def find_closest_command(recognized_text):
    recognized_soundex = soundex(recognized_text.lower())
    print(f"Recognized text: '{recognized_text}', Soundex: {recognized_soundex}")
    
    potential_commands = [command for command in commands if soundex(command) == recognized_soundex]
    print(f"Potential commands: {potential_commands}")

    if not potential_commands:
        # Fallback to using Levenshtein distance on all commands
        potential_commands = commands

    closest_command = None
    min_distance = float('inf')

    for command in potential_commands:
        distance = Levenshtein.distance(recognized_text, command)
        if distance < min_distance:
            min_distance = distance
            closest_command = command

    return closest_command, min_distance

def clear_console():
    os.system('clear' if os.name == 'posix' else 'cls')

def text_detected(text):
    global displayed_text
    sentences_with_style = [
        f"{Fore.YELLOW + sentence + Style.RESET_ALL if i % 2 == 0 else Fore.CYAN + sentence + Style.RESET_ALL} "
        for i, sentence in enumerate(full_sentences)
    ]
    new_text = "".join(sentences_with_style).strip() + " " + text if len(sentences_with_style) > 0 else text

    if new_text != displayed_text:
        displayed_text = new_text
        clear_console()
        print(f"Language: {recorder.detected_language} (realtime: {recorder.detected_realtime_language})")
        print(displayed_text, end="", flush=True)

def process_text(text):
    global processed_text, current_motor_speed
    if text not in processed_text:
        full_sentences.append(text)
        processed_text.add(text)
        text_detected("")
        
        closest_command, distance = find_closest_command(text)
        threshold = 12  
        min_length = 2
        max_length = 50  

        print(f"Recognized text: '{text}', Distance: {distance}")

        if distance <= threshold and min_length <= len(text) <= max_length:
            print(f"Detected command: {closest_command}")
            # Handle the command here (e.g., control lights, fan, etc.)
            if closest_command == "fan faster":
                current_motor_speed = min(current_motor_speed + 0.2, 1.0)
                url = f"http://localhost:8180/MOTOR={current_motor_speed}"
            elif closest_command == "fan slower":
                current_motor_speed = max(current_motor_speed - 0.2, 0.0)
                url = f"http://localhost:8180/MOTOR={current_motor_speed}"
            else:
                url = command_urls.get(closest_command)

            if url:
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        print(f"Command '{closest_command}' executed successfully.")
                    else:
                        print(f"Failed to execute command '{closest_command}'.")
                except Exception as e:
                    print(f"Error sending request: {str(e)}")
            return
        
        try:
            prompt = f"{text} Please provide a concise response in 40-50 words."
            response = model.generate_content(prompt)
            print("\nGemini Response:")
            print(Fore.GREEN + response.text + Style.RESET_ALL)
            
            # Text-to-Speech for Gemini response
            engine = pyttsx3.init()
            engine.say(response.text)
            engine.runAndWait()
            
        except Exception as e:
            print(f"\nError: {str(e)}")

def STT():
    print("Initializing")
    colorama.init()

    global full_sentences, displayed_text, processed_text, last_input_time, recorder

    full_sentences = []
    displayed_text = ""
    processed_text = set()
    last_input_time = time.time()

    recorder_config = {
        'spinner': False,
        'model': 'base',
        'silero_sensitivity': 0.7,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': 0.5,
        'min_length_of_recording': 0,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.1,
        'realtime_model_type': 'base',
        'on_realtime_transcription_update': text_detected,
        'silero_deactivity_detection': True,
    }

    recorder = AudioToTextRecorder(**recorder_config)

    clear_console()
    print("Say something", end="", flush=True)

    while True:
        current_time = time.time()
        
        # Check if there has been no input for more than 2 seconds
        if current_time - last_input_time > 2 and full_sentences:
            # Process the accumulated text
            process_text(" ".join(full_sentences))
            
            # Reset the list of sentences
            full_sentences.clear()
        
        recorder.text(process_text)
        
        # Update the last input time
        if full_sentences:
            last_input_time = current_time

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Press 'c' to capture an image or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Capture the frame
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            break
        elif key == ord('q'):
            img = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return img

def sendImg(img):
    if img is None:
        print("No image captured.")
        return

    # Convert the PIL image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Send the image to the server
    files = {'image': ('image.jpg', img_byte_arr, 'image/jpeg')}
    response = requests.post(urlFacematch, files=files)

    # Handle the server's response
    if response.status_code == 200:
        result = response.json()
        if result['name'] == "Unknown":
            print("Face is unknown. Please register your face.")
        else:
            print(f"Name: {result['name']}, Role: {result['role']}, Distance: {result['distance']}")
            STT()
    else:
        print("Failed to get a response from the server")

if __name__ == "__main__":
    img = capture_image()
    sendImg(img)