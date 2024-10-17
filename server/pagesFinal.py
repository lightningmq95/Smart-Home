import streamlit as st
import os
import requests
from PIL import Image
import io
import threading
import queue
import time
import pyttsx3
import fuzzy
import Levenshtein
import colorama
from colorama import Fore, Style
from RealtimeSTT import AudioToTextRecorder
import google.generativeai as genai
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel

response_text = ''

def go_next(view):
    if view == "front":
        st.session_state.front_view = st.session_state["front_view_input"]
        next_step = 2
    elif view == "left":
        st.session_state.left_view = st.session_state["left_view_input"]
        next_step = 3
    elif view == "right":
        st.session_state.right_view = st.session_state["right_view_input"]
        next_step = 4
        st.session_state.images_ready = True

    if st.session_state.recapture:
        st.session_state.images_ready = True
        st.session_state.capturing = False
        st.session_state.recapture = False
    else:
        st.session_state.step = next_step

def change_view(view):
    if view == "front":
        st.session_state.step = 1
    elif view == "left":
        st.session_state.step = 2
    elif view == "right":
        st.session_state.step = 3
    st.session_state.capturing = True
    st.session_state.images_ready = False
    st.session_state.recapture = True

def submit_images():
    files = [
        ('images', ('front_view.jpg', st.session_state.front_view.getvalue(), 'image/jpeg')),
        ('images', ('left_view.jpg', st.session_state.left_view.getvalue(), 'image/jpeg')),
        ('images', ('right_view.jpg', st.session_state.right_view.getvalue(), 'image/jpeg'))
    ]
    data = {
        'name': st.session_state.person_name,
        'role': st.session_state.role  
    }
    response = requests.post('http://127.0.0.1:8000/register_user', files=files, data=data)
    if response.status_code == 200:
        st.success("Images submitted successfully!")
    else:
        st.error("Failed to submit images.")
    st.session_state.capturing = False
    st.session_state.images_ready = False

def start_capturing(person_name, role):
    if person_name:
        st.session_state.person_name = person_name
        st.session_state.role = role
        st.session_state.capturing = True
        st.session_state.images_ready = False
        st.session_state.step = 1
    else:
        st.warning("Please enter the person's name")

def change_password(old_password, new_password):
    response = requests.post('http://127.0.0.1:8000/change_password', data={'username': st.session_state.username, 'old_password': old_password, 'new_password': new_password})
    if response.status_code == 200:
        st.success("Password changed successfully!")
    else:
        st.error("Failed to change password. Please check your old password.")

@st.dialog("Change Password")
def show_change_password_modal():
    old_password = st.text_input("Old Password", type="password")
    new_password = st.text_input("New Password", type="password")
    st.button("Submit", on_click=change_password, args=(old_password, new_password))

role_list = ['owner', 'member']

def login(on_click):
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    st.button("Login", on_click=on_click, args=(username, password))

def add_user(on_click):
    if not st.session_state.logged_in:
        login(on_click)
    else:
        st.button("Change Password", on_click=show_change_password_modal)
        st.subheader("Enter Person's Name")
        person_name = st.text_input("Person's Name", value=st.session_state.person_name)
        role = st.selectbox("Select Role", role_list, index=role_list.index(st.session_state.role))
        st.button("Start Capturing", on_click=start_capturing, args=(person_name, role))

        if st.session_state.capturing:
            st.write("Capturing in progress...")

        if st.session_state.images_ready:
            st.button("Submit Images", on_click=submit_images)

# URLs for controlling devices
urlFacematch = "http://127.0.0.1:8000/face_match"
urlLedOn = "http://10.20.19.173:8002/LED=1"
urlLedOff = "http://10.20.19.173:8002/LED=0"
urlMotorOn = "http://10.20.19.173:8002/MOTOR=1"
urlMotorOff = "http://10.20.19.173:8002/MOTOR=0"
urlMotorFast = "http://10.20.19.173:8002/MOTOR=FASTER"
urlMotorSlow = "http://10.20.19.173:8002/MOTORLOWER"

# Gemini API
API_KEY = "AIzaSyBJ2cMXQFnuyR5wbj5STTBWF124i91mxeI"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

soundex = fuzzy.Soundex(4)

commands = ["Jarvis", "jarvis lights on", "jarvis lights off", "jarvis fan faster", "jarvis fan slower", "jarvis fan on", "jarvis fan off", "jarvis gestures on", "jarvis gestures off"]

command_urls = {
    "jarvis lights on": urlLedOn,
    "jarvis lights off": urlLedOff,
    "jarvis fan faster": urlMotorFast,
    "jarvis fan slower": urlMotorSlow,
    "jarvis fan on": urlMotorOn,
    "jarvis fan off": urlMotorOff
}

gesture_thread = None
gesture_thread_running = False
gesture_queue = queue.Queue()

stop_gesture_event = threading.Event()

gesture_detection_active = False
gesture_detection_thread = None

end_of_sentence_detection_pause = 0.45
unknown_sentence_detection_pause = 0.7
mid_sentence_detection_pause = 2.0

prev_text = ""

console = Console()
live = Live(console=console, refresh_per_second=10, screen=False)
live.start()

def find_closest_command(recognized_text):
    recognized_soundex = soundex(recognized_text.lower())
    potential_commands = [command for command in commands if soundex(command) == recognized_soundex]

    if not potential_commands:
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

def preprocess_text(text):
    # Remove leading whitespaces
    text = text.lstrip()

    #  Remove starting ellipses if present
    if text.startswith("..."):
        text = text[3:]

    # Remove any leading whitespaces again after ellipses removal
    text = text.lstrip()

    # Uppercase the first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text

def keyword_detected(text):
    return "jarvis" in text.lower()

# Global variable to track if the console should be cleared on next input
should_clear_console = False

def text_detected(text):
    global prev_text, displayed_text, rich_text_stored, should_clear_console

    text = preprocess_text(text)

    sentence_end_marks = ['.', '!', '?', '。'] 
    if text.endswith("..."):
        recorder.post_speech_silence_duration = mid_sentence_detection_pause
    elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
        recorder.post_speech_silence_duration = end_of_sentence_detection_pause
    else:
        recorder.post_speech_silence_duration = unknown_sentence_detection_pause

    prev_text = text

    if should_clear_console:
        clear_console()
        should_clear_console = False
        full_sentences.clear()
        displayed_text = ""

    rich_text = Text()
    for i, sentence in enumerate(full_sentences):
        if i % 2 == 0:
            rich_text += Text(sentence, style="yellow") + Text(" ")
        else:
            rich_text += Text(sentence, style="cyan") + Text(" ")
    
    if text:
        rich_text += Text(text, style="bold yellow")

    new_displayed_text = rich_text.plain

    if new_displayed_text != displayed_text:
        displayed_text = new_displayed_text
        panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
        live.update(panel)
        rich_text_stored = rich_text

def process_text(text):
    global processed_text, gesture_detection_active, response_text, should_clear_console
    if text not in processed_text:
        full_sentences.append(text)
        processed_text.add(text)
        text_detected("")

        closest_command, distance = find_closest_command(text)
        threshold = 10  
        min_length = 9
        max_length = 50 

        if distance <= threshold and min_length <= len(text) <= max_length:
            if closest_command == "jarvis gestures on":
                start_gesture_detection()
            elif closest_command == "jarvis gestures off":
                stop_gesture_detection()
            st.write(closest_command)
            print(closest_command)
            response_text = closest_command
            st.rerun()
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
        
        if keyword_detected(text):
            try:
                prompt = f"{text} Please provide a concise response in 40-50 words."
                response = model.generate_content(prompt)
                
                print("\nGemini Response:")
                print(Fore.GREEN + response.text + Style.RESET_ALL)
                st.write(response.text)
                
                st.session_state.gemini_response = response.text
                response_text = response.text
                st.rerun()
                
                # Set the flag to indicate the console should be cleared on next input
                should_clear_console = True
                
                engine = pyttsx3.init()
                engine.say(response.text)
                engine.runAndWait()

                
            except Exception as e:
                print(f"\nError: {e}")

def start_gesture_detection():
    global gesture_detection_active, gesture_detection_thread, stop_gesture_event
    if not gesture_detection_active:
        gesture_detection_active = True
        stop_gesture_event.clear()
        gesture_detection_thread = threading.Thread(target=run_gesture_detection)
        gesture_detection_thread.start()
        st.success("Gesture detection started.")

def stop_gesture_detection():
    global gesture_detection_active, gesture_detection_thread
    if gesture_detection_active:
        gesture_detection_active = False
        stop_gesture_event.set()
        if gesture_detection_thread:
            gesture_detection_thread.join()
        st.success("Gesture detection stopped.")

def process_gestures():
    while True:
        try:
            gesture = gesture_queue.get(timeout=0.1)
            process_gesture(gesture)
        except queue.Empty:
            pass

def process_gesture(gesture):
    try:
        if gesture == "open":
            requests.get(urlLedOn)
            requests.get(urlMotorOn)
            print("LED and Motor turned on")
        elif gesture == "close":
            requests.get(urlLedOff)
            requests.get(urlMotorOff)
            print("LED and Motor turned off")
        elif gesture == "clockwise":
            requests.get(urlMotorFast)
            print("Motor speed increased")
        elif gesture == "counterclockwise":
            requests.get(urlMotorSlow)
            print("Motor speed decreased")
    except requests.exceptions.RequestException as e:
        print(f"Error processing gesture '{gesture}': {e}")

def run_gesture_detection():
    import gestureDetection
    gestureDetection.main(gesture_queue, stop_gesture_event)

def STT():
    placeholder = st.empty()
    placeholder.write("Initializing")
    print("Initializing")
    colorama.init()

    global full_sentences, displayed_text, processed_text, last_input_time, recorder

    full_sentences = []
    displayed_text = ""
    processed_text = set()

    recorder_config = {
        'spinner': False,
        'model': 'distil-medium.en',
        'input_device_index': 1,
        'realtime_model_type': 'tiny.en',
        'language': 'en',
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.1,        
        'min_gap_between_recordings': 0,                
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.02,
        'on_realtime_transcription_update': text_detected,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'beam_size': 5,
        'beam_size_realtime': 3,
        'no_log_file': True,
        'initial_prompt': "Use ellipses for incomplete sentences like: I went to the..."        
    }

    recorder = AudioToTextRecorder(**recorder_config)
    
    initial_text = Panel(Text("Say something...", style="cyan bold"), title="[bold yellow]Waiting for Input[/bold yellow]", border_style="bold yellow")
    live.update(initial_text)

    gesture_processing_thread = threading.Thread(target=process_gestures)
    gesture_processing_thread.daemon = True
    gesture_processing_thread.start()

    try:
        while True:
            recorder.text(process_text)
            with placeholder.container():
                st.write(displayed_text)
                global response_text
                if response_text:
                    st.write("\nGemini Response:")
                    st.write(response_text)
    except KeyboardInterrupt:
        live.stop()
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
        exit(0)

def integrated_page():
    st.title("Integrated Page")

    img = st.camera_input("Capture Image")

    if img:
        img_pil = Image.open(img)
        verified = sendImg(img_pil)
        if verified:
            try:
                STT()
            except KeyboardInterrupt:
                st.write("Interrupted by user")
            finally:
                stop_gesture_event.set()
                if gesture_detection_thread:
                    gesture_detection_thread.join()

def sendImg(img):
    if img is None:
        st.error("No image captured.")
        return

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'image': ('image.jpg', img_byte_arr, 'image/jpeg')}
    response = requests.post(urlFacematch, files=files)

    if response.status_code == 200:
        result = response.json()
        if result['name'] == "Unknown":
            st.error("Face not recognized.")
            return False
        else:
            st.success(f"Welcome, {result['name']}!")
            return True
    else:
        st.error("Failed to get a response from the server")
        return False

# Run the integrated page
if __name__ == "__main__":
    integrated_page()