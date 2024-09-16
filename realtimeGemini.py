import google.generativeai as genai
from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Back, Style
import colorama
import os
import time

# Gemini API configuration
API_KEY = "AIzaSyBJ2cMXQFnuyR5wbj5STTBWF124i91mxeI"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

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
    global processed_text
    if text not in processed_text:
        full_sentences.append(text)
        processed_text.add(text)
        text_detected("")
        
        try:
            response = model.generate_content(text)
            print("\nGemini Response:")
            print(Fore.GREEN + response.text + Style.RESET_ALL)
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == '__main__':
    print("Initializing Realtime STT with Gemini integration...")
    colorama.init()

    full_sentences = []
    displayed_text = ""
    processed_text = set()
    last_input_time = time.time()

    recorder_config = {
        'spinner': False,
        'model': 'base',
        'silero_sensitivity': 0.5,
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
    print("Say something...", end="", flush=True)

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