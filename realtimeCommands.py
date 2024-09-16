from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Back, Style
import colorama
import os
import fuzzy
import Levenshtein

soundex = fuzzy.Soundex(4)

commands = ["lights on", "lights off", "fan faster", "fan slower", "fan on", "fan off"]

def find_closest_command(recognized_text):
    recognized_soundex = soundex(recognized_text)
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

if __name__ == '__main__':

    print("Initializing RealtimeSTT test...")

    colorama.init()

    full_sentences = []
    displayed_text = ""

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
            # clear_console()
            print(f"Language: {recorder.detected_language} (realtime: {recorder.detected_realtime_language})")
            print(displayed_text, end="", flush=True)

    def process_text(text):
        if not text:
            print("No text detected.")
            return

        closest_command, distance = find_closest_command(text)
        threshold = 8  # Define a more appropriate threshold for command matching
        min_length = 2
        max_length = 50  # Maximum length of recognized text to consider for command matching
        print(f"Recognized text: '{text}', Distance: {distance}")

        if distance <= threshold and min_length <= len(text) <= max_length:
            corrected_text = closest_command
        else:
            corrected_text = text

        full_sentences.append(corrected_text)
        text_detected("")

    recorder_config = {
        'spinner': False,
        'model': 'tiny',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 2,
        'post_speech_silence_duration': 0.4,
        'use_main_model_for_realtime': True,
        'min_length_of_recording': 3,
        'min_gap_between_recordings': 1,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.2,
        'realtime_model_type': 'tiny',
        'on_realtime_transcription_update': text_detected, 
        'silero_deactivity_detection': True,
        'realtime_model_type': 'large-v2'
    }

    recorder = AudioToTextRecorder(**recorder_config)

    clear_console()
    print("Say something...", end="", flush=True)

    while True:
        try:
            recorder.text(process_text)
        except Exception as e:
            print(f"Error during transcription: {e}")