from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Back, Style
import colorama
import os
from difflib import get_close_matches
from fuzzy import Soundex


if __name__ == '__main__':

    print("Initializing RealtimeSTT test...")

    colorama.init()

    full_sentences = []
    displayed_text = ""

    valid_commands = ["lights on", "lights off", "fans on", "fans off"]
    soundex = Soundex(4)


    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    def is_ascii(s):
        return all(ord(c) < 128 for c in s)

    def find_closest_command(text):
            
        matches = get_close_matches(text, valid_commands, n=1, cutoff=0.6)
        return matches[0] if matches else text
    
    def find_closest_command(text):
        if not is_ascii(text):
            return text
        text_soundex = soundex(text)
        matches = [(command, soundex(command)) for command in valid_commands]
        closest_match = min(matches, key=lambda x: abs(len(text_soundex) - len(x[1])))
        return closest_match[0] if closest_match else text


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
        corrected_text = find_closest_command(text)
        full_sentences.append(corrected_text)
        text_detected("")

    recorder_config = {
        'spinner': False,
        'model': 'tiny',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 2,
        'post_speech_silence_duration': 0.4,
        'min_length_of_recording': 0,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.2,
        'realtime_model_type': 'tiny',
        'on_realtime_transcription_update': text_detected, 
        'silero_deactivity_detection': True,
    }

    recorder = AudioToTextRecorder(**recorder_config)

    clear_console()
    print("Say something...", end="", flush=True)

    while True:
        recorder.text(process_text)
