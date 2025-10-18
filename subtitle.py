#!/usr/bin/env python3
import os
import sys
import re
import json
import requests
import math
import time
import unicodedata
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Global Directories & Configuration
# -----------------------------------------------------------------------------
OUTPUT_DIR = "/Volumes/Data/Download/Subs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
os.chdir(OUTPUT_DIR)

OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_LLM_MODEL = "gpt-oss:120b-cloud"
DEBUG = False

TOP_LANGUAGES = [
    "Afrikaans", "Albanian", "Arabic", "Armenian", "Basque", "Bengali", "Bosnian", "Burmese",
    "Catalan", "Chinese (Simplified)", "Chinese (Traditional)", "Czech", "Croatian", "Danish",
    "Dutch", "English", "Esperanto", "Estonian", "Farsi", "Filipino", "Finnish", "French", "German",
    "Greek", "Gujarati", "Hausa", "Hebrew", "Hindi", "Hungarian", "Icelandic", "Indonesian",
    "Italian", "Japanese", "Javanese", "Kannada", "Khmer", "Korean", "Latin", "Latvian",
    "Macedonian", "Malay", "Malayalam", "Marathi", "Nepali", "Norwegian", "Polish",
    "Portuguese", "Romanian", "Russian", "Sinhala", "Slovak", "Serbian", "Sundanese",
    "Swahili", "Swedish", "Tamil", "Telugu", "Thai", "Turkish", "Ukrainian", "Urdu",
    "Vietnamese", "Welsh", "Xhosa", "Yoruba", "Zulu", "Persian"
]

LLM_LIST = ["gemma2:27b", "gemma3:27b"]

NON_LATIN_LANGUAGES = {
    "Arabic", "Armenian", "Chinese (Simplified)", "Chinese (Traditional)",
    "Hebrew", "Japanese", "Korean", "Russian", "Thai", "Greek",
    "Hindi", "Bengali", "Tamil", "Telugu", "Gujarati", "Persian"
}

RTL_LANGUAGES = {"Arabic", "Hebrew", "Persian", "Urdu"}

SRT_MODE_DESC = {
    "1": "Only fluent translation",
    "3": "Full original format (index, original text, translation)",
    "4": "Original sentence + translation"
}

# -----------------------------------------------------------------------------
# Text Normalization & Cleaning
# -----------------------------------------------------------------------------
def normalize_and_clean_text(text):
    # Normalize to NFC
    text = unicodedata.normalize('NFC', text)
    # Replace curly quotes or angle quotes with simpler ASCII
    replacements = {
        '“': '"', '”': '"',
        '‘': "'", '’': "'",
        '«': '"', '»': '"'
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text

# -----------------------------------------------------------------------------
# Helper Functions: Timestamp Conversion
# -----------------------------------------------------------------------------
def timestamp_to_seconds(ts):
    parts = ts.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_parts = parts[2].split(",")
    seconds = int(sec_parts[0])
    milliseconds = int(sec_parts[1])
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

def seconds_to_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

# -----------------------------------------------------------------------------
# SRT File Parsing and Writing
# -----------------------------------------------------------------------------
def parse_srt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks_raw = re.split(r'\n\s*\n', content.strip())
    blocks = []
    for block in blocks_raw:
        lines = block.splitlines()
        if len(lines) >= 3:
            try:
                index = int(lines[0].strip())
            except:
                index = None
            time_line = lines[1].strip()
            time_match = re.match(r'(.+?)\s*-->\s*(.+)', time_line)
            if time_match:
                start_time = time_match.group(1).strip()
                end_time = time_match.group(2).strip()
            else:
                start_time = end_time = ""
            text = "\n".join(lines[2:])
            blocks.append({
                "index": index,
                "start": start_time,
                "end": end_time,
                "text": text
            })
    return blocks

def get_existing_block_count(output_path):
    if not os.path.exists(output_path):
        return 0
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return 0
    blocks = re.split(r'\n\s*\n', content)
    return len(blocks)

def append_blocks_to_output(new_blocks, output_path):
    # Only write BOM if file doesn't exist yet
    file_exists = os.path.exists(output_path)
    mode = "a" if file_exists else "w"
    encoding = "utf-8" if file_exists else "utf-8-sig"

    current_count = get_existing_block_count(output_path)
    with open(output_path, mode, encoding=encoding) as f:
        for i, block in enumerate(new_blocks, start=1):
            index = current_count + i
            f.write(f"{index}\n")
            f.write(f"{block['start']} --> {block['end']}\n")
            f.write(f"{block['text']}\n\n")

# -----------------------------------------------------------------------------
# LLM-Based Translation Function
# -----------------------------------------------------------------------------
def translate_text_simple(text, input_language, target_language, llm_model=DEFAULT_LLM_MODEL):
    wrapped_text = f"<<<{text}>>>"
    prompt = (
        f"Translate the following text from {input_language} to {target_language}.\n"
        "The text to be translated is enclosed between <<< and >>>.\n"
        "Provide ONLY the translated text on a SINGLE LINE, without any extra commentary."
    )
    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": wrapped_text}
        ],
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            result = response.json().get("message", {}).get("content", "").strip()
            return result
        else:
            if DEBUG:
                print(f"Translation error: {response.status_code} {response.text}")
            return "N/A"
    except Exception as e:
        if DEBUG:
            print(f"Exception during translation: {e}")
        return "N/A"

# -----------------------------------------------------------------------------
# Transliteration Function
# -----------------------------------------------------------------------------
def transliterate_sentence(translated_sentence, target_language):
    lang = target_language.lower()
    # Fallback to LLM-based transliteration
    prompt = (
        f"Transliterate the following sentence in {target_language} for English pronounciation.\n"
        "Provide ONLY the transliteration on a SINGLE LINE without ANY additional text or commentary."
    )
    payload = {
        "model": DEFAULT_LLM_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": translated_sentence}
        ],
        "stream": False
    }
    try:
        if DEBUG:
            print("Sending transliteration request via LLM fallback...")
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            result = response.json().get("message", {}).get("content", "").strip()
            return result
        else:
            if DEBUG:
                print(f"LLM transliteration error: {response.status_code} {response.text}")
            return ""
    except Exception as e:
        if DEBUG:
            print(f"Exception during LLM transliteration: {e}")
        return ""

# -----------------------------------------------------------------------------
# Highlighting Helper for Progressive Mode
# -----------------------------------------------------------------------------
def generate_progressive_highlighting(text, highlight_count):
    words = text.split()
    highlighted = []
    for i, word in enumerate(words):
        if i < highlight_count:
            # For the newly added (current) word, use bold formatting with orange color
            if i == highlight_count - 1:
                highlighted.append(f"<font color='#FFA500'><b>{word}</b></font>")
            else:
                highlighted.append(f"<font color='yellow'>{word}</font>")
        else:
            highlighted.append(word)
    return " ".join(highlighted)

# -----------------------------------------------------------------------------
# Process a Single SRT Block
# -----------------------------------------------------------------------------
def process_block(block, effective_mode, input_language, target_language,
                  llm_model, transliteration_enabled, highlighting_enabled):
    original_text = block["text"]
    # Get the translation
    translation = translate_text_simple(original_text, input_language, target_language, llm_model)

    # Possibly compute transliteration
    transliteration_text = ""
    if transliteration_enabled and target_language in NON_LATIN_LANGUAGES:
        transliteration_text = transliterate_sentence(translation, target_language)

    # Normalize / clean both translation & transliteration
    translation = normalize_and_clean_text(translation)
    transliteration_text = normalize_and_clean_text(transliteration_text)

    # We'll display the cleaned translation
    displayed_translation = translation

    # If RTL, wrap with directional markers
    if target_language in RTL_LANGUAGES:
        displayed_translation = f"\u202B{displayed_translation}\u202C"
        if transliteration_text:
            transliteration_text = f"\u202B{transliteration_text}\u202C"

    # Render based on mode
    if effective_mode == "1":
        new_text = displayed_translation
        if transliteration_text:
            new_text += "\n\n" + transliteration_text
        return [{"start": block["start"], "end": block["end"], "text": new_text}]

    elif effective_mode == "3":
        new_text = f"{block.get('index','')}\n{original_text}\n\n{displayed_translation}"
        if transliteration_text:
            new_text += "\n" + transliteration_text
        return [{"start": block["start"], "end": block["end"], "text": new_text}]

    elif effective_mode == "4":
        # Possibly do progressive highlighting
        if highlighting_enabled:
            start_sec = timestamp_to_seconds(block["start"])
            end_sec = timestamp_to_seconds(block["end"])
            duration = end_sec - start_sec
            words = displayed_translation.split()
            num_words = len(words)
            if num_words == 0:
                block_text = f"{original_text}\n{displayed_translation}"
                if transliteration_text:
                    block_text += "\n" + transliteration_text
                return [{"start": block["start"], "end": block["end"], "text": block_text}]
            step_duration = duration / num_words
            new_blocks = []
            for i in range(1, num_words + 1):
                step_start = start_sec + (i - 1) * step_duration
                step_end = start_sec + i * step_duration
                highlighted_translation = generate_progressive_highlighting(displayed_translation, i)
                block_text = f"{original_text}\n{highlighted_translation}"
                if transliteration_text:
                    highlighted_translit = generate_progressive_highlighting(transliteration_text, i)
                    block_text += f"\n{highlighted_translit}"
                new_blocks.append({
                    "start": seconds_to_timestamp(step_start),
                    "end": seconds_to_timestamp(step_end),
                    "text": block_text
                })
            return new_blocks
        else:
            new_text = f"{original_text}\n{displayed_translation}"
            if transliteration_text:
                new_text += "\n" + transliteration_text
            return [{"start": block["start"], "end": block["end"], "text": new_text}]

    else:
        new_text = displayed_translation
        if transliteration_text:
            new_text += "\n\n" + transliteration_text
        return [{"start": block["start"], "end": block["end"], "text": new_text}]

# -----------------------------------------------------------------------------
# Configuration File Handling
# -----------------------------------------------------------------------------
def get_config_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "subtitle.json")

def load_config():
    config_path = get_config_path()
    if os.path.exists(config_path):
        try:
            return json.load(open(config_path, "r", encoding="utf-8"))
        except Exception as e:
            if DEBUG:
                print(f"Error loading config: {e}")
    return {
        "input_file": "",
        "translation_mode": "1",
        "input_language": "English",
        "target_language": "Arabic",
        "llm_model": DEFAULT_LLM_MODEL,
        "blocks_per_batch": 10,
        "debug": False,
        "transliteration": False,
        "start_time": 0,
        "batch_mode": False,
        "highlighting": False,
        "thread_count": 5
    }

def save_config(config):
    config_path = get_config_path()
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        if DEBUG:
            print(f"Error saving config: {e}")

# -----------------------------------------------------------------------------
# Interactive Menu
# -----------------------------------------------------------------------------
def interactive_menu():
    config = load_config()
    srt_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".srt")]
    if not config.get("input_file"):
        config["input_file"] = srt_files[0] if srt_files else ""
    while True:
        print("\n=== SRT Subtitle Translator Settings ===")
        print(f"1. Input SRT file: {config['input_file']}")
        print(f"2. Translation mode: {config['translation_mode']} ({SRT_MODE_DESC.get(config['translation_mode'], '')})")
        print(f"3. Source language: {config['input_language']}")
        print(f"4. Target language: {config['target_language']}")
        print(f"5. LLM model: {config['llm_model']}")
        print(f"6. Blocks per batch: {config['blocks_per_batch']}")
        print(f"7. Debug mode: {'On' if config['debug'] else 'Off'}")
        print(f"8. Progressive highlighting toggle: {'On' if config.get('highlighting', False) else 'Off'}")
        print(f"9. Transliteration toggle: {'On' if config['transliteration'] else 'Off'}")
        print(f"10. Start time in minutes: {config.get('start_time', 0)}")
        print(f"11. Batch mode: {'On' if config.get('batch_mode', False) else 'Off'}")
        print(f"12. Thread count: {config.get('thread_count', 5)}")
        print("Press Enter (no input) to continue with these settings.")
        choice = input("Select a parameter number to change: ").strip()
        if not choice:
            break
        if choice == "1":
            print("\nAvailable SRT files in the directory:")
            for idx, fname in enumerate(srt_files, start=1):
                print(f"  {idx}. {fname}")
            sel = input("Enter the number of the SRT file to use: ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(srt_files):
                config["input_file"] = srt_files[int(sel)-1]
            else:
                print("Invalid selection. Keeping previous value.")
        elif choice == "2":
            print("\nSelect Translation Mode:")
            for key, desc in SRT_MODE_DESC.items():
                print(f"  {key}. {desc}")
            mode = input("Enter mode number: ").strip()
            if mode in SRT_MODE_DESC:
                config["translation_mode"] = mode
            else:
                print("Invalid mode. Keeping previous value.")
        elif choice == "3":
            print("\nSelect Source Language:")
            for idx, lang in enumerate(TOP_LANGUAGES, start=1):
                print(f"  {idx}. {lang}")
            sel = input("Enter the number of the source language (default: English): ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(TOP_LANGUAGES):
                config["input_language"] = TOP_LANGUAGES[int(sel)-1]
            else:
                config["input_language"] = "English"
        elif choice == "4":
            print("\nSelect Target Language:")
            for idx, lang in enumerate(TOP_LANGUAGES, start=1):
                print(f"  {idx}. {lang}")
            sel = input("Enter the number of the target language (default: Arabic): ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(TOP_LANGUAGES):
                config["target_language"] = TOP_LANGUAGES[int(sel)-1]
            else:
                config["target_language"] = "Arabic"
        elif choice == "5":
            print("\nSelect LLM Model:")
            for idx, model in enumerate(LLM_LIST, start=1):
                print(f"  {idx}. {model}")
            sel = input(f"Enter the number of the LLM model (default: {DEFAULT_LLM_MODEL}): ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(LLM_LIST):
                config["llm_model"] = LLM_LIST[int(sel)-1]
            else:
                config["llm_model"] = DEFAULT_LLM_MODEL
        elif choice == "6":
            num = input("Enter number of subtitle blocks per batch (default: 10): ").strip()
            if num.isdigit():
                config["blocks_per_batch"] = int(num)
            else:
                print("Invalid input. Keeping previous value.")
        elif choice == "7":
            dbg = input("Enable debug mode? (yes/no, default no): ").strip().lower()
            config["debug"] = True if dbg in ["yes", "y"] else False
        elif choice == "8":
            tog = input("Toggle progressive highlighting? (yes/no, default no): ").strip().lower()
            config["highlighting"] = True if tog in ["yes", "y"] else False
        elif choice == "9":
            tog = input("Toggle transliteration for non‑Latin languages? (yes/no, default no): ").strip().lower()
            config["transliteration"] = True if tog in ["yes", "y"] else False
        elif choice == "10":
            st = input("Enter start time in minutes (default 0): ").strip()
            if st.isdigit():
                config["start_time"] = int(st)
            else:
                print("Invalid input. Keeping previous value.")
        elif choice == "11":
            tog = input("Toggle batch mode? (yes/no, default no): ").strip().lower()
            config["batch_mode"] = True if tog in ["yes", "y"] else False
        elif choice == "12":
            num_threads = input("Enter number of threads to use (default: 5): ").strip()
            if num_threads.isdigit() and int(num_threads) > 0:
                config["thread_count"] = int(num_threads)
            else:
                print("Invalid input. Keeping previous value.")
        else:
            print("Invalid choice. Try again.")
    save_config(config)
    return config

# -----------------------------------------------------------------------------
# Time Format Helper
# -----------------------------------------------------------------------------
def format_hhmmss(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# -----------------------------------------------------------------------------
# Main Processing Function
# -----------------------------------------------------------------------------
def main():
    config = interactive_menu()
    script_start = time.time()
    if "-b" in sys.argv:
        config["batch_mode"] = True

    effective_mode = config["translation_mode"]
    input_lang = config["input_language"]
    target_lang = config["target_language"]
    llm_model = config["llm_model"]
    blocks_per_batch = config["blocks_per_batch"]
    thread_count = config.get("thread_count", 5)
    transliteration_enabled = config["transliteration"]
    highlighting_enabled = config.get("highlighting", False)

    # Determine which files to process
    if config.get("batch_mode", False):
        batch_folder = os.path.join(OUTPUT_DIR, "batch")
        if not os.path.exists(batch_folder):
            print(f"Batch folder '{batch_folder}' not found. Exiting.")
            return
        files_to_process = [
            os.path.join(batch_folder, f)
            for f in os.listdir(batch_folder)
            if f.lower().endswith(".srt")
        ]
        if not files_to_process:
            print(f"No SRT files found in batch folder '{batch_folder}'. Exiting.")
            return
    else:
        files_to_process = [config["input_file"]]

    # If in batch mode, compute overall SRT duration for all files
    if config.get("batch_mode", False):
        overall_total_srt_duration = 0
        file_durations = {}
        for file in files_to_process:
            blocks_temp = parse_srt(file)
            if blocks_temp:
                duration = timestamp_to_seconds(blocks_temp[-1]["end"])
            else:
                duration = 0
            file_durations[file] = duration
            overall_total_srt_duration += duration
        overall_processed_srt_duration = 0
        overall_start_time = time.time()

    for input_file in files_to_process:
        print(f"\nProcessing SRT file: {input_file}")
        blocks = parse_srt(input_file)
        total_blocks = len(blocks)
        print(f"Total subtitle blocks found: {total_blocks}")

        # Filter by start time in minutes
        start_time_min = config.get("start_time", 0)
        if start_time_min > 0:
            start_time_sec = start_time_min * 60
            blocks = [
                block for block in blocks
                if timestamp_to_seconds(block["start"]) >= start_time_sec
            ]
            print(f"Filtered blocks with start time >= {start_time_min} minutes. Total blocks now: {len(blocks)}")

        base_name, _ = os.path.splitext(os.path.basename(input_file))
        output_filename = f"{input_lang}_{target_lang}_{base_name}_translated.srt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        batch_number = 1
        if config.get("batch_mode", False):
            file_duration = file_durations[input_file]
        else:
            file_duration = timestamp_to_seconds(blocks[-1]["end"])

        pbar = tqdm(total=file_duration, desc=f"Processing {base_name}", bar_format="{l_bar}{bar} {postfix}")
        pbar.set_postfix_str("Batch: 0 | SRT: 00:00 | Speed: 0.00 cps | File ETA: 00:00:00 | Overall ETA: calculating...")
        file_start_time = time.time()
        total_chars = 0

        from concurrent.futures import ThreadPoolExecutor
        num_blocks = len(blocks)
        
        # Process blocks in batches concurrently
        for batch_start in range(0, num_blocks, blocks_per_batch):
            batch = blocks[batch_start: batch_start + blocks_per_batch]
            
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                results = list(executor.map(lambda block: process_block(
                    block, effective_mode, input_lang, target_lang,
                    llm_model, transliteration_enabled, highlighting_enabled
                ), batch))
            
            batch_blocks = []
            for res in results:
                batch_blocks.extend(res)
            
            if batch:
                current_progress = timestamp_to_seconds(batch[-1]["end"])
                delta = current_progress - pbar.n
                pbar.update(delta)
            
            for new_block in batch_blocks:
                total_chars += len(new_block["text"])
            elapsed_time = time.time() - file_start_time
            avg_cps = total_chars / elapsed_time if elapsed_time > 0 else 0
            processed_blocks = batch_start + len(batch)
            estimated_total_chars = (total_chars / processed_blocks) * num_blocks if processed_blocks > 0 else 0
            remaining_chars = estimated_total_chars - total_chars
            eta_seconds = remaining_chars / avg_cps if avg_cps > 0 else 0
            file_eta = eta_seconds

            if config.get("batch_mode", False):
                overall_current_progress = overall_processed_srt_duration + pbar.n
                overall_elapsed = time.time() - overall_start_time
                if overall_current_progress > 0:
                    overall_speed = overall_current_progress / overall_elapsed
                    overall_eta = (overall_total_srt_duration - overall_current_progress) / overall_speed
                else:
                    overall_eta = overall_total_srt_duration
                overall_eta_str = format_hhmmss(overall_eta)
            else:
                overall_eta_str = "N/A"
            
            current_srt_time = time.strftime("%M:%S", time.gmtime(pbar.n))
            runtime_str = format_hhmmss(time.time() - script_start)
            formatted_postfix = (
                f"Batch: {batch_number} | {processed_blocks}/{num_blocks} blocks | Runtime: {runtime_str} | SRT: {current_srt_time} | Speed: {avg_cps:.2f} cps "
                f"File ETA: {format_hhmmss(file_eta)} | Overall ETA: {overall_eta_str}"
            )
            pbar.set_postfix_str(formatted_postfix)
            
            append_blocks_to_output(batch_blocks, output_path)
            batch_number += 1

        pbar.close()
        if config.get("batch_mode", False):
            overall_processed_srt_duration += file_duration

        print(f"\nAll blocks processed for {input_file}. Final output file: {output_path}\n")

if __name__ == "__main__":
    main()
