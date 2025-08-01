import os
import json
import numpy as np
import parselmouth
from praatio import textgrid
from tqdm import tqdm  # For a helpful progress bar

# --- Configuration ---
# Directory containing your TextGrid alignment files from MFA
ALIGN_DIR = "/Users/kennychan/Downloads/MCE_Dataset/mfa_training_merged/alignment"
# Directory containing your audio files (both .wav and .mp3)
AUDIO_DIR = "/Users/kennychan/Downloads/MCE_Dataset/consolidated_wavs"
# The name of the output JSON file
OUTPUT_JSON = "merged_data_tone_pitch_features.json"

def extract_features():
    """
    Extracts pitch features (average frequency and slope) for each toned syllable
    based on MFA alignments and audio files (supports .wav and .mp3).
    """
    tone_features = []
    
    # Get a list of TextGrid files to process
    tg_files = [f for f in os.listdir(ALIGN_DIR) if f.endswith(".TextGrid")]
    if not tg_files:
        print(f"Error: No .TextGrid files found in {ALIGN_DIR}")
        return

    print(f"Found {len(tg_files)} TextGrid files. Starting feature extraction...")

    # Use tqdm for a progress bar
    for tg_file in tqdm(tg_files, desc="Processing files"):
        audio_base = os.path.splitext(tg_file)[0]
        # Check for both .wav and .mp3 extensions
        audio_paths = [
            os.path.join(AUDIO_DIR, f"{audio_base}.wav"),
            os.path.join(AUDIO_DIR, f"{audio_base}.mp3")
        ]
        # Find the existing audio file
        audio_path = next((p for p in audio_paths if os.path.exists(p)), None)

        if not audio_path:
            # Skip if no matching audio file
            continue

        # --- Load Alignment and Audio ---
        try:
            # Load TextGrid, ignoring intervals with empty labels
            tg_path = os.path.join(ALIGN_DIR, tg_file)
            tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
            
            # Get the tiers for words and phones
            words_tier = tg.getTier("words")
            phones_tier = tg.getTier("phones")
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not process {tg_file} due to an error: {e}. Skipping.")
            continue

        # --- Sanity Check ---
        if len(words_tier.entries) != len(phones_tier.entries):
            continue

        # --- Pitch Extraction ---
        try:
            # Load audio (supports .wav and .mp3 if ffmpeg is installed)
            sound = parselmouth.Sound(audio_path)
        except Exception as e:
            print(f"Warning: Could not load audio {audio_path}: {e}. Skipping.")
            continue
        
        # Calculate pitch contour with high resolution
        pitch = sound.to_pitch(time_step=0.001)
        pitch_times = pitch.ts()
        all_pitch_values = pitch.selected_array['frequency']

        # --- Iterate Through Each Word/Syllable ---
        for i in range(len(words_tier.entries)):
            word_entry = words_tier.entries[i]
            phone_entry = phones_tier.entries[i]

            word = word_entry.label
            phone = phone_entry.label
            
            # Skip special symbols
            if phone in ["sil", "spn"]:
                continue

            # Extract tone (last character if it's a digit)
            tone = phone[-1]
            if not tone.isdigit():
                continue

            word_start, word_end = word_entry.start, word_entry.end

            # --- Select Pitch Values in Interval ---
            indices = (pitch_times >= word_start) & (pitch_times < word_end)
            pitch_values_in_interval = all_pitch_values[indices]
            pitch_values = pitch_values_in_interval[pitch_values_in_interval > 0]
            
            if len(pitch_values) < 3:
                continue

            # --- Calculate Features ---
            avg_freq = np.mean(pitch_values)
            time_values = pitch_times[indices][pitch_values_in_interval > 0]
            slope = np.polyfit(time_values, pitch_values, 1)[0]

            # Append results
            tone_features.append({
                "audio": audio_base,
                "word": word,
                "phone": phone,
                "tone": tone,
                "start": round(word_start, 3),
                "end": round(word_end, 3),
                "avg_freq_hz": round(avg_freq, 2),
                "pitch_slope": round(slope, 2)
            })

    # --- Save Results ---
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(tone_features, f, ensure_ascii=False, indent=2)

    print(f"\nExtraction complete.")
    print(f"Successfully extracted {len(tone_features)} features.")
    print(f"Results saved to {OUTPUT_JSON}")

# --- Run the main function ---
if __name__ == "__main__":
    extract_features()