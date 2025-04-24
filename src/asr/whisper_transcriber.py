# src/asr/whisper_transcriber.py

import whisper
import time
import os
import json

def transcribe_audio(file_path: str, model_size: str = "base", save_to: str = None) -> dict:
    """
    Transcribes the given audio file using Whisper and optionally saves structured and plain-text output.

    Args:
        file_path (str): Path to the audio file.
        model_size (str): One of "tiny", "base", "small", etc.
        save_to (str): Output prefix path (no extension). Two files will be saved: `.json` and `.txt`.

    Returns:
        dict: Transcript with full text, segments, and language.
    """
    print(f"⏳ Loading Whisper model '{model_size}' (this may take a moment)...")
    model = whisper.load_model(model_size)

    print(f"🎧 Transcribing: {file_path}")
    start = time.time()
    result = model.transcribe(file_path)
    end = time.time()
    print(f"✅ Transcription complete in {end - start:.2f} seconds.")

    output = {
        "text": result["text"],
        "segments": result["segments"],
        "language": result["language"]
    }

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)

        # Save structured output (.json)
        with open(save_to + ".json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        # Save human-readable transcript (.txt)
        with open(save_to + ".txt", "w", encoding="utf-8") as f:
            f.write("FULL TRANSCRIPT\n\n")
            for segment in result["segments"]:
                start_ts = format_timestamp(segment["start"])
                end_ts = format_timestamp(segment["end"])
                f.write(f"[{start_ts} - {end_ts}] {segment['text']}\n")
        
        print(f"💾 Transcript saved to {save_to}.json and {save_to}.txt")

    return output


def format_timestamp(seconds: float) -> str:
    """Converts float seconds to H:MM:SS format."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02}:{s:02}"
