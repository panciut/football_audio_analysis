# src/asr/whisper_transcriber.py

import whisper
import time
import os
import json

def transcribe_audio(file_path: str, model_size: str = "base", save_to: str = None) -> dict:
    """
    Transcribes the given audio file using Whisper and optionally saves the output.

    Args:
        file_path (str): Path to the audio file.
        model_size (str): Model to use ("tiny", "base", "small", "medium", "large").
        save_to (str): Path to save the result as JSON (optional).

    Returns:
        dict: A dictionary with text, segments, and language.
    """
    print(f"⏳ Loading Whisper model '{model_size}' (this may take a moment)...")
    model = whisper.load_model(model_size)

    print(f"🎧 Transcribing: {file_path}")
    start_time = time.time()
    result = model.transcribe(file_path)
    end_time = time.time()

    print(f"✅ Transcription complete in {end_time - start_time:.2f} seconds.")
    
    output = {
        "text": result["text"],
        "segments": result["segments"],
        "language": result["language"]
    }

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"💾 Transcript saved to {save_to}")

    return output
