# src/utils/io_utils.py

import os
import json

def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02}:{s:02}"

def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved to {path}")

def save_plaintext(segments, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("FULL TRANSCRIPT\n\n")
        for seg in segments:
            ts_start = format_timestamp(seg["start"])
            ts_end = format_timestamp(seg["end"])
            f.write(f"[{ts_start} - {ts_end}] {seg['text']}\n")
    print(f"💾 Saved to {path}")

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
