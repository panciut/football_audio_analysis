# src/emphasis/scorer.py

import numpy as np

def score_emphasis(segments, pitch_weight=0.5, energy_weight=0.5, threshold_percentile=85):
    pitches = np.array([s["pitch"] for s in segments])
    energies = np.array([s["energy"] for s in segments])

    norm_pitches = (pitches - pitches.min()) / (pitches.ptp() + 1e-6)
    norm_energies = (energies - energies.min()) / (energies.ptp() + 1e-6)

    scores = pitch_weight * norm_pitches + energy_weight * norm_energies
    threshold = np.percentile(scores, threshold_percentile)

    for i, seg in enumerate(segments):
        seg["emphasis_score"] = round(float(scores[i]), 4)
        seg["emphasized"] = bool(scores[i] >= threshold)  # <-- Fix here

    return segments
