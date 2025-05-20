# src/config.py

AUDIO_PATH = "data/raw/match_sample.mp3"
OUTPUT_PREFIX = "outputs/transcripts/match_sample"
WHISPER_MODEL = "small"  # Options: "tiny", "base", "small", "medium", "large"
GAZETTEER_PATH = "data/gazetteers/players_and_teams.json"

# Logging controls
LOG_TRANSCRIPT_PREVIEW = True
LOG_SENTENCES = False
LOG_NER = False
LOG_EMPHASIS = False
LOG_EVENTS = True

# Substitution control
ENABLE_CANONICAL_SUBSTITUTION = True



SHOULD_TRANSCRIBE = False  # Set to False to skip transcription