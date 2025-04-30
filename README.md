# ⚽ Football Commentary Summarization System

## This project processes football match commentary audio into a structured, readable summary of what happened during the game. The pipeline involves automatic transcription, enrichment (prosody/emphasis), linguistic analysis, and event-driven summarization.

## ✅ Current Features

| Feature            | Status        | Description                                                              |
| ------------------ | ------------- | ------------------------------------------------------------------------ |
| Transcription      | ✔ Implemented | Uses Whisper to transcribe match audio                                   |
| Prosody Extraction | ✔ Implemented | Extracts pitch and energy for emphasis detection using `librosa`         |
| Emphasis Scoring   | ✔ Implemented | Scores utterances for emotional/semantic importance                      |
| Output Saving      | ✔ Implemented | Saves transcript as `.json` and `.txt` formats                           |
| Modular Codebase   | ✔ Implemented | Logic split into `pipeline/`, `utils/`, and configurable via `config.py` |

---

## 🔧 Project Structure

```bash
src/
├── config.py              # Central config values
├── main.py                # Pipeline entry point
├── utils/
│   ├── io.py              # Save JSON/TXT and format timestamps
├── pipeline/
│   ├── transcriber.py     # Whisper + prosody extraction
│   ├── emphasis.py        # Emphasis scoring logic
```

---

## 🚧 Development Roadmap

### 🧱 Stage 1 — Text Pipeline Foundation

| Step | Component            | Task                                                  | Tools/Notes        |
| ---- | -------------------- | ----------------------------------------------------- | ------------------ |
| 1.1  | **Transcription**    | Use Whisper to transcribe audio. Already implemented. | `whisper`          |
| 1.2  | **Prosody Analysis** | Extract pitch and energy. Already implemented.        | `librosa`          |
| 1.3  | **Text Correction**  | Clean up transcription artifacts and fix grammar.     | Small LLM or rules |

➡️ **Goal:** Clean, accurate, and enriched transcript.

---

### 🧠 Stage 2 — Linguistic & Structural Processing

| Step | Component                 | Task                                                             | Tools/Notes                 |
| ---- | ------------------------- | ---------------------------------------------------------------- | --------------------------- |
| 2.1  | **Sentence Segmentation** | Break transcript into manageable units (sentences/utterances).   | `spaCy`, rule-based         |
| 2.2  | **Speaker Attribution**   | (Optional) Tag speaker turns if multi-speaker setup is detected. | Simplified labeling         |
| 2.3  | **NER & Normalization**   | Detect player names, teams, competitions, time expressions.      | `spaCy`, `stanza`, patterns |
| 2.4  | **Time Tagging**          | Extract and normalize time mentions ("5th minute", "early on").  | Regex + logic               |

➡️ **Goal:** Transcript is tokenized, tagged, and semantically structured.

---

### ⚽ Stage 3 — Game Event Extraction

| Step | Component                 | Task                                                  | Tools/Notes              |
| ---- | ------------------------- | ----------------------------------------------------- | ------------------------ |
| 3.1  | **Event Schema**          | Define events (goal, shot, foul, substitution, etc.). | JSON schema, YAML        |
| 3.2  | **Rule-Based Extraction** | Extract events using keyword patterns + time anchors. | Regex, heuristics        |
| 3.3  | **Entity Linking**        | Link players to actions and teams, avoid ambiguity.   | Memory buffer/dictionary |
| 3.4  | **Chronology Building**   | Order all events by inferred or explicit time.        | Event timestamping logic |

➡️ **Goal:** A full timeline of structured match events.

---

### 📝 Stage 4 — Summary Generation

| Step | Component                        | Task                                                                     | Tools/Notes              |
| ---- | -------------------------------- | ------------------------------------------------------------------------ | ------------------------ |
| 4.1  | **Template-Based Summary**       | Generate summaries from structured event lists using sentence templates. | Python templates         |
| 4.2  | **LLM Summarization (optional)** | Use LLM (e.g., `llama.cpp`, `mistral`, etc.) to generate fluent output.  | Lightweight local LLM    |
| 4.3  | **Highlight Generation**         | Emphasize most important events (e.g., goals, red cards).                | Emphasis scoring + logic |

➡️ **Goal:** Readable and informative match summary, automatically generated.

---

### 🔁 Stage 5 — Evaluation and Iteration

| Step | Component                 | Task                                                        | Tools/Notes                   |
| ---- | ------------------------- | ----------------------------------------------------------- | ----------------------------- |
| 5.1  | **Human Review**          | Manually verify accuracy of extracted events and summaries. | CSV logs, plain text          |
| 5.2  | **Component Tuning**      | Improve event detection, summarization fluency, or NER.     | Evaluate with match reports   |
| 5.3  | **End-to-End Test Suite** | Create test cases for the full pipeline.                    | Python `unittest` or `pytest` |

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Ensure `ffmpeg` is installed and available in your system path.

---

## 🚀 Run the Pipeline

```bash
python src/main.py
```
