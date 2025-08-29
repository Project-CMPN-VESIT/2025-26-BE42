# Local Data Agent (LDA) — README

A local-first pipeline for ingesting audio/video/text sessions with privacy-first storage (AES-GCM encrypted at rest), wav2vec2 audio embeddings, optional segment-level ASR post-fill, face tracking + OpenFace AU extraction, and simple QA assembly.

This README is a single-file reference containing everything implemented so far, plus exact commands and config snippets so you can reproduce the system end-to-end.

---

## Table of contents

1. Quick summary  
2. Repo layout  
3. System & Python installation (exact commands)  
4. OpenFace: download / build / initialization  
5. Full recommended configuration (`configs/local_config.yaml`)  
6. Model pre-download (avoid transformers cache migration / multiprocessing issues)  
7. Running the API & pipeline (examples)  
8. Decrypting and inspecting outputs (parquet / clips)  
9. Exporting flattened CSV (embedding expansion)  
10. Troubleshooting (common errors & fixes)  
11. Security & privacy notes  
12. Next steps & optional extras  
13. Backwards-compatibility alias (text)  
14. Useful helper scripts (suggested tools)

---

## 1 — Quick summary

- Audio pipeline computes **wav2vec2 embeddings** (chunked to avoid OOM).  
- `session_processor` runs VAD → diarization (pyannote optional) → per-segment transcription attempts → **post-fill** of missing transcripts using configured ASR backend (`whisper` or Hugging Face `transformers`).  
- Video pipeline detects faces (OpenCV Haar), produces a faces-blurred MP4, and runs **OpenFace FeatureExtraction** (CSV output) when enabled.  
- All artifacts (clips, processed video, OpenFace CSVs, parquet manifests) are encrypted at rest using a `SecureStore` (AES-GCM with HKDF-derived per-directory keys).  
- Per-operation receipts are written and HMAC-signed (`ReceiptManager`).  
- Tools suggested: `tools/decrypt_and_write.py`, `tools/export_session_csv.py`, `tools/asr_runner.py`.

---

## 2 — Repo layout (recommended)

```

LDA/
├─ app/
│  ├─ main.py
│  ├─ pipelines/
│  │  ├─ audio.py
│  │  ├─ session\_processor.py
│  │  ├─ text.py
│  │  ├─ video.py
│  ├─ security/
│  │  ├─ secure\_store.py
│  ├─ utils/
│  │  ├─ receipts.py
├─ configs/
│  ├─ local\_config.yaml
├─ tools/
│  ├─ export\_session\_csv.py
│  ├─ decrypt\_and\_write.py
│  ├─ asr\_runner.py
├─ requirements.txt
└─ README.md

````

The files above are the ones we implemented / iterated on in the session.

---

## 3 — System & Python installation

Assumes Linux (Ubuntu/Debian). Adjust for macOS/Windows.

### System packages
```bash
sudo apt update
sudo apt install -y ffmpeg build-essential cmake git wget unzip
sudo apt install -y libsm6 libxext6 libxrender-dev
````

### Python virtualenv & core packages

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install fastapi uvicorn pyyaml pandas pyarrow cryptography numpy soundfile
pip install "transformers>=4.30.0" torch
pip install librosa
```

Optional heavy extras:

```bash
pip install openai-whisper       # Whisper ASR (if you want)
pip install opencv-python        # video processing (blurring/detection)
# pyannote.audio (diarization) follows its own install instructions (HF token, extra deps)
```

**Note:** Install `torch` using the wheel appropriate to your CUDA version if using GPU.

---

## 4 — OpenFace: setup & initialization

Video pipeline will call the OpenFace `FeatureExtraction` binary if `video_pipe.openface.enabled` is true.

### Option A — Use a prebuilt binary

Place `FeatureExtraction` somewhere and point `video_pipe.openface.binary_path` at the absolute path. Ensure `classifiers` (Haar xml) exist (either relative to binary or set `haar_path`).

### Option B — Build from source (summary)

```bash
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
# follow the OpenFace README for dependencies (dlib, OpenCV, CMake etc.)
mkdir build && cd build
cmake ..
make -j$(nproc)
# binary: build/bin/FeatureExtraction
```

### Haar cascade classifiers

The pipeline will try (in order):

1. `video_pipe.openface.haar_path` (config),
2. `../classifiers/*.xml` relative to OpenFace binary,
3. OpenCV built-in cascade `cv2.data.haarcascades + "haarcascade_frontalface_default.xml"`.

If OpenFace produced `classifiers/*.xml`, set `haar_path` explicitly for best results.

---

## 5 — Full recommended configuration (`configs/local_config.yaml`)

Create `configs/local_config.yaml` with this content and edit absolute paths:

```yaml
mode: interactive

ingest:
  audio:
    enabled: true
    sr: 16000
    receipt_dir: "./receipts"
    persist_clips: true
    features:
      wav2vec2:
        enabled: true
        model: "facebook/wav2vec2-base-960h"
        pool: "mean"
        chunk_seconds: 20.0
        device: null   # "cpu" | "cuda" | null (auto)

  video:
    enabled: true
    output_dir: "./processed/video"
    receipt_dir: "./receipts"
    persist_clips: true

  text:
    enabled: true
    output_dir: "./processed/text"
    receipt_dir: "./receipts"

privacy_policy:
  export_raw_media: false
  pii_scrub_text: true

video_pipe:
  head_pose_normalize: true
  openface:
    enabled: true
    binary_path: "/absolute/path/to/OpenFace/build/bin/FeatureExtraction"
    haar_path: "/absolute/path/to/OpenFace/build/classifiers/haarcascade_frontalface_alt.xml"

audio_pipe:
  vad: "webrtc"
  denoise: true

text_pipe:
  asr_enabled: true
  asr_backend: "hf"         # "whisper" or "hf"
  asr_model: "facebook/wav2vec2-base-960h"

storage:
  root: "./secure_store"
```

---

## 6 — Model pre-download (avoid Transformers cache migration / multiprocessing issues)

Run these once before starting uvicorn (especially if you use `--reload` or multiple workers):

```bash
# Preload Wav2Vec2 model & processor
python - <<'PY'
from transformers import Wav2Vec2Processor, Wav2Vec2Model
name = "facebook/wav2vec2-base-960h"
Wav2Vec2Processor.from_pretrained(name)
Wav2Vec2Model.from_pretrained(name)
print("Wav2Vec2 cached")
PY

# If using HF ASR pipeline
python - <<'PY'
from transformers import pipeline
pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
print("ASR pipeline cached")
PY

# If using Whisper
python - <<'PY'
import whisper
whisper.load_model("small")
print("Whisper cached")
PY
```

This prevents the "cache migration" step or parallel downloads causing uvicorn spawn-time errors.

---

## 7 — Running the API & pipeline

Start the FastAPI server (development):

```bash
uvicorn app.main:app --reload --port 8000
```

Call the `/local/preprocess` endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/local/preprocess" \
  -H "Content-Type: application/json" \
  -d '{
    "mode":"session",
    "inputs": {
      "video_dir": "input_videos",
      "audio_dir": "input_wavs",
      "text_dir": "input_texts"
    },
    "config_uri": "file://configs/local_config.yaml"
  }'
```

* `mode = session` processes `.mp4` files in `video_dir`. If a matching `.wav` with the same stem exists in `audio_dir`, it will be used instead of extracting audio from the video file.
* Response contains `session_id`, `artifact_manifest` (encrypted `file://` URI), `receipts` and `count`.

---

## 8 — Decrypting and inspecting outputs

All stored artifacts are encrypted JSON files with base64 `nonce` + `ct`. Use `SecureStore.decrypt_read(uri)`.

Example:

```python
from app.security.secure_store import SecureStore
import pyarrow.parquet as pq, io

store = SecureStore("./secure_store")
manifest_bytes = store.decrypt_read("file://secure_store/sess-.../manifest/2025-08-25/18.jsonl.enc")
print(manifest_bytes.decode())

parquet_bytes = store.decrypt_read("file://secure_store/sess-.../session/2025-08-25/18.parquet.enc")
buf = io.BytesIO(parquet_bytes)
table = pq.read_table(buf)
df = table.to_pandas()
print(df.head())
```

**Tip:** If you want a CLI helper, create `tools/decrypt_and_write.py` to provide `--uri` and `--out` options.

---

## 9 — Exporting flattened CSV (embedding expansion)

Parquet rows include a `features` dict with `w2v2_embedding` (a list). To expand embeddings into columns:

```python
import io
import pyarrow.parquet as pq
from app.security.secure_store import SecureStore
import pandas as pd

store = SecureStore("./secure_store")
b = store.decrypt_read("file://secure_store/sess-.../session/2025-08-25/18.parquet.enc")
buf = io.BytesIO(b)
df = pq.read_table(buf).to_pandas()

embs = df['features'].apply(lambda f: f.get('w2v2_embedding') if isinstance(f, dict) else None)
emb_df = pd.DataFrame(embs.tolist())
emb_df.columns = [f"w2v2_{i}" for i in emb_df.columns]
df_flat = pd.concat([df.drop(columns=['features']), emb_df], axis=1)
df_flat.to_csv("session_flat.csv", index=False)
```

**Note:** Flattened CSVs with high-dim embeddings get very large. Consider storing embeddings as `.npy` per-session and only writing references in parquet.

---

## 10 — Troubleshooting (common errors & fixes)

### `cryptography.exceptions.InvalidTag`

* Causes: HKDF `info` mismatch, moved files, wrong master key encoding (raw vs base64), truncated ciphertext.
* Fixes:

  * Ensure `SecureStore` uses the same master secret and context (we derive keys from `str(p.parent)`).
  * Use a persistent master key:

    ```bash
    python - <<'PY'
    ```

import base64, secrets
print(base64.b64encode(secrets.token\_bytes(32)).decode())
PY

# copy output and:

export SECURE\_STORE\_MASTER\_KEY="PASTE\_BASE64\_KEY"

````

### `ImportError: cannot import name 'process_text_sources'`
- If `main.py` imports `process_text_sources` but `text.py` defines `process_text_file`, add alias at end of `text.py`:
```python
process_text_sources = process_text_file
````

### `transformers` cache-migration / SpawnProcess errors on uvicorn startup

* Pre-download models (section 6) before starting uvicorn/workers.

### `Wav2Vec2 warning: Some weights ... newly initialized`

* Warning appears when loading `Wav2Vec2Model` (feature extractor) but not CTC head. For ASR use `Wav2Vec2ForCTC` or a fine-tuned CTC model to avoid the warning.

### OOM during embedding computation

* Reduce `chunk_seconds` in config (5–10s), or run on a machine with larger GPU memory or CPU-only (slower).

### `ffmpeg not found`

* Install system `ffmpeg` (see section 3).

---

## 11 — Security & privacy notes

* **Encryption:** All artifacts under `storage.root` are AES-GCM encrypted using HKDF-derived keys from a master secret. Protect `SECURE_STORE_MASTER_KEY` or `master.key`. Losing the master key = losing data.
* **PII scrubbing:** `text.py` uses regex + optional spaCy NER; this is heuristic. For compliance, use validated redaction and human review.
* **Model cache:** HF model cache lives in `~/.cache/huggingface/transformers`. Secure it if required.

---

## 12 — Next steps & optional extras

* Provide `tools/export_session_csv.py` to decrypt and combine parquets into a single flattened CSV (I can produce this).
* Store embeddings as `float32` `.npy` per session for efficient ML loading.
* Add background worker (Celery/RQ) for ASR post-fill to keep API responses fast and handle heavy jobs asynchronously.
* Integrate `pyannote.audio` for better diarization (requires HF token).
* Add unit/integration tests and a small sample dataset.

---

## 13 — Backwards-compatibility alias (text)

If `main.py` or other callers still import `process_text_sources`, add this alias to the bottom of `app/pipelines/text.py`:

```python
# backwards compatibility
process_text_sources = process_text_file
```

---

## 14 — Useful helper scripts (suggested content)

`tools/decrypt_and_write.py` — decrypt a `file://` URI and write the raw bytes to a file:

```python
# pseudo: implement in tools/decrypt_and_write.py
from app.security.secure_store import SecureStore
import argparse, os

p = argparse.ArgumentParser()
p.add_argument("--uri", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

s = SecureStore("./secure_store")
data = s.decrypt_read(args.uri)
with open(args.out, "wb") as f:
    f.write(data)
print("Wrote", args.out)
```

`tools/export_session_csv.py` — decrypt session parquet(s) for a `session_id`, flatten embeddings and write CSV (you can request I provide full code).

`tools/asr_runner.py` — a simple script to run ASR on encrypted audio clips (decrypt, run ASR backend, re-encrypt transcript / create receipt).

---

## Final quick commands

Pre-download models:

```bash
# wav2vec2
python - <<'PY'
from transformers import Wav2Vec2Processor, Wav2Vec2Model
n = "facebook/wav2vec2-base-960h"
Wav2Vec2Processor.from_pretrained(n)
Wav2Vec2Model.from_pretrained(n)
print("cached")
PY
```

Start server:

```bash
uvicorn app.main:app --reload --port 8000
```

Decrypt & inspect a parquet:

```python
from app.security.secure_store import SecureStore
import pyarrow.parquet as pq, io
s = SecureStore("./secure_store")
b = s.decrypt_read("file://secure_store/sess-.../session/...parquet.enc")
buf = io.BytesIO(b)
df = pq.read_table(buf).to_pandas()
print(df.head())
```

---


