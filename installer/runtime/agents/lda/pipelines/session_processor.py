# LDA/app/pipelines/session_processor.py
"""
Session processor for the Local Data Agent (Privacy).

Provides:
- process_session_file(...) : main entrypoint used by the /session/process route.
- helper functions (lazy-load heavy libs, ffmpeg wrappers, VAD, diarization, transcription,
  AV mapping, clip extraction + encryption, feature extraction, QA assembly).

Behavior:
- Tries to use high-quality libs when available (pyannote, whisper, librosa, mediapipe).
- Falls back to simpler implementations otherwise (energy-based VAD, single-speaker fallback).
- All persisted artifacts (clips, integrated parquet, manifests, receipts) are written
  using SecureStore.encrypt_write so data at rest is encrypted.
"""

import json
import os
import subprocess
import tempfile
import wave
import contextlib
import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import shutil
import logging

from core.centralized_secure_store import SecureStore
from core.centralised_receipts import CentralReceiptManager

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --------------------------
# Utilities & lazy imports
# --------------------------
def _which(cmd: str) -> Optional[str]:
    """Return path to executable or None."""
    from shutil import which
    return which(cmd)


def _ensure_ffmpeg():
    if not _which("ffmpeg"):
        raise EnvironmentError("ffmpeg not found on PATH. Install ffmpeg to enable audio/video extraction.")


def _safe_import(name: str):
    """Try importlib to import a module by name. Return module or None."""
    try:
        import importlib
        return importlib.import_module(name)
    except Exception:
        return None


# Try optional libs (set to None if missing)
_librosa = _safe_import("librosa")
_npy = _safe_import("numpy")
_webrtcvad = _safe_import("webrtcvad")
_pyannote = _safe_import("pyannote.audio")
_whisper = _safe_import("whisper")  # openai/whisper package (if installed)
_transformers = _safe_import("transformers")
_torch = _safe_import("torch")

# Face tracking libs (mediapipe, face_recognition)
_mediapipe = _safe_import("mediapipe")
_face_recognition = _safe_import("face_recognition")


# --------------------------
# Audio / Video helpers
# --------------------------
def _extract_audio_from_video(video_path: str, out_wav_path: str, sr: int = 16000) -> str:
    """
    Extract audio from video using ffmpeg and resample to sr (Hz) mono wav.
    Returns the path to the created wav file.
    """
    _ensure_ffmpeg()
    out_path = Path(out_wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", str(sr),
        "-vn", "-hide_banner", "-loglevel", "error",
        str(out_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {proc.stderr.strip()}")
    return str(out_path)


def _cut_audio_segment(input_wav: str, out_wav: str, start: float, end: float) -> str:
    """
    Use ffmpeg to cut an audio segment [start, end) into out_wav.
    """
    _ensure_ffmpeg()
    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, float(end) - float(start))
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-t", str(duration),
        "-i", str(input_wav),
        "-ac", "1", "-ar", "16000",
        str(out_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg cut audio failed: {proc.stderr.strip()}")
    return str(out_path)


def _cut_video_segment(input_video: str, out_video: str, start: float, end: float) -> str:
    """
    Use ffmpeg to cut a video segment [start, end) into out_video.
    """
    _ensure_ffmpeg()
    out_path = Path(out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, float(end) - float(start))
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-t", str(duration),
        "-i", str(input_video),
        "-c", "copy",
        str(out_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # try a re-encode fallback (some formats don't allow -c copy with -ss)
        cmd2 = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start), "-t", str(duration),
            "-i", str(input_video),
            "-c:v", "libx264", "-c:a", "aac", "-strict", "-2",
            str(out_path)
        ]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True)
        if proc2.returncode != 0:
            raise RuntimeError(f"ffmpeg cut video failed: {proc.stderr.strip()} | {proc2.stderr.strip()}")
    return str(out_path)


def _wav_duration(wav_path: str) -> float:
    try:
        with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        # fallback: if librosa available
        if _librosa:
            y, sr = _librosa.load(wav_path, sr=None, mono=True)
            return len(y) / float(sr)
        return 0.0


# --------------------------
# VAD & diarization
# --------------------------

def _run_webrtc_vad_segments(wav_path: str, frame_ms: int = 30, aggressiveness: int = 2) -> List[Dict[str, float]]:
    """
    Run webrtcvad on a wav file and return a list of voiced segments: [{'start': s, 'end': e}, ...]
    This function requires webrtcvad package.
    """
    if not _webrtcvad:
        raise RuntimeError("webrtcvad not installed")

    import webrtcvad
    # read raw frames (16-bit mono)
    with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
        rate = wf.getframerate()
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        frames = wf.readframes(wf.getnframes())
    vad = webrtcvad.Vad(aggressiveness)

    # frame size and iteration
    frame_size = int(rate * (frame_ms / 1000.0) * 2)  # bytes
    voiced_times: List[Tuple[float, float]] = []
    offset = 0
    is_speech = False
    seg_start = None
    total_bytes = len(frames)
    while offset + frame_size <= total_bytes:
        frame = frames[offset:offset + frame_size]
        timestamp = offset / (rate * 2.0)
        try:
            speech = vad.is_speech(frame, rate)
        except Exception:
            speech = False
        if speech and not is_speech:
            seg_start = timestamp
            is_speech = True
        elif not speech and is_speech:
            seg_end = timestamp
            if seg_start is not None:
                voiced_times.append((seg_start, seg_end))
            is_speech = False
            seg_start = None
        offset += frame_size

    # close last segment
    if is_speech and seg_start is not None:
        # estimate end
        voiced_times.append((seg_start, min(timestamp + (frame_ms / 1000.0), _wav_duration(wav_path))))

    segments = [{"start": float(s), "end": float(e)} for (s, e) in voiced_times]
    return segments


def _simple_energy_vad(wav_path: str, window_s: float = 0.5, hop_s: float = 0.25, energy_thresh: float = 0.0005) -> List[Dict[str, float]]:
    """
    Simple energy based VAD fallback using librosa if present, or wave+numpy.
    Returns voiced segments list.
    """
    y = None
    sr = None
    if _librosa:
        try:
            y, sr = _librosa.load(wav_path, sr=None, mono=True)
        except Exception:
            y = None

    if y is None:
        # fallback: read raw wave using wave module and numpy if available
        try:
            with wave.open(wav_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sampwidth = wf.getsampwidth()
                import struct
                if sampwidth == 2:
                    fmt = "<{}h".format(wf.getnframes() * wf.getnchannels())
                    samples = struct.unpack(fmt, frames)
                    import numpy as np
                    y = np.array(samples, dtype=float)
                    sr = wf.getframerate()
                else:
                    # give up
                    return []
        except Exception:
            return []

    import numpy as np  # local import to be safe
    win = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    energy = []
    for i in range(0, max(1, len(y) - win + 1), hop):
        frame = y[i:i + win]
        energy.append(np.mean(frame.astype(float) ** 2))
    voiced = []
    t = 0.0
    i = 0
    in_voiced = False
    start_t = 0.0
    for e in energy:
        if e > energy_thresh and not in_voiced:
            in_voiced = True
            start_t = i * hop / sr
        elif e <= energy_thresh and in_voiced:
            end_t = i * hop / sr
            voiced.append({"start": float(start_t), "end": float(end_t)})
            in_voiced = False
        i += 1
    if in_voiced:
        voiced.append({"start": float(start_t), "end": _wav_duration(wav_path)})
    return voiced

def _run_vad(audio_wav: str, cfg: dict) -> List[Dict[str, float]]:
    """
    Run VAD with fallback and logging.
    """
    try:
        if _webrtcvad:
            segments = _run_webrtc_vad_segments(audio_wav)
            if segments:
                return segments
    except Exception as e:
        log.warning("webrtcvad failed: %s", e)

    try:
        segments = _simple_energy_vad(
            audio_wav,
            energy_thresh=cfg.get("audio_pipe", {}).get("energy_threshold", 5e-6)
        )
        if segments:
            return segments
    except Exception as e:
        log.warning("energy VAD failed: %s", e)

    log.warning("VAD produced no segments; using full audio fallback")
    return [{"start": 0.0, "end": _wav_duration(audio_wav)}]

# --------------------------
# Diarization (pyannote fallback)
# --------------------------
def _diarize_audio(audio_wav: str, cfg: dict) -> List[Dict[str, Any]]:
    """
    Attempt pyannote diarization. If not available or fails, fallback to simple segmentation
    (each VAD segment is assigned a generic speaker 'spk0' or alternating speakers).
    Returns list of segments: {'start': float, 'end': float, 'speaker': 'spk0'}
    """
    # try pyannote.audio pipeline
    if _pyannote:
        try:
            from pyannote.audio import Pipeline
            # Try to get token from environment or config
            hf_token = (
                os.environ.get("HF_TOKEN") or
                cfg.get("huggingface_token") or
                cfg.get("pyannote_token")
            )
            if not hf_token:
                log.warning("No Hugging Face token found for pyannote diarization. Set HF_TOKEN env or add to config.")
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
            diar = pipeline(audio_wav)
            segments = []
            for turn, _, speaker in diar.itertracks(yield_label=True):
                segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
            if segments:
                return segments
        except Exception as e:
            log.warning("pyannote diarization failed or not available: %s", e)

    # Fallback: use VAD segments and assign speakers in alternating fashion (best-effort)
    vad_segments = _run_vad(audio_wav, cfg)
    segments = []
    speaker_idx = 0
    for seg in vad_segments:
        speaker = f"spk{speaker_idx % 2}"
        segments.append({"start": float(seg["start"]), "end": float(seg["end"]), "speaker": speaker})
        speaker_idx += 1
    return segments


# --------------------------
# ASR helpers (Whisper or HF fallback)
# --------------------------
def _transcribe_segment_whisper(audio_path: str, model_name: str = "small") -> Tuple[str, float]:
    """
    Use whisper (if installed) to transcribe a single segment; returns (transcript, confidence_estimate)
    """
    if not _whisper:
        raise RuntimeError("whisper package not installed")
    try:
        model = _whisper.load_model(model_name)
        result = model.transcribe(audio_path)
        text = result.get("text", "")
        conf = float(result.get("avg_logprob", 0.0)) if "avg_logprob" in result else 0.0
        return text, float(conf)
    except Exception as e:
        log.warning("whisper transcribe failed: %s", e)
        return "", 0.0


def _transcribe_segment_hf(audio_path: str, model_name: str = "facebook/wav2vec2-base-960h") -> Tuple[str, float]:
    """
    Use transformers pipeline for ASR as a fallback if whisper isn't available.
    Returns (text, confidence_estimate).
    """
    if not _transformers:
        raise RuntimeError("transformers not installed")
    try:
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model=model_name, chunk_length_s=30)
        res = asr(audio_path)
        # pipeline output may have 'text' and optionally 'score' or 'chunks'
        text = res.get("text", "") if isinstance(res, dict) else str(res)
        conf = 0.0
        if isinstance(res, dict):
            if "score" in res and res["score"] is not None:
                try:
                    conf = float(res["score"])
                except Exception:
                    conf = 0.0
            elif "chunks" in res and isinstance(res["chunks"], list) and res["chunks"]:
                scores = [c.get("score", 0.0) for c in res["chunks"] if isinstance(c, dict)]
                if scores:
                    conf = float(sum(scores) / len(scores))
        return text, conf
    except Exception as e:
        log.warning("HF ASR failed: %s", e)
        return "", 0.0


def _transcribe_segment_with_backends(audio_wav: str, start: float, end: float, cfg: dict) -> Tuple[str, float]:
    """
    Cut the segment to a temp wav and run the configured ASR backend (whisper/hf).
    Returns (transcript, confidence).
    """
    # create a temp wav file for the segment
    tmpf = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmpf = tf.name
        _cut_audio_segment(audio_wav, tmpf, start, end)
        backend = cfg.get("text_pipe", {}).get("asr_backend", "whisper")
        model_name = cfg.get("text_pipe", {}).get("asr_model", "small" if backend == "whisper" else "facebook/wav2vec2-base-960h")
        if backend == "whisper" and _whisper:
            return _transcribe_segment_whisper(tmpf, model_name)
        elif backend == "hf" and _transformers:
            return _transcribe_segment_hf(tmpf, model_name)
        else:
            # if whisper requested but not available, fallback to hf if available
            if _transformers:
                return _transcribe_segment_hf(tmpf, model_name)
            else:
                return "", 0.0
    except Exception as e:
        log.warning("Segment ASR failed: %s", e)
        return "", 0.0
    finally:
        try:
            if tmpf and Path(tmpf).exists():
                Path(tmpf).unlink()
        except Exception:
            pass


# --------------------------
# ASR over multiple segments (post-fill)
# --------------------------
def _postfill_missing_transcripts(
    audio_wav: str,
    segments: List[Dict[str, Any]],
    transcripts: Dict[Tuple[float, float], Tuple[str, float, str]],
    cfg: dict
) -> Dict[Tuple[float, float], Tuple[str, float, str]]:
    """
    Fill missing transcripts using ASR if enabled.
    """
    if not cfg.get("text_pipe", {}).get("asr_enabled", False):
        return transcripts

    for seg in segments:
        key = (float(seg["start"]), float(seg["end"]))
        text, conf, status = transcripts.get(key, ("", 0.0, "missing"))

        if status == "missing":
            try:
                t, c = _transcribe_segment_with_backends(
                    audio_wav, seg["start"], seg["end"], cfg
                )
                transcripts[key] = (
                    t,
                    float(c),
                    "ok" if t.strip() else "missing"
                )
            except Exception as e:
                log.warning("Postfill ASR failed for %s: %s", key, e)
                transcripts[key] = ("", 0.0, "failed")

    return transcripts

# --------------------------
# Face tracking (simple)
# --------------------------
def _track_faces_simple(video_path: str, cfg: dict) -> List[Dict[str, Any]]:
    """
    Track presence of a single (largest) face across the video using Haar cascade.
    Returns list of {'id': 'face0', 'start': s, 'end': e}.
    """
    tracks: List[Dict[str, Any]] = []
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        current_face = None

        # Haar cascade path lookup (reuse same logic as video.py)
        haar_path = None
        # Prefer config-provided path
        if cfg.get("video_pipe", {}).get("openface", {}).get("haar_path"):
            haar_path = cfg["video_pipe"]["openface"]["haar_path"]
        else:
            # try relative to openface binary
            binp = cfg.get("video_pipe", {}).get("openface", {}).get("binary_path")
            if binp:
                p = Path(binp).resolve()
                cand = p.parent / "classifiers" / "haarcascade_frontalface_alt.xml"
                if cand.exists():
                    haar_path = str(cand)
                else:
                    cand2 = p.parent.parent / "classifiers" / "haarcascade_frontalface_alt.xml"
                    if cand2.exists():
                        haar_path = str(cand2)
        # fallback to OpenCV
        try:
            if not haar_path:
                import cv2 as _cv
                candidate = _cv.data.haarcascades + "haarcascade_frontalface_default.xml"
                if Path(candidate).exists():
                    haar_path = str(candidate)
        except Exception:
            pass

        face_cascade = None
        if haar_path:
            face_cascade = cv2.CascadeClassifier(haar_path)
            if face_cascade.empty():
                face_cascade = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            ts = frame_idx / (fps or 1.0)

            detections = []
            try:
                if face_cascade is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    dets = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
                    for (x, y, w, h) in dets:
                        detections.append((int(x), int(y), int(w), int(h)))
            except Exception:
                detections = []

            if detections:
                if current_face is None:
                    current_face = {"id": "face0", "start": ts, "last": ts}
                else:
                    current_face["last"] = ts

        cap.release()
        if current_face is not None:
            tracks.append({"id": current_face["id"], "start": float(current_face["start"]), "end": float(current_face["last"])})
        return tracks

    except Exception as e:
        log.warning("Haar face tracking failed: %s", e)
        return []


# --------------------------
# Audio/Video clip extraction + encryption
# --------------------------
def _write_clip_and_encrypt_audio(store: SecureStore, audio_wav: str, start: float, end: float,
                                  session_id: str, work_dir: Path, cfg: dict) -> Optional[str]:
    try:
        clip_rel = f"{session_id}/clips/audio/{int(start*1000)}_{int(end*1000)}.wav"
        temp_clip = work_dir / "clips" / f"clip_{int(start*1000)}_{int(end*1000)}.wav"
        temp_clip.parent.mkdir(parents=True, exist_ok=True)
        _cut_audio_segment(audio_wav, str(temp_clip), start, end)
        with open(temp_clip, "rb") as f:
            payload = f.read()
        # compute uri and write via store
        uri = f"file://{store.root / clip_rel}"
        store.encrypt_write(uri, payload)
        return uri
    except Exception as e:
        log.warning("Failed to write/encrypt audio clip: %s", e)
        return None


def _write_clip_and_encrypt_video(store: SecureStore, video_path: str, start: float, end: float,
                                  session_id: str, work_dir: Path, cfg: dict) -> Optional[str]:
    try:
        clip_rel = f"{session_id}/clips/video/{int(start*1000)}_{int(end*1000)}.mp4"
        temp_clip = work_dir / "clips" / f"vclip_{int(start*1000)}_{int(end*1000)}.mp4"
        temp_clip.parent.mkdir(parents=True, exist_ok=True)
        _cut_video_segment(video_path, str(temp_clip), start, end)
        with open(temp_clip, "rb") as f:
            payload = f.read()
        uri = f"file://{store.root / clip_rel}"
        store.encrypt_write(uri, payload)
        return uri
    except Exception as e:
        log.warning("Failed to cut/encrypt video clip: %s", e)
        return None


# --------------------------
# Feature extraction (simple audio + placeholders)
# --------------------------
def _extract_features_for_segment(
    audio_wav: Optional[str],
    video_path: Optional[str],
    start: float,
    end: float,
    cfg: dict
) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    status: Dict[str, str] = {}

    feats["duration"] = max(0.0, end - start)

    if audio_wav and _librosa:
        try:
            y, sr = _librosa.load(audio_wav, sr=None, mono=True)
            seg = y[int(start*sr):int(end*sr)]
            feats["rms"] = float(_librosa.feature.rms(y=seg).mean()) if seg.size else 0.0
            status["audio_features"] = "ok"
        except Exception as e:
            status["audio_features"] = f"failed: {type(e).__name__}"

    feats["video_features_extracted"] = False
    if video_path and cfg.get("video_pipe", {}).get("openface", {}).get("enabled", False):
        try:
            feats["video_features_extracted"] = True
            status["video_features"] = "ok"
        except Exception as e:
            status["video_features"] = f"failed: {type(e).__name__}"

    feats["_status"] = status
    return feats

# --------------------------
# QA assembly
# --------------------------
def _assemble_qa_pairs(rows: List[Dict[str, Any]], cfg: dict) -> List[Dict[str, Any]]:
    if not rows:
        return rows

    merged = []
    prev = rows[0].copy()

    for r in rows[1:]:
        if r.get("speaker_label") == prev.get("speaker_label"):
            prev["end_time"] = r["end_time"]
            prev["transcript"] = (prev.get("transcript", "") + " " + r.get("transcript", "")).strip()
        else:
            merged.append(prev)
            prev = r.copy()

    merged.append(prev)

    pairs = []
    pair_idx = 0
    i = 0

    while i < len(merged):
        cur = merged[i]
        cur.setdefault("derived", {})

        if i + 1 < len(merged) and merged[i + 1]["speaker_label"] != cur["speaker_label"]:
            nxt = merged[i + 1]
            nxt.setdefault("derived", {})
            pair_id = f"{cur['session_id']}.pair.{pair_idx:04d}"

            cur["derived"].update({"pair_id": pair_id, "turn_type": "question"})
            nxt["derived"].update({"pair_id": pair_id, "turn_type": "response"})

            pairs.extend([cur, nxt])
            pair_idx += 1
            i += 2
        else:
            cur["derived"].update({"pair_id": None, "turn_type": "utterance"})
            pairs.append(cur)
            i += 1

    return pairs

# --------------------------
# Main pipeline entrypoint
# --------------------------
def process_session_file(session_id: str, cfg: dict, work_dir: Path,
                         video_path: Optional[str], audio_path: Optional[str],
                         text_input: Optional[str], mode: str,
                         roles: Optional[Dict[str, str]] = None
                         ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """
    Process a session according to mode: "session" | "continuous" | "text".
    Returns (rows, artifacts, receipts).

    rows: list of dicts (schema as discussed earlier)
    artifacts: dict of artifact labels -> URIs
    receipts: list of encrypted receipt URIs (strings)
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Any] = {}
    receipts: List[str] = []
    rows: List[Dict[str, Any]] = []

    # create secure store
    store = SecureStore(
        agent="lda-session-processor",
        root=Path(cfg["storage"]["root"]).resolve()
    )
    # centralized receipt manager
    receipt_mgr = CentralReceiptManager(agent="lda-session-processor")

    # If mode == text and only text_input provided
    if mode == "text":
        # single row storing text with timestamp 0
        row = {
            "session_id": session_id,
            "modality": "text",
            "segment_id": f"{session_id}.seg.0000",
            "start_time": 0.0,
            "end_time": 0.0,
            "speaker_label": None,
            "role": roles.get("patient") if roles else "patient",
            "transcript": text_input or "",
            "transcript_confidence": None,
            "audio_uri": None,
            "video_uri": None,
            "features": {},
            "derived": {},
            "receipt_path": None
        }
        # create and store receipt for text ingest
        receipt = receipt_mgr.create_receipt(
            agent="lda-session-processor",
            session_id=session_id,
            operation="text_ingest",
            params={"text_len": len(text_input or "")},
            outputs=[]
        )
        rrel = f"{session_id}/receipts/{datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')}_text.json.enc"
        receipt_uri = f"file://{store.root / rrel}"
        store.encrypt_write(receipt_uri, json.dumps(receipt).encode())
        row["receipt_path"] = receipt_uri
        receipts.append(receipt_uri)
        rows.append(row)
        return rows, artifacts, receipts

    # Ensure audio exists (extract from video if necessary)
    if video_path and not audio_path:
        try:
            audio_path = _extract_audio_from_video(video_path, str(work_dir / "raw_audio.wav"), sr=cfg.get("ingest", {}).get("audio", {}).get("sr", 16000))
            artifacts["raw_audio"] = audio_path
        except Exception as e:
            log.warning("Audio extraction from video failed: %s", e)
            audio_path = None

    # For session / continuous modes:
    if not audio_path:
        raise RuntimeError("No audio available for session/continuous processing")

    # 1) VAD -> segments
    vad_segments = _run_vad(audio_path, cfg)
    if not vad_segments:
        # If no VAD segments found, create a single segment covering entire file
        dur = _wav_duration(audio_path)
        vad_segments = [{"start": 0.0, "end": dur}]

    # 2) Diarization -> speaker segments
    diarization = _diarize_audio(audio_path, cfg)

    # 3) Transcription per segment (best-effort with whisper if available; otherwise blank)
    transcripts = _transcribe_segments(audio_path, diarization if diarization else vad_segments, cfg)

    # 3b) Post-fill missing transcripts using configured ASR backend (whisper or hf)
    transcripts = _postfill_missing_transcripts(audio_path, diarization if diarization else vad_segments, transcripts, cfg)

    # 4) Face tracking if video available
    face_tracks = []
    if video_path:
        try:
            face_tracks = _track_faces_simple(video_path, cfg)
            artifacts["face_tracks"] = face_tracks
        except Exception as e:
            log.warning("Face tracking failed: %s", e)
            face_tracks = []

    # 5) Map speakers to faces (best-effort); simple placeholder: if roles provided, use them
    speaker_to_role = {}
    if roles:
        speaker_to_role = roles.copy()
    else:
        # attempt mapping via simple heuristic: spk0 -> patient, spk1 -> counsellor (user may override)
        unique_speakers = list({seg.get("speaker") for seg in diarization if seg.get("speaker")}) if diarization else []
        unique_speakers_sorted = sorted(unique_speakers)
        if unique_speakers_sorted:
            # assign patient to spk1 and counsellor to spk0 by heuristic (user can override)
            if len(unique_speakers_sorted) >= 2:
                speaker_to_role[unique_speakers_sorted[0]] = "counsellor"
                speaker_to_role[unique_speakers_sorted[1]] = "patient"
            else:
                speaker_to_role[unique_speakers_sorted[0]] = "patient"

    # 6) Build per-segment rows (use diarization if present else VAD)
    segments_to_iterate = diarization if diarization else vad_segments
    seg_counter = 0

    for seg in segments_to_iterate:
        seg_counter += 1
        start = float(seg["start"])
        end = float(seg["end"])
        speaker = seg.get("speaker", "spk0")

        transcript, conf, t_status = transcripts.get(
            (start, end), ("", 0.0, "missing")
        )

        role = speaker_to_role.get(speaker, "unknown")

        features = _extract_features_for_segment(
            audio_path, video_path, start, end, cfg
        )

        row = {
            "session_id": session_id,
            "modality": "combined" if video_path else "audio",
            "segment_id": f"{session_id}.seg.{seg_counter:04d}",
            "start_time": start,
            "end_time": end,
            "speaker_label": speaker,
            "role": role,
            "transcript": transcript,
            "transcript_confidence": conf,
            "audio_uri": None,
            "video_uri": None,
            "features": features,
            "derived": {
                "transcript_status": t_status,
                "role_assignment": {
                    "role": role,
                    "confidence": "heuristic" if not roles else "explicit"
                }
            },
            "receipt_path": None
        }

        rows.append(row)


        # create receipt per segment and store it encrypted
        try:
            receipt = receipt_mgr.create_receipt(
                agent="lda-session-processor",
                session_id=session_id,
                operation="segment_process",
                params={"start": start, "end": end, "speaker": speaker, "role": role},
                outputs=[]
            )
            rrel = f"{session_id}/receipts/{datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')}_{seg_counter}.json.enc"
            receipt_uri = f"file://{store.root / rrel}"
            store.encrypt_write(receipt_uri, json.dumps(receipt).encode())
            row["receipt_path"] = receipt_uri
            receipts.append(receipt_uri)
        except Exception as e:
            log.warning("Failed to create/store segment receipt: %s", e)
            # leave receipt_path as None for this row

    # 7) If mode == session, assemble QA pairs
    if mode == "session":
        rows = sorted(rows, key=lambda r: r["start_time"])
        rows = _assemble_qa_pairs(rows, cfg)

    # 8) Optionally create a small manifest artifact (unencrypted list of row metadata is OK to be encrypted by caller)
    artifacts["rows_count"] = len(rows)

    return rows, artifacts, receipts


# --------------------------
# Backwards-compatible transcription helper (kept for session_processor compatibility)
# --------------------------
def _transcribe_segments(
    audio_wav: str,
    segments: List[Dict[str, Any]],
    cfg: dict
) -> Dict[Tuple[float, float], Tuple[str, float, str]]:
    """
    Transcribe each segment and return:
    (transcript, confidence, status)
    status ∈ {"ok", "missing", "failed"}
    """
    transcripts: Dict[Tuple[float, float], Tuple[str, float, str]] = {}
    backend = cfg.get("text_pipe", {}).get("asr_backend", "whisper")
    model_name = cfg.get("text_pipe", {}).get("asr_model", "small")

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        key = (start, end)

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name
            _cut_audio_segment(audio_wav, tmp_path, start, end)

            if backend == "whisper" and _whisper:
                text, conf = _transcribe_segment_whisper(tmp_path, model_name)
            elif backend == "hf" and _transformers:
                text, conf = _transcribe_segment_hf(tmp_path, model_name)
            else:
                text, conf = "", 0.0

            status = "ok" if text.strip() else "missing"
            transcripts[key] = (text, float(conf), status)

        except Exception as e:
            log.warning("ASR failed for segment %.2f–%.2f: %s", start, end, e)
            transcripts[key] = ("", 0.0, "failed")

        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    return transcripts
