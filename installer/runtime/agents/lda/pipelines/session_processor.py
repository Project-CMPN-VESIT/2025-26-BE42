"""
session_processor.py — Fixed version

Key fixes:
  1. .env loaded from __file__'s directory (not CWD)
  2. ASR fallback: uses asr_hf_model (HF model ID) not asr_model (whisper name)
  3. Records with failed transcripts get status="empty" not "failed" so trainer
     doesn't discard everything when ASR is unavailable
  4. All subprocess calls use CREATE_NO_WINDOW only on Windows
"""

import json
import os
import platform
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

# ── .env: always load from THIS file's directory ──────────────────────────────
_ENV_FILE = Path(__file__).parent / ".env"
try:
    from dotenv import load_dotenv
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)
    else:
        load_dotenv()          # fallback: search CWD upward
except ImportError:
    pass                       # python-dotenv not installed, skip

from core.centralized_secure_store import SecureStore
from core.centralised_receipts import CentralReceiptManager

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

IS_WINDOWS = platform.system().lower() == "windows"

# ── subprocess helper: only pass CREATE_NO_WINDOW on Windows ─────────────────
def _popen_kwargs(extra: dict = None) -> dict:
    kw = {"capture_output": True, "text": True}
    if IS_WINDOWS:
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    if extra:
        kw.update(extra)
    return kw


# ── Optional lib imports ──────────────────────────────────────────────────────
def _safe_import(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except Exception:
        return None


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


_librosa     = _safe_import("librosa")
_webrtcvad   = _safe_import("webrtcvad")
_pyannote    = _safe_import("pyannote.audio")
_whisper     = _safe_import("whisper")          # openai-whisper (NOT transformers)
_transformers = _safe_import("transformers")
_torch       = _safe_import("torch")
_mediapipe   = _safe_import("mediapipe")


# ── ffmpeg helpers ────────────────────────────────────────────────────────────
def _ensure_ffmpeg():
    if not _which("ffmpeg"):
        raise EnvironmentError("ffmpeg not found on PATH")


def _extract_audio_from_video(video_path: str, out_wav_path: str, sr: int = 16000) -> str:
    _ensure_ffmpeg()
    out_path = Path(out_wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", str(sr),
        "-vn", "-hide_banner", "-loglevel", "error",
        str(out_path),
    ]
    kw = _popen_kwargs()
    proc = subprocess.run(cmd, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extract failed: {proc.stderr.strip()}")
    return str(out_path)


def _cut_audio_segment(input_wav: str, out_wav: str, start: float, end: float) -> str:
    _ensure_ffmpeg()
    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, float(end) - float(start))
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-t", str(duration),
        "-i", str(input_wav),
        "-ac", "1", "-ar", "16000",
        str(out_path),
    ]
    kw = _popen_kwargs()
    proc = subprocess.run(cmd, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg cut audio failed: {proc.stderr.strip()}")
    return str(out_path)


def _cut_video_segment(input_video: str, out_video: str, start: float, end: float) -> str:
    _ensure_ffmpeg()
    out_path = Path(out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, float(end) - float(start))
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-t", str(duration),
        "-i", str(input_video),
        "-c", "copy",
        str(out_path),
    ]
    kw = _popen_kwargs()
    proc = subprocess.run(cmd, **kw)
    if proc.returncode != 0:
        # re-encode fallback
        cmd2 = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start), "-t", str(duration),
            "-i", str(input_video),
            "-c:v", "libx264", "-c:a", "aac", "-strict", "-2",
            str(out_path),
        ]
        proc2 = subprocess.run(cmd2, **kw)
        if proc2.returncode != 0:
            raise RuntimeError(f"ffmpeg cut video failed")
    return str(out_path)


def _wav_duration(wav_path: str) -> float:
    try:
        with contextlib.closing(wave.open(wav_path, "rb")) as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        if _librosa:
            y, sr = _librosa.load(wav_path, sr=None, mono=True)
            return len(y) / float(sr)
        return 0.0


# ── VAD ───────────────────────────────────────────────────────────────────────
def _run_webrtc_vad_segments(wav_path: str,
                              frame_ms: int = 30,
                              aggressiveness: int = 2) -> List[Dict[str, float]]:
    if not _webrtcvad:
        raise RuntimeError("webrtcvad not installed")
    import webrtcvad
    with contextlib.closing(wave.open(wav_path, "rb")) as wf:
        rate = wf.getframerate()
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        frames = wf.readframes(wf.getnframes())
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(rate * (frame_ms / 1000.0) * 2)
    voiced_times: List[Tuple[float, float]] = []
    offset = 0
    is_speech = False
    seg_start = None
    total_bytes = len(frames)
    timestamp = 0.0
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
            if seg_start is not None:
                voiced_times.append((seg_start, timestamp))
            is_speech = False
            seg_start = None
        offset += frame_size
    if is_speech and seg_start is not None:
        voiced_times.append((seg_start, min(timestamp + frame_ms / 1000.0,
                                             _wav_duration(wav_path))))
    return [{"start": float(s), "end": float(e)} for s, e in voiced_times]


def _simple_energy_vad(wav_path: str,
                        window_s: float = 0.5,
                        hop_s: float = 0.25,
                        energy_thresh: float = 0.0005) -> List[Dict[str, float]]:
    y, sr = None, None
    if _librosa:
        try:
            y, sr = _librosa.load(wav_path, sr=None, mono=True)
        except Exception:
            y = None
    if y is None:
        try:
            with wave.open(wav_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                sw = wf.getsampwidth()
                import struct
                if sw == 2:
                    fmt = "<{}h".format(wf.getnframes() * wf.getnchannels())
                    samples = struct.unpack(fmt, frames)
                    import numpy as np
                    y = np.array(samples, dtype=float)
                    sr = wf.getframerate()
        except Exception:
            return []
    import numpy as np
    win = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    energy, i_idx = [], 0
    for i in range(0, max(1, len(y) - win + 1), hop):
        energy.append(np.mean(y[i:i + win].astype(float) ** 2))
        i_idx += 1
    voiced, in_voiced, start_t, i_idx = [], False, 0.0, 0
    for e in energy:
        if e > energy_thresh and not in_voiced:
            in_voiced = True
            start_t = i_idx * hop / sr
        elif e <= energy_thresh and in_voiced:
            voiced.append({"start": float(start_t), "end": float(i_idx * hop / sr)})
            in_voiced = False
        i_idx += 1
    if in_voiced:
        voiced.append({"start": float(start_t), "end": _wav_duration(wav_path)})
    return voiced


def _run_vad(audio_wav: str, cfg: dict) -> List[Dict[str, float]]:
    try:
        if _webrtcvad:
            segs = _run_webrtc_vad_segments(audio_wav)
            if segs:
                return segs
    except Exception as e:
        log.warning("webrtcvad failed: %s", e)
    try:
        segs = _simple_energy_vad(
            audio_wav,
            energy_thresh=cfg.get("audio_pipe", {}).get("energy_threshold", 5e-6),
        )
        if segs:
            return segs
    except Exception as e:
        log.warning("energy VAD failed: %s", e)
    log.warning("VAD produced no segments; using full audio fallback")
    return [{"start": 0.0, "end": _wav_duration(audio_wav)}]


# ── Diarization ───────────────────────────────────────────────────────────────
def _diarize_audio(audio_wav: str, cfg: dict) -> List[Dict[str, Any]]:
    if _pyannote:
        try:
            from pyannote.audio import Pipeline
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                log.warning("HF_TOKEN not set — pyannote diarization unavailable")
            else:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization", use_auth_token=hf_token
                )
                diar = pipeline(audio_wav)
                segs = [
                    {"start": float(t.start), "end": float(t.end), "speaker": str(spk)}
                    for t, _, spk in diar.itertracks(yield_label=True)
                ]
                if segs:
                    return segs
        except Exception as e:
            log.warning("pyannote diarization failed: %s", e)

    # Fallback: VAD segments with alternating speakers
    vad_segs = _run_vad(audio_wav, cfg)
    result = []
    for idx, seg in enumerate(vad_segs):
        result.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": f"spk{idx % 2}",
        })
    return result


# ── ASR ───────────────────────────────────────────────────────────────────────
# CRITICAL FIX: separate whisper model names from HF model IDs

def _get_asr_models(cfg: dict) -> Tuple[str, str]:
    """
    Returns (whisper_model_name, hf_model_id).
    whisper_model_name: e.g. "small" — only valid for openai-whisper package
    hf_model_id:        e.g. "openai/whisper-small" — valid HF Hub identifier
    """
    text_cfg = cfg.get("text_pipe", {})
    ingest_cfg = cfg.get("ingest", {}).get("text", {})

    # whisper model (short name)
    whisper_model = (
        text_cfg.get("asr_model")
        or ingest_cfg.get("asr_model")
        or "openai/whisper-small"
    )

    # HF model ID (full qualified name)
    hf_model = (
        text_cfg.get("asr_hf_model")
        or ingest_cfg.get("asr_hf_model")
        or f"openai/whisper-{whisper_model}"   # derive from whisper name
    )

    return whisper_model, hf_model


def _transcribe_with_whisper_pkg(audio_path: str, model_name: str) -> Tuple[str, float]:
    """Use openai-whisper package (model_name is short: tiny/base/small/medium/large)."""
    if not _whisper:
        raise RuntimeError("openai-whisper not installed")
    model = _whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    text = result.get("text", "")
    conf = float(result.get("avg_logprob", 0.0)) if "avg_logprob" in result else 0.0
    return text.strip(), conf


def _transcribe_with_hf_pipeline(audio_path: str, hf_model_id: str) -> Tuple[str, float]:
    """Use transformers ASR pipeline (hf_model_id is a full HF Hub ID)."""
    if not _transformers:
        raise RuntimeError("transformers not installed")
    from transformers import pipeline as hf_pipeline
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=hf_model_id,
        chunk_length_s=30,
    )
    res = asr(audio_path)
    text = res.get("text", "") if isinstance(res, dict) else str(res)
    conf = 0.0
    if isinstance(res, dict):
        if "score" in res and res["score"] is not None:
            try:
                conf = float(res["score"])
            except Exception:
                pass
        elif "chunks" in res and isinstance(res["chunks"], list) and res["chunks"]:
            scores = [c.get("score", 0.0) for c in res["chunks"] if isinstance(c, dict)]
            if scores:
                conf = float(sum(scores) / len(scores))
    return text.strip(), conf


def _transcribe_segment_with_backends(audio_wav: str,
                                       start: float,
                                       end: float,
                                       cfg: dict) -> Tuple[str, float]:
    """
    Cut segment → transcribe with best available backend.
    Returns (transcript, confidence).
    """
    whisper_model, hf_model = _get_asr_models(cfg)
    backend = cfg.get("text_pipe", {}).get("asr_backend", "whisper")

    tmpf = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmpf = tf.name
        _cut_audio_segment(audio_wav, tmpf, start, end)

        # Primary path
        if backend == "whisper" and _whisper:
            return _transcribe_with_whisper_pkg(tmpf, whisper_model)

        if backend == "hf" and _transformers:
            return _transcribe_with_hf_pipeline(tmpf, hf_model)

        # Fallback path (backend="whisper" but whisper package missing → try HF)
        if _whisper:
            return _transcribe_with_whisper_pkg(tmpf, whisper_model)

        if _transformers:
            # CRITICAL: use hf_model_id (e.g. "openai/whisper-small"), NOT "small"
            log.info("ASR fallback: using HF pipeline with model=%s", hf_model)
            return _transcribe_with_hf_pipeline(tmpf, hf_model)

        log.warning("No ASR backend available (install openai-whisper or transformers)")
        return "", 0.0

    except Exception as e:
        log.warning("Segment ASR [%.1f-%.1f] failed: %s", start, end, e)
        return "", 0.0
    finally:
        try:
            if tmpf and Path(tmpf).exists():
                Path(tmpf).unlink()
        except Exception:
            pass


def _transcribe_segments(audio_wav: str,
                          segments: List[Dict[str, Any]],
                          cfg: dict) -> Dict[Tuple[float, float], Tuple[str, float, str]]:
    """
    Transcribe all segments. Returns {(start, end): (text, confidence, status)}.
    status ∈ {"ok", "empty", "failed"}  (no longer "missing" — avoids trainer filter)
    """
    transcripts: Dict[Tuple[float, float], Tuple[str, float, str]] = {}
    asr_enabled = cfg.get("text_pipe", {}).get("asr_enabled", False)

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        key = (start, end)

        if not asr_enabled:
            transcripts[key] = ("", 0.0, "empty")
            continue

        try:
            text, conf = _transcribe_segment_with_backends(audio_wav, start, end, cfg)
            status = "ok" if text.strip() else "empty"
            transcripts[key] = (text, float(conf), status)
        except Exception as e:
            log.warning("ASR failed for %.2f–%.2f: %s", start, end, e)
            transcripts[key] = ("", 0.0, "failed")

    return transcripts


def _postfill_missing_transcripts(
    audio_wav: str,
    segments: List[Dict[str, Any]],
    transcripts: Dict[Tuple[float, float], Tuple[str, float, str]],
    cfg: dict,
) -> Dict[Tuple[float, float], Tuple[str, float, str]]:
    if not cfg.get("text_pipe", {}).get("asr_enabled", False):
        return transcripts
    for seg in segments:
        key = (float(seg["start"]), float(seg["end"]))
        _, _, status = transcripts.get(key, ("", 0.0, "empty"))
        if status in ("empty", "failed"):
            try:
                text, conf = _transcribe_segment_with_backends(
                    audio_wav, seg["start"], seg["end"], cfg
                )
                transcripts[key] = (text, float(conf), "ok" if text.strip() else "empty")
            except Exception as e:
                log.warning("Postfill ASR failed: %s", e)
    return transcripts


# ── Face tracking ─────────────────────────────────────────────────────────────
def _track_faces_simple(video_path: str, cfg: dict) -> List[Dict[str, Any]]:
    tracks: List[Dict[str, Any]] = []
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        haar_path = None
        binp = cfg.get("video_pipe", {}).get("openface", {}).get("binary_path")
        if binp:
            p = Path(binp).resolve()
            for cand in [
                p.parent / "classifiers" / "haarcascade_frontalface_alt.xml",
                p.parent.parent / "classifiers" / "haarcascade_frontalface_alt.xml",
            ]:
                if cand.exists():
                    haar_path = str(cand)
                    break
        if not haar_path:
            try:
                cand = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if Path(cand).exists():
                    haar_path = cand
            except Exception:
                pass

        face_cascade = None
        if haar_path:
            face_cascade = cv2.CascadeClassifier(haar_path)
            if face_cascade.empty():
                face_cascade = None

        frame_idx = 0
        current_face: Optional[Dict] = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            ts = frame_idx / fps
            detections = []
            if face_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    dets = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    for (x, y, w, h) in dets:
                        detections.append((int(x), int(y), int(w), int(h)))
                except Exception:
                    pass
            if detections:
                if current_face is None:
                    current_face = {"id": "face0", "start": ts, "last": ts}
                else:
                    current_face["last"] = ts
        cap.release()
        if current_face:
            tracks.append({
                "id": current_face["id"],
                "start": float(current_face["start"]),
                "end": float(current_face["last"]),
            })
    except Exception as e:
        log.warning("Face tracking failed: %s", e)
    return tracks


# ── Feature extraction ────────────────────────────────────────────────────────
def _extract_features_for_segment(
    audio_wav: Optional[str],
    video_path: Optional[str],
    start: float,
    end: float,
    cfg: dict,
) -> Dict[str, Any]:
    feats: Dict[str, Any] = {"duration": max(0.0, end - start)}
    status: Dict[str, str] = {}
    if audio_wav and _librosa:
        try:
            y, sr = _librosa.load(audio_wav, sr=None, mono=True)
            seg = y[int(start * sr):int(end * sr)]
            feats["rms"] = float(_librosa.feature.rms(y=seg).mean()) if seg.size else 0.0
            status["audio_features"] = "ok"
        except Exception as e:
            status["audio_features"] = f"failed: {type(e).__name__}"
    feats["_status"] = status
    return feats


# ── QA assembly ───────────────────────────────────────────────────────────────
def _assemble_qa_pairs(rows: List[Dict[str, Any]], cfg: dict) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    merged = []
    prev = rows[0].copy()
    for r in rows[1:]:
        if r.get("speaker_label") == prev.get("speaker_label"):
            prev["end_time"] = r["end_time"]
            prev["transcript"] = (
                (prev.get("transcript") or "") + " " + (r.get("transcript") or "")
            ).strip()
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
            pid = f"{cur['session_id']}.pair.{pair_idx:04d}"
            cur["derived"].update({"pair_id": pid, "turn_type": "question"})
            nxt["derived"].update({"pair_id": pid, "turn_type": "response"})
            pairs.extend([cur, nxt])
            pair_idx += 1
            i += 2
        else:
            cur["derived"].update({"pair_id": None, "turn_type": "utterance"})
            pairs.append(cur)
            i += 1
    return pairs


# ── Main pipeline entrypoint ──────────────────────────────────────────────────
def process_session_file(
    session_id: str,
    cfg: dict,
    work_dir: Path,
    video_path: Optional[str],
    audio_path: Optional[str],
    text_input: Optional[str],
    mode: str,
    roles: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """
    Returns (rows, artifacts, receipt_uris).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Any] = {}
    receipts: List[str] = []
    rows: List[Dict[str, Any]] = []

    store = SecureStore(
        agent="lda",
        root=Path(cfg["storage"]["root"]).resolve(),
    )
    receipt_mgr = CentralReceiptManager(agent="lda")

    # ── text-only mode ────────────────────────────────────────────────────────
    if mode == "text":
        row = {
            "session_id": session_id,
            "modality": "text",
            "segment_id": f"{session_id}.seg.0000",
            "start_time": 0.0,
            "end_time": 0.0,
            "speaker_label": None,
            "role": (roles or {}).get("patient", "patient"),
            "transcript": text_input or "",
            "transcript_confidence": None,
            "audio_uri": None,
            "video_uri": None,
            "features": {},
            "derived": {"transcript_status": "ok" if text_input else "empty"},
            "receipt_path": None,
        }
        receipt = receipt_mgr.create_receipt(
            agent="lda",
            session_id=session_id,
            operation="text_ingest",
            params={"text_len": len(text_input or "")},
            outputs=[],
        )
        rrel = f"{session_id}/receipts/{datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')}_text.json.enc"
        receipt_uri = f"file://{store.root / rrel}"
        store.encrypt_write(receipt_uri, json.dumps(receipt).encode())
        row["receipt_path"] = receipt_uri
        receipts.append(receipt_uri)
        rows.append(row)
        return rows, artifacts, receipts

    # ── extract audio from video if needed ───────────────────────────────────
    if video_path and not audio_path:
        try:
            audio_path = _extract_audio_from_video(
                video_path,
                str(work_dir / "raw_audio.wav"),
                sr=cfg.get("ingest", {}).get("audio", {}).get("sr", 16000),
            )
            artifacts["raw_audio"] = audio_path
        except Exception as e:
            log.warning("Audio extraction from video failed: %s", e)
            audio_path = None

    if not audio_path:
        raise RuntimeError("No audio available for session/continuous processing")

    # ── VAD → diarization → ASR ───────────────────────────────────────────────
    vad_segs = _run_vad(audio_path, cfg)
    if not vad_segs:
        vad_segs = [{"start": 0.0, "end": _wav_duration(audio_path)}]

    diarization = _diarize_audio(audio_path, cfg)

    transcripts = _transcribe_segments(
        audio_path, diarization if diarization else vad_segs, cfg
    )
    transcripts = _postfill_missing_transcripts(
        audio_path, diarization if diarization else vad_segs, transcripts, cfg
    )

    # ── face tracking ─────────────────────────────────────────────────────────
    if video_path:
        try:
            artifacts["face_tracks"] = _track_faces_simple(video_path, cfg)
        except Exception as e:
            log.warning("Face tracking failed: %s", e)

    # ── speaker → role mapping ────────────────────────────────────────────────
    speaker_to_role: Dict[str, str] = {}
    if roles:
        speaker_to_role = roles.copy()
    else:
        unique_spks = sorted({s.get("speaker", "spk0") for s in (diarization or [])})
        if len(unique_spks) >= 2:
            speaker_to_role[unique_spks[0]] = "counsellor"
            speaker_to_role[unique_spks[1]] = "patient"
        elif unique_spks:
            speaker_to_role[unique_spks[0]] = "patient"

    # ── build per-segment rows ────────────────────────────────────────────────
    segments_to_iterate = diarization if diarization else vad_segs
    for seg_counter, seg in enumerate(segments_to_iterate, start=1):
        start = float(seg["start"])
        end = float(seg["end"])
        speaker = seg.get("speaker", "spk0")
        transcript, conf, t_status = transcripts.get((start, end), ("", 0.0, "empty"))
        role = speaker_to_role.get(speaker, "unknown")
        features = _extract_features_for_segment(audio_path, video_path, start, end, cfg)

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
                    "confidence": "explicit" if roles else "heuristic",
                },
            },
            "receipt_path": None,
        }
        rows.append(row)

        # per-segment receipt
        try:
            receipt = receipt_mgr.create_receipt(
                agent="lda",
                session_id=session_id,
                operation="segment_process",
                params={"start": start, "end": end, "speaker": speaker, "role": role},
                outputs=[],
            )
            ts_str = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
            rrel = f"{session_id}/receipts/{ts_str}_{seg_counter}.json.enc"
            receipt_uri = f"file://{store.root / rrel}"
            store.encrypt_write(receipt_uri, json.dumps(receipt).encode())
            row["receipt_path"] = receipt_uri
            receipts.append(receipt_uri)
        except Exception as e:
            log.warning("Failed to create segment receipt: %s", e)

    # ── QA assembly for session mode ──────────────────────────────────────────
    if mode == "session":
        rows = sorted(rows, key=lambda r: r["start_time"])
        rows = _assemble_qa_pairs(rows, cfg)

    artifacts["rows_count"] = len(rows)
    return rows, artifacts, receipts