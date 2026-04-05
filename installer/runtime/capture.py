"""
capture.py — Capture audio/video from device for a fixed duration.

Returns the session directory containing:
  session_TIMESTAMP/
    video.mp4   (if camera available)
    audio.wav   (always attempted)
"""

import os
import time
import wave
import struct
import platform
import logging
import threading
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

IS_WINDOWS = platform.system().lower() == "windows"

BASE     = Path.home() / ".federated"
DATA_DIR = BASE / "data" / "input"


# ── helpers ───────────────────────────────────────────────────────────────────
def _popen_kw() -> dict:
    kw: dict = {}
    if IS_WINDOWS:
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kw


def _ffmpeg_available() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


# ── Audio capture (ffmpeg → wav) ──────────────────────────────────────────────
def _capture_audio_ffmpeg(out_path: Path, duration_s: int) -> bool:
    """Capture `duration_s` seconds of audio to out_path using ffmpeg."""
    if not _ffmpeg_available():
        log.warning("ffmpeg not found — skipping audio capture")
        return False

    if IS_WINDOWS:
        # DirectShow audio device
        audio_input = ["-f", "dshow", "-i", "audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\\wave_{00000000-0000-0000-0000-000000000000}"]
    else:
        # ALSA default
        audio_input = ["-f", "alsa", "-i", "default"]

    cmd = [
        "ffmpeg", "-y",
        *audio_input,
        "-t", str(duration_s),
        "-ac", "1", "-ar", "16000",
        "-hide_banner", "-loglevel", "error",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, timeout=duration_s + 15, **_popen_kw(),
                              capture_output=True)
        if proc.returncode != 0:
            log.warning("Audio capture failed: %s", proc.stderr.decode(errors="replace"))
            return False
        return out_path.exists()
    except subprocess.TimeoutExpired:
        log.warning("Audio capture timed out")
        return False
    except Exception as e:
        log.warning("Audio capture error: %s", e)
        return False


# ── Video capture (ffmpeg → mp4) ─────────────────────────────────────────────
def _capture_video_ffmpeg(out_path: Path, duration_s: int) -> bool:
    """Capture `duration_s` seconds of video (+audio) to out_path using ffmpeg."""
    if not _ffmpeg_available():
        return False

    if IS_WINDOWS:
        video_input = ["-f", "dshow", "-i", "video=Integrated Camera"]
    else:
        video_input = ["-f", "v4l2", "-i", "/dev/video0"]

    cmd = [
        "ffmpeg", "-y",
        *video_input,
        "-t", str(duration_s),
        "-vcodec", "libx264", "-preset", "ultrafast",
        "-acodec", "aac",
        "-hide_banner", "-loglevel", "error",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, timeout=duration_s + 30, **_popen_kw(),
                              capture_output=True)
        if proc.returncode != 0:
            log.debug("Video capture failed (device may be busy): %s",
                      proc.stderr.decode(errors="replace"))
            return False
        return out_path.exists()
    except Exception as e:
        log.debug("Video capture error: %s", e)
        return False


# ── Silence wav generator (fallback) ─────────────────────────────────────────
def _write_silence_wav(out_path: Path, duration_s: int, sr: int = 16000):
    """Write a silent WAV file as a placeholder when capture fails."""
    num_samples = sr * duration_s
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))


# ── Public API ────────────────────────────────────────────────────────────────
def capture_session(duration_s: int = 300) -> Path:
    """
    Capture audio (and video if available) for `duration_s` seconds.
    Returns path to session directory containing the captured files.
    Guarantees at least an audio.wav file exists (silence if capture fails).
    """
    ts = int(time.time())
    session_dir = DATA_DIR / f"session_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)

    audio_path = session_dir / "audio.wav"
    video_path = session_dir / "video.mp4"

    log.info("Capturing %ds session → %s", duration_s, session_dir)

    # Try video first (it includes audio)
    has_video = _capture_video_ffmpeg(video_path, duration_s)
    if has_video:
        log.info("Video captured: %s", video_path)
        # Extract clean audio track from video
        if _ffmpeg_available():
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(video_path),
                    "-ac", "1", "-ar", "16000", "-vn",
                    "-hide_banner", "-loglevel", "error",
                    str(audio_path),
                ], timeout=60, **_popen_kw(), capture_output=True)
            except Exception:
                pass

    # If no audio from video, try audio-only capture
    if not audio_path.exists():
        has_audio = _capture_audio_ffmpeg(audio_path, duration_s)
        if not has_audio:
            log.warning("Audio capture failed — writing silence placeholder")
            _write_silence_wav(audio_path, min(duration_s, 10))  # short silence

    log.info("Session ready: audio=%s, video=%s",
             audio_path.exists(), video_path.exists())
    return session_dir