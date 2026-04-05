"""
daemon.py — Continuous background federated learning daemon

Flow:
  1. Wait for system idle
  2. Capture audio/video for CAPTURE_WINDOW_S seconds
  3. Run full pipeline (LDA → Trainer → DP → Enc → Upload)
  4. Sleep UPLOAD_INTERVAL_S
  5. Repeat

The daemon writes a heartbeat file so the integrity watcher knows it's alive.
"""

import os
import time
import signal
import logging
import platform
import threading
from pathlib import Path
from typing import Optional

from runtime.idle import wait_until_idle
from runtime.pipeline import run_pipeline
from runtime.capture import capture_session

log = logging.getLogger(__name__)

IS_WINDOWS = platform.system().lower() == "windows"

BASE = Path.home() / ".federated"
HEARTBEAT_FILE = BASE / "state" / "daemon.heartbeat"
LOCK_FILE      = BASE / "state" / "runtime.lock"

UPLOAD_INTERVAL_S  = 60 * 60    # 1 hour between uploads
CAPTURE_WINDOW_S   = 300         # 5 minutes of capture per session
HEARTBEAT_INTERVAL = 30          # write heartbeat every 30s


# ── Heartbeat writer ──────────────────────────────────────────────────────────
class _HeartbeatThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="heartbeat")
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
                HEARTBEAT_FILE.write_text(str(time.time()))
            except Exception:
                pass
            self._stop.wait(HEARTBEAT_INTERVAL)

    def stop(self):
        self._stop.set()


# ── Graceful shutdown ─────────────────────────────────────────────────────────
_shutdown_event = threading.Event()

def _signal_handler(sig, frame):
    log.info("Daemon received signal %s — shutting down gracefully", sig)
    _shutdown_event.set()


def _register_signals():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT,  _signal_handler)


# ── Lock helpers ──────────────────────────────────────────────────────────────
def _acquire_lock() -> bool:
    try:
        if LOCK_FILE.exists():
            # Check if the PID in the lock file is still running
            try:
                pid = int(LOCK_FILE.read_text().strip())
                if IS_WINDOWS:
                    import ctypes
                    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                    if handle:
                        ctypes.windll.kernel32.CloseHandle(handle)
                        log.warning("Another daemon instance (PID %d) is running", pid)
                        return False
                else:
                    os.kill(pid, 0)  # raises OSError if process is dead
                    log.warning("Another daemon instance (PID %d) is running", pid)
                    return False
            except (ValueError, OSError, ProcessLookupError):
                log.info("Stale lock file found, removing")
                LOCK_FILE.unlink(missing_ok=True)

        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCK_FILE.write_text(str(os.getpid()))
        return True
    except Exception as e:
        log.error("Failed to acquire daemon lock: %s", e)
        return False


def _release_lock():
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ── Main daemon loop ──────────────────────────────────────────────────────────
def daemon_loop(stub, device_id: bytes, master_secret: bytes):
    """
    Runs indefinitely until SIGTERM/SIGINT or _shutdown_event is set.
    """
    _register_signals()

    if not _acquire_lock():
        log.error("Daemon already running — exiting")
        return

    hb = _HeartbeatThread()
    hb.start()

    log.info("Federated daemon started (PID %d)", os.getpid())
    log.info("  capture window : %ds", CAPTURE_WINDOW_S)
    log.info("  upload interval: %ds", UPLOAD_INTERVAL_S)

    try:
        while not _shutdown_event.is_set():
            # ── 1. Wait for idle ──────────────────────────────────────────────
            log.info("Waiting for system idle...")
            wait_until_idle(max_wait_seconds=UPLOAD_INTERVAL_S)

            if _shutdown_event.is_set():
                break

            # ── 2. Capture session data ───────────────────────────────────────
            session_dir: Optional[Path] = None
            try:
                log.info("Starting %ds capture session", CAPTURE_WINDOW_S)
                session_dir = capture_session(duration_s=CAPTURE_WINDOW_S)
                log.info("Capture complete: %s", session_dir)
            except Exception as e:
                log.error("Capture failed: %s — skipping this cycle", e)
                _shutdown_event.wait(60)
                continue

            # ── 3. Run pipeline ───────────────────────────────────────────────
            try:
                log.info("Running federated pipeline...")
                run_pipeline(stub, device_id, master_secret,
                             session_dir=session_dir)
                log.info("Pipeline complete ✓")
            except Exception as e:
                log.error("Pipeline failed: %s", e)

            # ── 4. Sleep until next cycle ─────────────────────────────────────
            log.info("Sleeping %ds until next cycle...", UPLOAD_INTERVAL_S)
            _shutdown_event.wait(timeout=UPLOAD_INTERVAL_S)

    finally:
        hb.stop()
        _release_lock()
        log.info("Daemon stopped cleanly")