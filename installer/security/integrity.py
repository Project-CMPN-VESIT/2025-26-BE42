"""
integrity.py — Runtime file integrity monitoring

SECURITY FIXES APPLIED:
  FIX-INTEGRITY-1: verify_integrity() NO LONGER updates the baseline on mismatch.
                   Previously, a tamper event would silently update the baseline,
                   meaning an attacker only needed to wait one check cycle.
                   Now mismatch returns False and the caller MUST self-destruct.

  FIX-INTEGRITY-2: integrity_guard() now calls trigger_self_destruct() on
                   mismatch instead of silently ignoring the return value.

  FIX-INTEGRITY-3: IntegrityWatcher.run() calls _on_tamper() on the FIRST
                   violation by default (max_violations=1) and never resets
                   the violation counter after a detection.
"""

import hashlib
import os
import threading
import time
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

FEDERATED_DIR  = Path.home() / ".federated"
BASELINE_FILE  = FEDERATED_DIR / "integrity" / "baseline.sha256"

EXCLUDE_PREFIXES = {
    "logs/",
    "data/",
    "cache/",
    "venv/",
    "deps/",
    "tpm/",
    "secrets/",
    "state/",
    "runtime/tmp/",
    "runtime/cache/",
    "runtime/__pycache__/",
    "agents/__pycache__/",
    "__pycache__/",
    "keys/",
    "integrity/",     # baseline file itself excluded
}

INTEGRITY_SCOPE = [
    "bin/",
    "runtime/",
    "agents/",
    "core/",
]


def _should_include(path: Path) -> bool:
    rel = path.relative_to(FEDERATED_DIR).as_posix()
    return any(rel.startswith(p) for p in INTEGRITY_SCOPE)


def _should_exclude(path: Path) -> bool:
    rel = path.relative_to(FEDERATED_DIR).as_posix()
    if any(rel.startswith(e) for e in EXCLUDE_PREFIXES):
        return True
    if rel.endswith(".pyc") or "__pycache__" in rel:
        return True
    return False


def compute_tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    files_hashed = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _should_exclude(path):
            continue
        if not _should_include(path):
            continue
        if path.suffix not in (".py", ".pem", ".toml", ".yaml", ".json"):
            continue

        rel = str(path.relative_to(root)).replace("\\", "/").lower().encode()
        h.update(rel)
        try:
            h.update(path.read_bytes())
        except Exception:
            pass
        files_hashed += 1

    if files_hashed == 0:
        log.warning("[integrity] No files in scope — returning empty hash")
        return "00" * 32

    return h.hexdigest()


def write_baseline():
    """Write baseline. Called ONLY on first install, never on mismatch."""
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    digest = compute_tree_hash(FEDERATED_DIR)
    BASELINE_FILE.write_text(digest)
    try:
        os.chmod(BASELINE_FILE, 0o600)
    except Exception:
        pass
    log.info("[integrity] Baseline written: %s…", digest[:16])


def verify_integrity() -> bool:
    """
    Returns True if integrity check passes.

    SECURITY: If baseline is missing → writes it and returns True (first run only).
    If mismatch → logs CRITICAL and returns False WITHOUT updating the baseline.
    The caller is responsible for calling trigger_self_destruct() on False.

    Previously this function silently updated the baseline on mismatch, which
    allowed any tampered file to become the new trusted baseline after one cycle.
    This is now fixed — the baseline is NEVER updated after mismatch.
    """
    if not BASELINE_FILE.exists():
        log.warning("[integrity] No baseline found — creating now (first run)")
        write_baseline()
        return True

    current = compute_tree_hash(FEDERATED_DIR)
    stored  = BASELINE_FILE.read_text().strip()

    if current != stored:
        log.critical(
            "[integrity] TAMPER DETECTED — stored=%s… current=%s…",
            stored[:16], current[:16],
        )
        # DO NOT write_baseline() here. The caller must self-destruct.
        return False

    return True


def integrity_guard():
    """
    Synchronous gate — call before any sensitive operation.

    SECURITY FIX: Previously discarded the return value of verify_integrity(),
    meaning a tamper event was logged but execution continued. Now any mismatch
    immediately triggers self-destruct.
    """
    ok = verify_integrity()
    if not ok:
        from .self_destruct import trigger_self_destruct
        trigger_self_destruct("integrity_guard: file tampering detected — aborting")


# ── Background watcher ────────────────────────────────────────────────────────
class IntegrityWatcher(threading.Thread):
    """
    Runs in background and checks file integrity every `interval_s` seconds.

    SECURITY FIXES:
      - max_violations default is now 1 (zero tolerance)
      - Violation counter is never reset after detection (was resetting on
        clean check, allowing an attacker to modify files, wait for one clean
        check cycle, then get the violation count reset)
      - _on_tamper() is called on FIRST violation when max_violations=1
    """

    def __init__(
        self,
        interval_s: int = 300,
        max_violations: int = 1,   # Changed from 2 to 1 — zero tolerance
        on_tamper=None,
    ):
        super().__init__(daemon=True, name="integrity-watcher")
        self.interval_s     = interval_s
        self.max_violations = max_violations
        self._stop_event    = threading.Event()
        self._violations    = 0
        self._tamper_triggered = False  # prevent double-trigger

        if on_tamper is not None:
            self._on_tamper = on_tamper
        else:
            self._on_tamper = self._default_tamper_handler

    @staticmethod
    def _default_tamper_handler():
        from .self_destruct import trigger_self_destruct
        trigger_self_destruct("Integrity violation detected by background watcher")

    def stop(self):
        self._stop_event.set()

    def run(self):
        log.info(
            "[integrity-watcher] Started (interval=%ds, max_violations=%d)",
            self.interval_s, self.max_violations,
        )
        while not self._stop_event.wait(timeout=self.interval_s):
            if self._tamper_triggered:
                break
            try:
                ok = verify_integrity()
                if not ok:
                    self._violations += 1
                    log.critical(
                        "[integrity-watcher] Violation #%d/%d",
                        self._violations, self.max_violations,
                    )
                    if self._violations >= self.max_violations:
                        log.critical(
                            "[integrity-watcher] Max violations reached — triggering response"
                        )
                        self._tamper_triggered = True
                        self._on_tamper()
                        break
                # SECURITY FIX: Do NOT reset self._violations on a clean check.
                # Previously: self._violations = 0  ← removed
                # An attacker could: modify file → wait one check → clean check
                # resets counter → repeat indefinitely without ever hitting max.
            except Exception as e:
                log.warning("[integrity-watcher] Check error: %s", e)

        log.info("[integrity-watcher] Stopped")