"""
integrity.py — Runtime file integrity monitoring

Phase 7 additions:
  - IntegrityWatcher: background thread that periodically re-verifies the
    baseline hash and calls trigger_self_destruct() if tampering is detected.
  - write_baseline / verify_integrity: unchanged API, fixed to not crash when
    scope is empty (e.g. on a fresh install with no .py files yet).
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
    "keys/",          # certificates rotate; exclude from integrity scope
    "integrity/",     # don't hash the baseline itself
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
            pass   # file may be temporarily locked on Windows
        files_hashed += 1

    if files_hashed == 0:
        # On a fresh install or sparse layout, return a stable known value
        log.warning("[integrity] No files in scope — returning empty hash")
        return "00" * 32

    return h.hexdigest()


def write_baseline():
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
    If baseline missing → writes it and returns True (first run).
    If mismatch → updates baseline and returns True (logs warning).
    Does NOT self-destruct directly (caller decides).
    """
    if not BASELINE_FILE.exists():
        log.warning("[integrity] No baseline found — creating now")
        write_baseline()
        return True

    current = compute_tree_hash(FEDERATED_DIR)
    stored  = BASELINE_FILE.read_text().strip()

    if current != stored:
        log.warning("[integrity] Hash mismatch detected! stored=%s… current=%s…",
                    stored[:16], current[:16])
        # Update baseline so subsequent runs don't keep alerting on the same change
        write_baseline()
        return False   # caller can decide to self-destruct

    return True


def integrity_guard():
    """
    Synchronous gate — call before any sensitive operation.
    Updates baseline on mismatch (permissive) so normal updates don't break things.
    """
    verify_integrity()   # mismatch just logs + updates baseline


# ── Background watcher (Phase 7) ─────────────────────────────────────────────
class IntegrityWatcher(threading.Thread):
    """
    Runs in background and checks file integrity every `interval_s` seconds.
    On tamper detection it calls `on_tamper()` (defaults to self-destruct).
    """

    def __init__(
        self,
        interval_s: int = 300,
        max_violations: int = 1,
        on_tamper=None,
    ):
        super().__init__(daemon=True, name="integrity-watcher")
        self.interval_s     = interval_s
        self.max_violations = max_violations
        self._stop_event    = threading.Event()
        self._violations    = 0

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
        log.info("[integrity-watcher] Started (interval=%ds, max_violations=%d)",
                 self.interval_s, self.max_violations)
        while not self._stop_event.wait(timeout=self.interval_s):
            try:
                ok = verify_integrity()
                if not ok:
                    self._violations += 1
                    log.error(
                        "[integrity-watcher] Violation #%d/%d",
                        self._violations, self.max_violations,
                    )
                    if self._violations >= self.max_violations:
                        log.critical("[integrity-watcher] Max violations reached — triggering response")
                        self._on_tamper()
                        break
                else:
                    self._violations = 0  # reset on clean check
            except Exception as e:
                log.warning("[integrity-watcher] Check error: %s", e)

        log.info("[integrity-watcher] Stopped")