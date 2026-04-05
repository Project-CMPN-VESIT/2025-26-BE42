"""
logging_config.py — Phase 11 structured logging + metrics + health

Usage (in federated_client.py):
    from runtime.logging_config import setup_logging, MetricsCollector, HealthReporter
    setup_logging()
    metrics = MetricsCollector()
    health  = HealthReporter()
"""

import json
import logging
import logging.handlers
import os
import platform
import sys
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

BASE     = Path.home() / ".federated"
LOG_DIR  = BASE / "logs"
LOG_FILE = LOG_DIR / "federated_client.log"

LOG_MAX_BYTES  = 10 * 1024 * 1024   # 10 MB per file
LOG_BACKUP_CNT = 5                   # keep 5 rotated files


# ── Structured JSON log formatter ────────────────────────────────────────────
class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
            "pid":     record.process,
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger:
    - Rotating file handler (JSON lines) at ~/.federated/logs/
    - StreamHandler (human-readable) to stderr
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # File handler — JSON lines
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_CNT,
        encoding="utf-8",
    )
    fh.setFormatter(_JSONFormatter())
    fh.setLevel(logging.DEBUG)

    # Console handler — human readable
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))

    root.addHandler(fh)
    root.addHandler(ch)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.WARNING)

    logging.info("Logging initialised (level=%s, file=%s)", level, LOG_FILE)


# ── Metrics collector ─────────────────────────────────────────────────────────
@dataclass
class _PipelineMetrics:
    rounds_attempted:  int = 0
    rounds_succeeded:  int = 0
    rounds_failed:     int = 0
    total_latency_s:   float = 0.0
    last_success_ts:   Optional[str] = None
    last_failure_msg:  Optional[str] = None


class MetricsCollector:
    """Thread-safe pipeline metrics."""

    def __init__(self):
        self._m   = _PipelineMetrics()
        self._lock = threading.Lock()

    def record_attempt(self):
        with self._lock:
            self._m.rounds_attempted += 1

    def record_success(self, latency_s: float):
        with self._lock:
            self._m.rounds_succeeded += 1
            self._m.total_latency_s  += latency_s
            self._m.last_success_ts   = datetime.now(timezone.utc).isoformat()

    def record_failure(self, msg: str):
        with self._lock:
            self._m.rounds_failed    += 1
            self._m.last_failure_msg  = msg

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            d = asdict(self._m)
            n = self._m.rounds_succeeded
            d["avg_latency_s"] = (self._m.total_latency_s / n) if n else 0.0
            d["success_rate"]  = (n / self._m.rounds_attempted) if self._m.rounds_attempted else 0.0
        return d

    def log_snapshot(self):
        snap = self.snapshot()
        logging.getLogger("metrics").info(
            "Pipeline metrics: %s", json.dumps(snap, indent=2)
        )


# ── Health reporter ───────────────────────────────────────────────────────────
HEALTH_FILE = BASE / "state" / "health.json"

class HealthReporter:
    """Writes a JSON health file that can be read by external monitoring."""

    def __init__(self, metrics: Optional[MetricsCollector] = None):
        self._metrics = metrics
        self._start_ts = datetime.now(timezone.utc).isoformat()

    def _write(self, status: str, extra: Dict[str, Any] = None):
        doc = {
            "status":    status,
            "ts":        datetime.now(timezone.utc).isoformat(),
            "started":   self._start_ts,
            "pid":       os.getpid(),
            "platform":  platform.system(),
            "python":    sys.version.split()[0],
        }
        if self._metrics:
            doc["metrics"] = self._metrics.snapshot()
        if extra:
            doc.update(extra)
        HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        HEALTH_FILE.write_text(json.dumps(doc, indent=2))

    def healthy(self, **extra):
        self._write("healthy", extra)

    def degraded(self, reason: str, **extra):
        self._write("degraded", {"reason": reason, **extra})

    def unhealthy(self, reason: str, **extra):
        self._write("unhealthy", {"reason": reason, **extra})