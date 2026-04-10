#!/usr/bin/env python3
"""
federated_client.py — Main entry point.

FIXES IN THIS VERSION:
  FIX-A: Removed str(BASE / "runtime") from sys.path.
          Adding ~/.federated/runtime to sys.path caused Python to find
          the local ~/.federated/runtime/grpc/ directory when doing
          `import grpc`, shadowing the real grpcio package and producing:
              AttributeError: module 'grpc' has no attribute 'ssl_channel_credentials'
          All runtime.* imports already work correctly with only BASE on
          sys.path because `runtime` is a package (has __init__.py) inside BASE.

  FIX-B: Removed str(BASE / "core") from sys.path for the same reason —
          it is not needed since `from core.X import Y` resolves correctly
          via BASE being on sys.path (core is a package inside BASE).

  FIX-C: Platform-aware VENV_PYTHON path (Windows vs Linux).

  FIX-D: Creates __init__.py files for runtime, installer, runtime/grpc
          packages on first run to ensure imports resolve.
"""

import sys
import platform
from pathlib import Path

IS_WINDOWS = platform.system().lower() == "windows"

# ── Locate ~/.federated ───────────────────────────────────────────────────────
# federated-client lives at ~/.federated/bin/federated-client
# parent = ~/.federated/bin, parent.parent = ~/.federated
BASE = Path(__file__).resolve().parent.parent   # ~/.federated

# FIX-C: platform-aware venv python path
if IS_WINDOWS:
    VENV_PYTHON = BASE / "venv" / "Scripts" / "python.exe"
else:
    VENV_PYTHON = BASE / "venv" / "bin" / "python"

# ── Redirect into venv if not already running inside it ───────────────────────
if Path(sys.executable).resolve() != VENV_PYTHON.resolve() and VENV_PYTHON.exists():
    import subprocess
    result = subprocess.run([str(VENV_PYTHON), __file__, *sys.argv[1:]])
    sys.exit(result.returncode)

# ── Build sys.path ────────────────────────────────────────────────────────────
# FIX-A/B: Only add BASE and BASE/installer.
# DO NOT add BASE/runtime — it contains a `grpc/` subdirectory that would
# shadow the real grpcio `grpc` package, breaking ssl_channel_credentials.
# DO NOT add BASE/core — same reasoning (unnecessary and risky).
# `from runtime.X import Y` works fine with BASE on the path because
# BASE/runtime/ is a Python package (has __init__.py).
_PATH_EXTRAS = [
    str(BASE),               # enables: runtime.*, agents.*, core.*
    str(BASE / "installer"), # enables: installer.security.*
    str(BASE / "bin"),
]
for _extra in _PATH_EXTRAS:
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

# ── Ensure package __init__.py files exist ────────────────────────────────────
# FIX-D: Without these, Python treats the directories as namespace packages
# and relative/absolute imports inside them may fail unpredictably.

def _ensure_init(pkg_path: Path):
    """Write an empty __init__.py if the directory exists and lacks one."""
    if pkg_path.is_dir():
        init = pkg_path / "__init__.py"
        if not init.exists():
            try:
                init.write_text("")
                init.chmod(0o600)
            except Exception:
                pass

_ensure_init(BASE / "runtime")
_ensure_init(BASE / "runtime" / "grpc")
_ensure_init(BASE / "installer")
_ensure_init(BASE / "installer" / "security")
_ensure_init(BASE / "core")
_ensure_init(BASE / "agents")
_ensure_init(BASE / "agents" / "lda")
_ensure_init(BASE / "agents" / "lda" / "pipelines")
_ensure_init(BASE / "agents" / "trainer")
_ensure_init(BASE / "agents" / "dp")
_ensure_init(BASE / "agents" / "enc")

# ── Phase 11: logging FIRST ───────────────────────────────────────────────────
from runtime.logging_config import setup_logging, MetricsCollector, HealthReporter
setup_logging(level="INFO")

import hashlib
import os
import time
import logging
from typing import Optional

from runtime.grpc.orchestrator_pb2 import CSR
from runtime.runtime_guard import runtime_guard
from runtime.grpc_client import create_grpc_stub, call_with_retry
from runtime.pipeline import run_pipeline
from runtime.tpm_guard import get_device_pubkey
from runtime.daemon import daemon_loop
from installer.security.integrity import IntegrityWatcher

log = logging.getLogger("federated_client")

_LOCK = BASE / "state" / "runtime.lock"

metrics = MetricsCollector()
health  = HealthReporter(metrics=metrics)


def _cleanup_lock():
    try:
        _LOCK.unlink(missing_ok=True)
    except Exception:
        pass


def main():
    mode = "daemon"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lstrip("-")

    log.info("Federated client starting (mode=%s)", mode)

    # ── Phase 7: start integrity watcher ─────────────────────────────────────
    watcher = IntegrityWatcher(interval_s=300, max_violations=2)
    watcher.start()
    log.info("Integrity watcher started")

    try:
        # ── Runtime security gate ─────────────────────────────────────────────
        master_secret = runtime_guard()
        device_pubkey = get_device_pubkey()
        if not device_pubkey:
            log.error("Failed to obtain device public key — is TPM initialised?")
            health.unhealthy("no device pubkey")
            sys.exit(1)

        device_id = hashlib.sha256(device_pubkey).digest()

        # ── Server address ────────────────────────────────────────────────────
        SERVER_ADDR = os.environ.get("FED_SERVER")
        if not SERVER_ADDR:
            SERVER_ADDR = input("Enter server address (host:port): ").strip()

        # ── Phase 3: dual-channel gRPC ────────────────────────────────────────
        stub = create_grpc_stub(SERVER_ADDR)

        # Register device (best-effort; may already be registered)
        try:
            call_with_retry(stub.RegisterDevice, CSR(device_pubkey=device_pubkey), timeout=10)
        except Exception as e:
            log.debug("RegisterDevice skipped: %s", e)

        health.healthy(server=SERVER_ADDR)

        # ── Dispatch mode ─────────────────────────────────────────────────────
        if mode in ("run-once", "run_once"):
            metrics.record_attempt()
            t0 = time.time()
            try:
                run_pipeline(stub, device_id, master_secret)
                metrics.record_success(time.time() - t0)
                health.healthy(last_run="success")
                log.info("Run-once pipeline complete ✓")
            except Exception as e:
                metrics.record_failure(str(e))
                health.degraded(str(e))
                log.error("Run-once pipeline failed: %s", e)
                raise

        elif mode == "daemon":
            log.info("Starting daemon loop...")
            health.healthy(daemon="running")
            daemon_loop(stub, device_id, master_secret)

        else:
            log.error("Unknown mode: %s (use: daemon | run-once)", mode)
            sys.exit(1)

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.exception("Fatal error: %s", e)
        health.unhealthy(str(e))
        sys.exit(1)
    finally:
        watcher.stop()
        _cleanup_lock()
        metrics.log_snapshot()
        log.info("Client exiting")


if __name__ == "__main__":
    main()
