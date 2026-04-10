#!/usr/bin/env python3
"""
federated_client.py — Main entry point.  FIXED VERSION.

Key fixes (every line reviewed):
  FIX-1: VENV_PYTHON was hard-coded to Windows path
          (venv/Scripts/python.exe).  Now platform-aware.
  FIX-2: sys.path setup now also adds BASE/runtime explicitly so that
          `from runtime.grpc_client import ...` works even when this
          file is invoked directly (e.g. `python federated-client`).
  FIX-3: Added BASE/core to sys.path so `from core.centralized_secure_store
          import SecureStore` resolves in the pipeline import chain.
  FIX-4: runtime/__init__.py is created on first run if missing, ensuring
          relative imports inside runtime_guard.py (`from .tpm_guard import`)
          resolve correctly when runtime is treated as a package.
  FIX-5: installer/__init__.py created on first run if missing, so that
          `from installer.security.integrity import integrity_guard` (used
          by dp_agent, enc_agent, trainer, lda/main) doesn't raise
          ModuleNotFoundError.
"""

import sys
import platform
from pathlib import Path

IS_WINDOWS = platform.system().lower() == "windows"

# ── Locate ~/.federated regardless of where this script sits ─────────────────
# federated-client is installed at ~/.federated/bin/federated-client
# so parent = ~/.federated/bin, parent.parent = ~/.federated
BASE = Path(__file__).resolve().parent.parent   # ~/.federated

# FIX-1: platform-aware venv python path
if IS_WINDOWS:
    VENV_PYTHON = BASE / "venv" / "Scripts" / "python.exe"
else:
    VENV_PYTHON = BASE / "venv" / "bin" / "python"

# ── Redirect into venv if we're not already running inside it ─────────────────
if Path(sys.executable).resolve() != VENV_PYTHON.resolve() and VENV_PYTHON.exists():
    import subprocess
    result = subprocess.run([str(VENV_PYTHON), __file__, *sys.argv[1:]])
    sys.exit(result.returncode)

# ── Build sys.path before any project imports ─────────────────────────────────
# FIX-2/3: added BASE/runtime and BASE/core
_PATH_EXTRAS = [
    str(BASE),             # enables: runtime.*, agents.*, core.*
    str(BASE / "runtime"), # enables: direct runtime module lookups
    str(BASE / "core"),    # enables: core.centralized_secure_store etc.
    str(BASE / "installer"), # enables: security.* (integrity_guard etc.)
    str(BASE / "bin"),
]
for _extra in _PATH_EXTRAS:
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

# ── FIX-4: ensure runtime/__init__.py exists so relative imports work ─────────
_runtime_init = BASE / "runtime" / "__init__.py"
if not _runtime_init.exists() and (BASE / "runtime").exists():
    try:
        _runtime_init.write_text("")
        _runtime_init.chmod(0o600)
    except Exception:
        pass

# ── FIX-5: ensure installer/__init__.py exists ────────────────────────────────
_installer_init = BASE / "installer" / "__init__.py"
if not _installer_init.exists() and (BASE / "installer").exists():
    try:
        _installer_init.write_text("")
        _installer_init.chmod(0o600)
    except Exception:
        pass

# ── FIX-5b: ensure grpc sub-package __init__.py exists ───────────────────────
_grpc_init = BASE / "runtime" / "grpc" / "__init__.py"
if not _grpc_init.exists() and (BASE / "runtime" / "grpc").exists():
    try:
        _grpc_init.write_text("")
        _grpc_init.chmod(0o600)
    except Exception:
        pass

# ── Phase 11: logging FIRST ──────────────────────────────────────────────────
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
from security.integrity import IntegrityWatcher

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

    # ── Phase 7: start integrity watcher ────────────────────────────────────
    watcher = IntegrityWatcher(interval_s=300, max_violations=2)
    watcher.start()
    log.info("Integrity watcher started")

    try:
        # ── Runtime security gate ────────────────────────────────────────────
        master_secret = runtime_guard()
        device_pubkey = get_device_pubkey()
        if not device_pubkey:
            log.error("Failed to obtain device public key — is TPM initialised?")
            health.unhealthy("no device pubkey")
            sys.exit(1)

        device_id = hashlib.sha256(device_pubkey).digest()

        # ── Server address ───────────────────────────────────────────────────
        SERVER_ADDR = os.environ.get("FED_SERVER")
        if not SERVER_ADDR:
            SERVER_ADDR = input("Enter server address (host:port): ").strip()

        # ── Phase 3: dual-channel gRPC ───────────────────────────────────────
        stub = create_grpc_stub(SERVER_ADDR)

        # Register device (best-effort; may already be registered)
        try:
            call_with_retry(stub.RegisterDevice, CSR(device_pubkey=device_pubkey), timeout=10)
        except Exception as e:
            log.debug("RegisterDevice skipped: %s", e)

        health.healthy(server=SERVER_ADDR)

        # ── Dispatch mode ────────────────────────────────────────────────────
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