#!/usr/bin/env python3
import sys
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent.parent
VENV_PYTHON = BASE / "venv" / "Scripts" / "python.exe"

# 🚨 Re-exec inside venv if not already
if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    print("[DEBUG] Switching to venv Python")
    subprocess.run([str(VENV_PYTHON), __file__, *sys.argv[1:]])
    sys.exit(0)

# ✅ Ensure imports work
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

print("[DEBUG] USING PYTHON:", sys.executable)
print("[DEBUG] PYTHONPATH =", sys.path[:3])

# ✅ FIX IMPORT PATH FOR installer.security
SECURITY_PATH = BASE / "installer"
if str(SECURITY_PATH) not in sys.path:
    sys.path.insert(0, str(SECURITY_PATH))

import hashlib
import sys
import os
from runtime.grpc.orchestrator_pb2 import CSR
from runtime.runtime_guard import runtime_guard
from runtime.grpc_client import create_grpc_stub
from runtime.pipeline import run_pipeline
from runtime.tpm_guard import get_device_pubkey
from runtime.daemon import daemon_loop
import time

def main():
    mode = "daemon"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    # 1. Security gate
    master_secret = runtime_guard()

    # 2. TPM-backed identity
    device_pubkey = get_device_pubkey()
    device_id = hashlib.sha256(device_pubkey).digest()

    # 3. Secure channel
    SERVER_ADDR = os.environ.get("FED_SERVER")

    if not SERVER_ADDR:
        SERVER_ADDR = input("Enter server address (host:port): ").strip()

    stub = create_grpc_stub(SERVER_ADDR)

    # 4. Idempotent registration
    try:
        stub.RegisterDevice(
            CSR(device_pubkey=device_pubkey),
            timeout=10
        )
    except Exception:
        pass

    if mode == "run-once":
        run_pipeline(stub, device_id, master_secret)
        return

    if mode == "daemon":
        daemon_loop(stub, device_id, master_secret)
        return

    print("Unknown mode:", mode)


if __name__ == "__main__":
    main()
