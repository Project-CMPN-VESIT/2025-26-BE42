#!/usr/bin/env python3

import hashlib
import sys

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
    SERVER_ADDR = "server-address:50051"  # replaced by installer
    stub = create_grpc_stub(SERVER_ADDR)

    # 4. Idempotent registration
    try:
        stub.RegisterDevice(
            stub._request_serializer.CSR(device_pubkey=device_pubkey)
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
