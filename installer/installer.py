#!/usr/bin/env python3

import sys
import json
import grpc
import platform
from pathlib import Path

from fs.secure_layout import create_secure_layout
from fs.install_runtime import install_runtime

from security.anti_debug import anti_debug
from security.integrity import write_baseline
from security.tpm_attestation import provision_tpm_identity, get_device_pubkey
from security.tpm_seal import seal_master_secret

from runtime.grpc.orchestrator_pb2_grpc import OrchestratorStub
from runtime.grpc.orchestrator_pb2 import EnrollRequest

IS_WINDOWS = platform.system().lower() == "windows"

BASE_DIR = Path.home() / ".federated"
STATE_FILE = BASE_DIR / "state" / "install_state.json"
KEYS_DIR = BASE_DIR / "keys"

SERVER_ADDR = "SERVER_IP:50051"


def write_install_state():
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "installed": True,
        "version": "1.0",
        "platform": platform.system(),
    }, indent=2))
    STATE_FILE.chmod(0o600)


def otp_enrollment(device_pubkey: bytes):
    token = input("Enter enrollment OTP: ").strip()
    if len(token) < 6:
        sys.exit("[SECURITY] Invalid OTP")

    channel = grpc.secure_channel(
        SERVER_ADDR,
        grpc.ssl_channel_credentials(
            root_certificates=(KEYS_DIR / "ca.pem").read_bytes()
        )
    )

    stub = OrchestratorStub(channel)

    resp = stub.EnrollDevice(
        EnrollRequest(
            enrollment_token=token,
            device_pubkey=device_pubkey,
        )
    )

    if not resp.ok:
        sys.exit("[SECURITY] Enrollment failed")

    print("[OK] Device enrolled")


def main():
    print("[1] Anti-debug")
    anti_debug(strict=True, installer_mode=True)

    print("[2] TPM presence")
    provision_tpm_identity()

    print("[3] Secure filesystem")
    create_secure_layout()

    print("[4] TPM identity")
    device_pubkey = get_device_pubkey()

    print("[5] OTP enrollment")
    otp_enrollment(device_pubkey)

    print("[6] TPM sealed master secret")
    seal_master_secret()

    print("[7] Integrity baseline")
    write_baseline()

    print("[8] Install runtime payload")
    install_runtime()

    print("[9] Persist install state")
    write_install_state()

    print("[OK] Installation completed successfully")


def remove_installer():
    try:
        Path(__file__).unlink()
    except Exception:
        pass


if __name__ == "__main__":
    main()
    remove_installer()
