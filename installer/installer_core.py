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
from security.deps_windows import verify_windows_deps
from security.deps_windows import verify_python_and_pip
from fs.install_python_deps import install_python_deps
from fs.install_openface import install_openface
from fs.install_opensmile import install_opensmile


from runtime.grpc.orchestrator_pb2_grpc import OrchestratorStub
from runtime.grpc.orchestrator_pb2 import EnrollRequest

IS_WINDOWS = platform.system().lower() == "windows"

BASE_DIR = Path.home() / ".federated"
STATE_FILE = BASE_DIR / "state" / "install_state.json"
KEYS_DIR = BASE_DIR / "keys"

SERVER_ADDR = "42.111.108.31:50051"
INSTALLER_OTP = None

def write_install_state():
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "installed": True,
        "version": "1.0",
        "platform": platform.system(),
    }, indent=2))
    STATE_FILE.chmod(0o600)


def otp_enrollment(device_pubkey: bytes):
    global INSTALLER_OTP

    if INSTALLER_OTP:
        token = INSTALLER_OTP
    else:
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

    verify_windows_deps()

    if IS_WINDOWS:
        from security.windows_runtime import check_vc_runtime
        check_vc_runtime()

    print("[8] Install runtime payload")
    install_runtime()

    print("[STEP 11] Verifying Python")
    verify_python_and_pip()

    print("[STEP 12] Installing Python packages")
    install_python_deps()

    print("[STEP 13] Installing OpenFace")
    install_openface()

    print("[STEP 14] Installing openSMILE")
    install_opensmile()

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
