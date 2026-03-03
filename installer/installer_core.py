#!/usr/bin/env python3

import sys
import json
import grpc
import platform
from pathlib import Path
import subprocess

# -------------------------
# Filesystem & runtime
# -------------------------
from fs.secure_layout import create_secure_layout
from fs.install_runtime import install_runtime
from fs.install_python_deps import install_python_deps
from fs.install_openface import install_openface
from fs.install_opensmile import install_opensmile

# -------------------------
# Security
# -------------------------
from security.anti_debug import anti_debug
from security.integrity import write_baseline
from security.tpm_attestation import (
    provision_tpm_identity,
    get_device_pubkey_installer_safe,
)
from security.tpm_seal import seal_master_secret
from security.deps_windows import (
    verify_windows_deps,
    verify_python_and_pip,
)

# -------------------------
# gRPC
# -------------------------
from runtime.grpc.orchestrator_pb2_grpc import OrchestratorStub
from runtime.grpc.orchestrator_pb2 import EnrollRequest

# -------------------------
# Constants
# -------------------------
IS_WINDOWS = platform.system().lower() == "windows"

BASE_DIR = Path.home() / ".federated"
STATE_FILE = BASE_DIR / "state" / "install_state.json"
KEYS_DIR = BASE_DIR / "keys"

INSTALLER_OTP = None
INSTALLER_SERVER_ADDR = None


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def write_install_state():
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(
            {
                "installed": True,
                "version": "1.0",
                "platform": platform.system(),
            },
            indent=2,
        )
    )
    STATE_FILE.chmod(0o600)


def otp_enrollment(device_pubkey: bytes, token: str, server_addr: str):
    print("[DEBUG] Entered otp_enrollment()")
    print("[DEBUG] KEYS_DIR =", KEYS_DIR)
    global INSTALLER_OTP

    import subprocess

    if len(token) < 6:
        sys.exit("[SECURITY] Invalid OTP")

    KEYS_DIR.mkdir(parents=True, exist_ok=True)

    client_key = KEYS_DIR / "client.key"
    client_csr = KEYS_DIR / "client.csr"

    # 1. Generate private key
    subprocess.run([
        "openssl", "genrsa",
        "-out", str(client_key),
        "2048"
    ], check=True)

    # 2. Generate CSR
    subprocess.run([
        "openssl", "req",
        "-new",
        "-key", str(client_key),
        "-out", str(client_csr),
        "-subj", "/CN=federated-device"
    ], check=True)

    SERVER_ADDR = server_addr

    # 3. Create secure channel (server TLS only)
    channel = grpc.secure_channel(
        SERVER_ADDR,
        grpc.ssl_channel_credentials(
            root_certificates=(KEYS_DIR / "ca.pem").read_bytes()
        ),
        options=(
            ('grpc.ssl_target_name_override', 'localhost'),
        )
    )

    stub = OrchestratorStub(channel)

    # 4. Send enrollment request with CSR
    resp = stub.EnrollDevice(
        EnrollRequest(
            enrollment_token=token,
            device_pubkey=device_pubkey,
            csr=client_csr.read_bytes(),
        )
    )

    if not resp.ok:
        sys.exit("[SECURITY] Enrollment failed")

    # 5. Store signed certificate
    client_cert_path = KEYS_DIR / "client.pem"
    client_cert_path.write_bytes(resp.client_cert)
    client_cert_path.chmod(0o600)

    print("[OK] Device enrolled + client certificate installed")


# --------------------------------------------------
# Main installer
# --------------------------------------------------
def main():
    print("=== BUILD VERSION 2 WITH GUI INPUT FIX ===")
    # --------------------------------------------------
    # 1. Anti-debug (installer mode)
    # --------------------------------------------------
    print("[1] Anti-debug (installer mode)")
    anti_debug(strict=True, installer_mode=True)

    # --------------------------------------------------
    # 2. Secure filesystem layout
    # --------------------------------------------------
    print("[2] Secure filesystem layout")
    create_secure_layout()

    # --------------------------------------------------
    # 3. TPM identity (safe for installer)
    # --------------------------------------------------
    print("[3] TPM identity provisioning")
    provision_tpm_identity()
    device_pubkey = get_device_pubkey_installer_safe()

    # --------------------------------------------------
    # 4. Runtime payload (code + configs)
    # --------------------------------------------------
    print("[4] Installing runtime payload")
    install_runtime()

    # --------------------------------------------------
    # 5. Windows runtime prerequisites
    # --------------------------------------------------
    if IS_WINDOWS:
        print("[5] Verifying Python & VC runtime")
        from security.windows_runtime import check_vc_runtime

        check_vc_runtime()
        verify_python_and_pip()

    # --------------------------------------------------
    # 6. Python dependencies
    # --------------------------------------------------
    print("[6] Installing Python dependencies")
    try:
        install_python_deps()
        print("[DEBUG] Python deps installed")
    except Exception as e:
        print("[ERROR] install_python_deps crashed:", e)
        raise

    # --------------------------------------------------
    # 7. Native ML dependencies
    # --------------------------------------------------
    if not IS_WINDOWS:
        print("[7] Installing OpenFace")
        install_openface()
    else:
        print("[7] Windows OpenFace already bundled")

    if not IS_WINDOWS:
        print("[8] Installing openSMILE")
        install_opensmile()
    else:
        print("[8] Windows opensmile already bundled")
        
    # --------------------------------------------------
    # 9. VERIFY dependencies (NOW they exist)
    # --------------------------------------------------
    print("[9] Verifying platform dependencies")
    verify_windows_deps()

    # --------------------------------------------------
    # 10. OTP enrollment (server is running)
    # --------------------------------------------------
    print("[10] OTP enrollment")
    otp_enrollment(device_pubkey, INSTALLER_OTP, INSTALLER_SERVER_ADDR)

    # --------------------------------------------------
    # 11. Seal master secret
    # --------------------------------------------------
    print("[11] Sealing master secret")
    seal_master_secret()

    # --------------------------------------------------
    # 12. Integrity baseline
    # --------------------------------------------------
    print("[12] Writing integrity baseline")
    write_baseline()

    # --------------------------------------------------
    # 13. Persist install state
    # --------------------------------------------------
    print("[13] Persisting install state")
    write_install_state()

    print("\n[OK] Installer finished successfully")


# --------------------------------------------------
# Entry
# --------------------------------------------------
if __name__ == "__main__":
    main()
