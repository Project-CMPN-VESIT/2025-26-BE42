#!/usr/bin/env python3
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

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
from fs.install_ffmpeg import install_ffmpeg
from fs.install_spacy_model import install_spacy_model
from runtime.validate_deps import check

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime

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
import logging
from pathlib import Path

LOG_FILE = Path.home() / ".federated" / "logs" / "installer.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("Installer started")

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
    logging.info(f"[DEBUG] OTP received by installer: {token}")
    logging.info("[DEBUG] SERVER_ADDR =", server_addr)
    logging.info("[DEBUG] CA exists:", (KEYS_DIR / "ca.pem").exists())
    logging.info("[DEBUG] About to create gRPC channel")
    global INSTALLER_OTP

    import subprocess

    token = token.strip()

    if len(token) < 6:
        sys.exit("[SECURITY] Invalid OTP")

    KEYS_DIR.mkdir(parents=True, exist_ok=True)

    client_key = KEYS_DIR / "client.key"
    client_csr = KEYS_DIR / "client.csr"

    # Generate private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Save key
    with open(client_key, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Create CSR
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(
            x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, u"federated-device"),
            ])
        )
        .sign(key, hashes.SHA256())
    )

    csr_bytes = csr.public_bytes(serialization.Encoding.PEM)

    with open(client_csr, "wb") as f:
        f.write(csr_bytes)

    SERVER_ADDR = server_addr

    # 3. Create secure channel (server TLS only)
    creds = grpc.ssl_channel_credentials(
        root_certificates=(KEYS_DIR / "ca.pem").read_bytes()
    )

    logging.info("🔥 STEP 10: STARTING ENROLLMENT 🔥")

    host = SERVER_ADDR.split(":")[0]

    channel = grpc.secure_channel(
        SERVER_ADDR,
        creds,
        options=[
            ('grpc.ssl_target_name_override', host),
            ('grpc.default_authority', host),
        ]
    )

    logging.info("[DEBUG] Waiting for channel ready...")

    grpc.channel_ready_future(channel).result(timeout=10)

    logging.info("[DEBUG] Channel READY ✅")

    logging.info("[DEBUG] Channel READY ✅")    

    stub = OrchestratorStub(channel)

    logging.info("[DEBUG] gRPC channel created")

    # 4. Send enrollment request with CSR
    try:
        logging.info("[DEBUG] Sending EnrollDevice RPC")
        resp = stub.EnrollDevice(
            EnrollRequest(
                enrollment_token=token,
                device_pubkey=device_pubkey,
                csr=client_csr.read_bytes(),
            ),
            timeout=10
        )
        logging.info("🔥 STEP 10: ENROLLMENT COMPLETED 🔥")
    except Exception as e:
        logging.info("[ERROR] gRPC failed:", e)
        raise

    if not resp.ok:
        sys.exit("[SECURITY] Enrollment failed")

    # 5. Store signed certificate
    client_cert_path = KEYS_DIR / "client.pem"
    client_cert_path.write_bytes(resp.client_cert)
    client_cert_path.chmod(0o600)

    logging.info("[OK] Device enrolled + client certificate installed")

def create_venv():
    BASE = Path.home() / ".federated"
    VENV_DIR = BASE / "venv"

    if VENV_DIR.exists():
        logging.info("[OK] venv already exists")
        return

    logging.info("[STEP] Creating virtual environment")

    # 🔥 Find real python (NOT PyInstaller exe)
    python_cmd = "python"

    try:
        subprocess.run(
            [python_cmd, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except Exception:
        raise RuntimeError("System Python not found. Please install Python.")

    subprocess.run(
        [python_cmd, "-m", "venv", str(VENV_DIR)],
        check=True
    )

    logging.info("[OK] venv created")

# --------------------------------------------------
# Main installer
# --------------------------------------------------
def main(otp=None, server_addr=None):

    global INSTALLER_OTP
    global INSTALLER_SERVER_ADDR

    INSTALLER_OTP = otp
    INSTALLER_SERVER_ADDR = server_addr
    
    logging.info("=== BUILD VERSION 2 WITH GUI INPUT FIX ===")
    # --------------------------------------------------
    # 1. Anti-debug (installer mode)
    # --------------------------------------------------
    logging.info("[1] Anti-debug (installer mode)")
    anti_debug(strict=True, installer_mode=True)

    # --------------------------------------------------
    # 2. Secure filesystem layout
    # --------------------------------------------------
    logging.info("[2] Secure filesystem layout")
    create_secure_layout()

    # --------------------------------------------------
    # 3. Runtime payload (code + configs)
    # --------------------------------------------------
    logging.info("[3] Installing runtime payload")
    install_runtime()
    
    # --------------------------------------------------
    # 4. TPM identity (safe for installer)
    # --------------------------------------------------
    logging.info("[4] TPM identity provisioning")
    provision_tpm_identity()
    device_pubkey = get_device_pubkey_installer_safe()

    

    # --------------------------------------------------
    # 5. Windows runtime prerequisites
    # --------------------------------------------------
    if IS_WINDOWS:
        logging.info("[5] Verifying Python & VC runtime")
        from security.windows_runtime import check_vc_runtime

        check_vc_runtime()
        verify_python_and_pip()

    create_venv()
    
    # --------------------------------------------------
    # 6. Python dependencies
    # --------------------------------------------------
    logging.info("[6] Installing Python dependencies")
    try:
        install_python_deps()
        logging.info("[DEBUG] Python deps installed")
    except Exception as e:
        logging.error("[ERROR] install_python_deps crashed: %s", e)
        raise

    logging.info("🔥 DEPS DONE → MOVING TO ENROLLMENT")

    check()
    logging.info("[6.1] Installing spaCy model")
    install_spacy_model()
    # --------------------------------------------------
    # 7. Native ML dependencies
    # --------------------------------------------------
    if not IS_WINDOWS:
        logging.info("[7] Installing OpenFace")
        install_openface()
    else:
        logging.info("[7] Windows OpenFace already bundled")

    if not IS_WINDOWS:
        logging.info("[8] Installing openSMILE")
        install_opensmile()
    else:
        logging.info("[8] Windows opensmile already bundled")

    install_ffmpeg()

    # --------------------------------------------------
    # 9. VERIFY dependencies (NOW they exist)
    # --------------------------------------------------
    logging.info("[9] Verifying platform dependencies")
    verify_windows_deps()

    # --------------------------------------------------
    # 10. OTP enrollment (server is running)
    # --------------------------------------------------
    logging.info("[10] OTP enrollment")
    otp_enrollment(device_pubkey, INSTALLER_OTP, INSTALLER_SERVER_ADDR)

    # --------------------------------------------------
    # 11. TPM identity already initialized earlier
    # --------------------------------------------------
    if IS_WINDOWS:
        logging.info("[11] Windows TPM signer already initialized")
    else:
        logging.info("[11] Sealing master secret")
        seal_master_secret()

    # --------------------------------------------------
    # 12. Integrity baseline
    # --------------------------------------------------
    logging.info("[12] Writing integrity baseline")
    write_baseline()

    # --------------------------------------------------
    # 13. Persist install state
    # --------------------------------------------------
    logging.info("[13] Persisting install state")
    write_install_state()

    logging.info("🔥 INSTALLER COMPLETED SUCCESSFULLY")


# --------------------------------------------------
# Entry
# --------------------------------------------------
if __name__ == "__main__":
    main()
