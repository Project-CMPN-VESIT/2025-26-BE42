#!/usr/bin/env python3
"""
installer_core.py

Phase 4: otp_enrollment uses plain ssl_channel_credentials with the server's
         IP SAN cert. No hostname override is needed or set.
BUG-3:   All logging calls use the %s positional placeholder.
FIX-ENR: otp_enrollment now has a TCP pre-flight check before TLS, catches
         grpc.FutureTimeoutError explicitly, and logs all failure paths so
         the cause is always visible in the log file.
"""

import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import json
import socket
import grpc
import platform
from pathlib import Path
import subprocess

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

from runtime.grpc.orchestrator_pb2_grpc import OrchestratorStub
from runtime.grpc.orchestrator_pb2 import EnrollRequest

IS_WINDOWS = platform.system().lower() == "windows"

BASE_DIR   = Path.home() / ".federated"
STATE_FILE = BASE_DIR / "state" / "install_state.json"
KEYS_DIR   = BASE_DIR / "keys"

INSTALLER_OTP         = None
INSTALLER_SERVER_ADDR = None

# ── Logging ───────────────────────────────────────────────────────────────────
import logging

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


# ── TCP pre-flight check ──────────────────────────────────────────────────────

def _tcp_reachable(host: str, port: int, timeout: float = 5.0) -> bool:
    """
    Check whether host:port accepts a TCP connection.
    This happens BEFORE TLS so we can tell 'server down' from 'cert mismatch'.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError as e:
        logging.warning("[TCP] Pre-check failed for %s:%d — %s", host, port, e)
        return False


def _parse_addr(server_addr: str):
    """Split 'host:port' into (host, int_port). Handles IPv6 brackets too."""
    if server_addr.startswith("["):
        bracket_end = server_addr.index("]")
        host = server_addr[1:bracket_end]
        port = int(server_addr[bracket_end + 2:])
    elif ":" in server_addr:
        parts = server_addr.rsplit(":", 1)
        host, port = parts[0], int(parts[1])
    else:
        host, port = server_addr, 50051
    return host, port


# ── OTP enrollment ────────────────────────────────────────────────────────────

def otp_enrollment(device_pubkey: bytes, token: str, server_addr: str):
    logging.info("[DEBUG] OTP received by installer: %s", token)
    logging.info("[DEBUG] SERVER_ADDR = %s", server_addr)
    logging.info("[DEBUG] CA exists: %s", (KEYS_DIR / "ca.pem").exists())

    token = token.strip()
    if len(token) < 6:
        sys.exit("[SECURITY] Invalid OTP — must be at least 6 characters")

    # ── 1. TCP pre-flight — fail fast with clear message ─────────────────────
    try:
        host, port = _parse_addr(server_addr)
    except Exception as e:
        logging.error("[ENROLL] Cannot parse server address %r: %s", server_addr, e)
        sys.exit(f"[ENROLL] Bad server address: {server_addr}")

    logging.info("[ENROLL] TCP pre-check → %s:%d", host, port)
    if not _tcp_reachable(host, port, timeout=5.0):
        msg = (
            f"[ENROLL] Cannot reach server at {host}:{port}.\n"
            f"  — Is the Rust orchestrator running on the server?\n"
            f"  — Is port {port} open in the firewall?\n"
            f"  — Is the server address correct? (got: {server_addr})"
        )
        logging.error(msg)
        sys.exit(msg)

    logging.info("[ENROLL] TCP pre-check OK — server is reachable at %s:%d", host, port)

    # ── 2. Key + CSR generation ───────────────────────────────────────────────
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    client_key = KEYS_DIR / "client.key"
    client_csr = KEYS_DIR / "client.csr"

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with open(client_key, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"federated-device"),
        ]))
        .sign(key, hashes.SHA256())
    )
    with open(client_csr, "wb") as f:
        f.write(csr.public_bytes(serialization.Encoding.PEM))

    # ── 3. Build gRPC TLS channel ─────────────────────────────────────────────
    ca_pem = KEYS_DIR / "ca.pem"
    if not ca_pem.exists():
        msg = (
            f"[ENROLL] CA certificate not found at {ca_pem}.\n"
            f"  Run: bash certs/gen_certs.sh {host}\n"
            f"  Then copy certs/ca.pem → installer/runtime/keys/ca.pem and rebuild."
        )
        logging.error(msg)
        sys.exit(msg)

    creds = grpc.ssl_channel_credentials(
        root_certificates=ca_pem.read_bytes()
    )
    channel = grpc.secure_channel(
        server_addr,
        creds,
        options=[
            ("grpc.keepalive_time_ms", 10_000),
            ("grpc.keepalive_timeout_ms", 5_000),
        ]
    )

    # ── 4. Wait for channel ready — explicit timeout + error logging ──────────
    logging.info("STEP 10: STARTING ENROLLMENT")
    logging.info("[ENROLL] Waiting for TLS channel ready (timeout=15s)...")

    try:
        grpc.channel_ready_future(channel).result(timeout=15)
        logging.info("[ENROLL] TLS channel READY")
    except grpc.FutureTimeoutError:
        channel.close()
        msg = (
            f"[ENROLL] TLS handshake timed out connecting to {server_addr}.\n"
            f"  Most likely causes:\n"
            f"  1. The CA certificate on the client does not match the server's CA.\n"
            f"     Regenerate: bash certs/gen_certs.sh {host}\n"
            f"     Copy: certs/ca.pem → installer/runtime/keys/ca.pem, then rebuild.\n"
            f"  2. The server certificate has no SAN for {host}.\n"
            f"     Check: openssl x509 -in certs/server.pem -noout -ext subjectAltName\n"
            f"  3. The Rust server started but TLS config is wrong."
        )
        logging.error(msg)
        sys.exit(msg)
    except Exception as e:
        channel.close()
        msg = f"[ENROLL] gRPC channel error: {type(e).__name__}: {e}"
        logging.error(msg)
        sys.exit(msg)

    # ── 5. Send EnrollDevice RPC ──────────────────────────────────────────────
    stub = OrchestratorStub(channel)
    logging.info("[ENROLL] Sending EnrollDevice RPC...")

    try:
        resp = stub.EnrollDevice(
            EnrollRequest(
                enrollment_token=token,
                device_pubkey=device_pubkey,
                csr=client_csr.read_bytes(),
            ),
            timeout=15
        )
        logging.info("STEP 10: ENROLLMENT COMPLETED")
    except grpc.RpcError as e:
        channel.close()
        code = e.code()
        details = e.details()
        if code == grpc.StatusCode.PERMISSION_DENIED:
            msg = (
                f"[ENROLL] Server rejected OTP: {details}\n"
                f"  The OTP may have expired (valid 60 seconds) or is incorrect.\n"
                f"  Generate a new OTP from the server console and retry."
            )
        elif code == grpc.StatusCode.UNAVAILABLE:
            msg = (
                f"[ENROLL] Server unavailable: {details}\n"
                f"  The server may have crashed or restarted. Check server logs."
            )
        else:
            msg = f"[ENROLL] gRPC RpcError ({code}): {details}"
        logging.error(msg)
        sys.exit(msg)
    except Exception as e:
        channel.close()
        msg = f"[ENROLL] Unexpected error during EnrollDevice: {type(e).__name__}: {e}"
        logging.error(msg)
        raise
    finally:
        try:
            channel.close()
        except Exception:
            pass

    if not resp.ok:
        sys.exit("[SECURITY] Enrollment failed — server returned ok=False")

    client_cert_path = KEYS_DIR / "client.pem"
    client_cert_path.write_bytes(resp.client_cert)
    client_cert_path.chmod(0o600)

    logging.info("[ENROLL] Device enrolled — client certificate installed at %s", client_cert_path)


def create_venv():
    BASE     = Path.home() / ".federated"
    VENV_DIR = BASE / "venv"

    print("[STEP] Creating virtual environment")

    if VENV_DIR.exists():
        print("[INFO] venv already exists, skipping")
        return

    python_cmd = "python"

    try:
        result = subprocess.run(
            [python_cmd, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError("Python not found")
    except Exception:
        raise RuntimeError("System Python not available")

    print("[DEBUG] Using system python:", python_cmd)

    result = subprocess.run(
        [python_cmd, "-m", "venv", str(VENV_DIR)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError("Failed to create venv")

    python_path = VENV_DIR / "Scripts" / "python.exe"
    if not python_path.exists():
        raise RuntimeError("Venv created but python.exe missing")

    print("[OK] venv created successfully")


def main(otp=None, server_addr=None):
    global INSTALLER_OTP, INSTALLER_SERVER_ADDR

    INSTALLER_OTP         = otp
    INSTALLER_SERVER_ADDR = server_addr

    logging.info("=== BUILD VERSION 4 — enrollment diagnostics ===")

    logging.info("[1] Anti-debug (installer mode)")
    anti_debug(strict=True, installer_mode=True)

    logging.info("[2] Secure filesystem layout")
    create_secure_layout()

    logging.info("[3] Installing runtime payload")
    install_runtime()
    create_venv()

    logging.info("[4] TPM identity provisioning")
    provision_tpm_identity()
    if IS_WINDOWS:
        logging.info("[TPM] Initializing Windows signer")
        signer = BASE_DIR / "bin" / "windows_signer.exe"
        try:
            subprocess.run(
                [str(signer), "--init"],
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            logging.info("[TPM] Windows signer initialized")
        except Exception as e:
            logging.error("[TPM] Signer init failed: %s", e)
            raise

    device_pubkey = get_device_pubkey_installer_safe()

    if IS_WINDOWS:
        logging.info("[5] Verifying Python & VC runtime")
        from security.windows_runtime import check_vc_runtime
        check_vc_runtime()
        verify_python_and_pip()

    logging.info("[6] Installing Python dependencies")
    try:
        install_python_deps()
        logging.info("[DEBUG] Python deps installed")
    except Exception as e:
        logging.error("[ERROR] install_python_deps crashed: %s", e)
        raise

    logging.info("DEPS DONE → MOVING TO ENROLLMENT")

    check()
    logging.info("[6.1] Installing spaCy model")
    install_spacy_model()

    if not IS_WINDOWS:
        logging.info("[7] Installing OpenFace")
        install_openface()
    else:
        logging.info("[7] Windows OpenFace already bundled")

    if not IS_WINDOWS:
        logging.info("[8] Installing openSMILE")
        install_opensmile()
    else:
        logging.info("[8] Windows openSMILE already bundled")

    install_ffmpeg()

    logging.info("[9] Verifying platform dependencies")
    verify_windows_deps()

    logging.info("[10] OTP enrollment")
    try:
        otp_enrollment(device_pubkey, INSTALLER_OTP, INSTALLER_SERVER_ADDR)
    except SystemExit:
        raise   # already logged with clear message above
    except Exception as e:
        logging.error("[10] Enrollment raised unexpected exception: %s", e, exc_info=True)
        raise

    if IS_WINDOWS:
        logging.info("[11] Creating Windows master secret")
        from installer.security.tpm_seal import create_master_secret_windows
        create_master_secret_windows()
    else:
        logging.info("[11] Sealing master secret")
        seal_master_secret()

    logging.info("[12] Persisting install state")
    write_install_state()

    logging.info("[13] Writing integrity baseline")
    write_baseline()

    logging.info("INSTALLER COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()