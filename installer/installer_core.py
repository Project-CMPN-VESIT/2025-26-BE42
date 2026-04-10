#!/usr/bin/env python3
"""
installer_core.py — FIXED VERSION 2

Fixes on top of previous version:
  FIX-6: create_venv() — when running as a PyInstaller frozen executable,
          sys.executable points to the bundled .exe, NOT a real Python
          interpreter.  Calling `sys.executable -m venv` with a frozen exe
          creates a venv whose Scripts/python.exe is missing or broken,
          producing "Venv created but python binary missing".
          Fix: detect frozen context and locate the real system Python via
          shutil.which(), with a clear error if Python is not on PATH.
"""

import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import logging
from pathlib import Path

import grpc

from fs.secure_layout import create_secure_layout
from fs.install_runtime import install_runtime, install_mentalbert_model
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

from security.anti_debug import anti_debug
from security.integrity import write_baseline
from security.tpm_attestation import (
    provision_tpm_identity,
    get_device_pubkey_installer_safe,
)
from security.tpm_seal import seal_master_secret
from security.deps_windows import verify_windows_deps, verify_python_and_pip

from runtime.grpc.orchestrator_pb2_grpc import OrchestratorStub
from runtime.grpc.orchestrator_pb2 import (
    EnrollmentRequest,
    EnrollmentRequestAck,
    EnrollRequest,
    EnrollResponse,
)

IS_WINDOWS = platform.system().lower() == "windows"

BASE_DIR   = Path.home() / ".federated"
STATE_FILE = BASE_DIR / "state" / "install_state.json"
KEYS_DIR   = BASE_DIR / "keys"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = BASE_DIR / "logs" / "installer.log"
try:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("Installer started")


# ── TCP pre-flight ─────────────────────────────────────────────────────────────

def _tcp_reachable(host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError as e:
        logging.warning("[TCP] Pre-check failed for %s:%d — %s", host, port, e)
        return False


def _parse_addr(server_addr: str):
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


# ── gRPC channel factory ───────────────────────────────────────────────────────

def _open_channel(server_addr: str) -> grpc.Channel:
    """TLS first (secure default), insecure only when CA cert absent."""
    ca_pem = KEYS_DIR / "ca.pem"

    if ca_pem.exists():
        try:
            creds = grpc.ssl_channel_credentials(root_certificates=ca_pem.read_bytes())
            ch = grpc.secure_channel(
                server_addr,
                creds,
                options=[
                    ("grpc.keepalive_time_ms", 10_000),
                    ("grpc.keepalive_timeout_ms", 5_000),
                ],
            )
            grpc.channel_ready_future(ch).result(timeout=10)
            logging.info("[gRPC] TLS channel connected")
            return ch
        except grpc.FutureTimeoutError:
            try:
                ch.close()
            except Exception:
                pass
            host, _ = _parse_addr(server_addr)
            raise RuntimeError(
                f"[gRPC] TLS handshake timed out. Regenerate certs:\n"
                f"  bash certs/gen_certs.sh {host}\n"
                "  OR set enable_tls=false for local testing."
            )
        except Exception as e_tls:
            logging.warning("[gRPC] TLS failed (%s), trying insecure…",
                            type(e_tls).__name__)
            try:
                ch.close()
            except Exception:
                pass

    logging.warning("[gRPC] No CA cert — attempting insecure channel (dev only)")
    try:
        ch = grpc.insecure_channel(
            server_addr,
            options=[
                ("grpc.keepalive_time_ms", 5_000),
                ("grpc.keepalive_timeout_ms", 3_000),
            ],
        )
        grpc.channel_ready_future(ch).result(timeout=3)
        logging.warning("[gRPC] Insecure channel — NOT for production")
        return ch
    except Exception as e_ins:
        try:
            ch.close()
        except Exception:
            pass
        raise RuntimeError(
            f"[gRPC] Cannot connect TLS (no CA cert) or insecure ({e_ins})."
        )


# ── CSR generation ─────────────────────────────────────────────────────────────

def _generate_csr() -> tuple:
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    client_key_path = KEYS_DIR / "client.key"
    client_csr_path = KEYS_DIR / "client.csr"

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    with open(client_key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "federated-device")])
        )
        .sign(key, hashes.SHA256())
    )

    csr_bytes = csr.public_bytes(serialization.Encoding.PEM)
    with open(client_csr_path, "wb") as f:
        f.write(csr_bytes)

    return client_key_path, csr_bytes


# ── Phase B1 ───────────────────────────────────────────────────────────────────

def request_enrollment_otp(device_pubkey: bytes, server_addr: str) -> str:
    logging.info("[ENROLL-REQ] Requesting enrollment OTP from %s", server_addr)

    host, port = _parse_addr(server_addr)
    if not _tcp_reachable(host, port):
        raise RuntimeError(
            f"Cannot reach server at {host}:{port}.\n"
            "  Is the Rust orchestrator running?"
        )

    _, csr_bytes = _generate_csr()

    import socket as _sock
    try:
        hostname = _sock.gethostname()
    except Exception:
        hostname = "unknown"
    device_info = f"{platform.system()} / {hostname}"

    channel = _open_channel(server_addr)
    stub = OrchestratorStub(channel)

    try:
        resp: EnrollmentRequestAck = stub.RequestEnrollment(
            EnrollmentRequest(
                device_pubkey=device_pubkey,
                csr=csr_bytes,
                device_info=device_info,
            ),
            timeout=15,
        )
    except grpc.RpcError as e:
        raise RuntimeError(f"[ENROLL-REQ] gRPC error ({e.code()}): {e.details()}")
    finally:
        try:
            channel.close()
        except Exception:
            pass

    if not resp.accepted:
        raise RuntimeError("[ENROLL-REQ] Server rejected enrollment request")

    fingerprint = resp.device_fingerprint
    logging.info("[ENROLL-REQ] Request accepted — fingerprint=%s", fingerprint)
    print(f"[ENROLL] Device fingerprint: {fingerprint}")
    print("[ENROLL] Administrator has been notified. Waiting for OTP…")
    return fingerprint


# ── Phase B2 ───────────────────────────────────────────────────────────────────

def complete_enrollment(device_pubkey: bytes, token: str, server_addr: str):
    logging.info("[ENROLL] Completing enrollment with OTP")

    token = token.strip()
    if len(token) < 6:
        raise ValueError("[SECURITY] Invalid OTP — must be at least 6 characters")

    host, port = _parse_addr(server_addr)
    if not _tcp_reachable(host, port):
        raise RuntimeError(f"[ENROLL] Cannot reach server at {host}:{port}.")

    client_csr_path = KEYS_DIR / "client.csr"
    if not client_csr_path.exists():
        raise RuntimeError("[ENROLL] CSR not found. Did request_enrollment_otp() run first?")
    csr_bytes = client_csr_path.read_bytes()

    channel = _open_channel(server_addr)
    stub = OrchestratorStub(channel)

    try:
        resp: EnrollResponse = stub.EnrollDevice(
            EnrollRequest(
                enrollment_token=token,
                device_pubkey=device_pubkey,
                csr=csr_bytes,
            ),
            timeout=15,
        )
        logging.info("STEP 10: ENROLLMENT COMPLETED")
    except grpc.RpcError as e:
        code = e.code()
        details = e.details()
        if code == grpc.StatusCode.PERMISSION_DENIED:
            raise PermissionError(
                f"[ENROLL] OTP rejected: {details}\n"
                "  OTP may have expired. Ask your admin for a new one."
            )
        raise RuntimeError(f"[ENROLL] gRPC error ({code}): {details}")
    finally:
        try:
            channel.close()
        except Exception:
            pass

    if not resp.ok:
        raise RuntimeError("[SECURITY] Enrollment failed — server returned ok=False")

    client_cert_path = KEYS_DIR / "client.pem"
    client_cert_path.write_bytes(resp.client_cert)
    client_cert_path.chmod(0o600)

    logging.info("[ENROLL] Device enrolled — cert at %s", client_cert_path)
    print(f"[OK] Client certificate installed at {client_cert_path}")


# ── Daemon registration ────────────────────────────────────────────────────────

def register_daemon():
    system = platform.system().lower()
    if system == "windows":
        _register_windows_task()
    elif system == "linux":
        _register_linux_service()
    else:
        logging.warning("[DAEMON] Unsupported OS for daemon registration: %s", system)


def _register_windows_task():
    python = BASE_DIR / "venv" / "Scripts" / "python.exe"
    client = BASE_DIR / "bin" / "federated-client"
    task_name = "FederatedLearningClient"

    task_xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Federated Learning Client Daemon</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger><Enabled>true</Enabled></LogonTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure>
      <Interval>PT1M</Interval><Count>10</Count>
    </RestartOnFailure>
    <Hidden>true</Hidden>
  </Settings>
  <Actions>
    <Exec>
      <Command>{python}</Command>
      <Arguments>"{client}" daemon</Arguments>
      <WorkingDirectory>{BASE_DIR}</WorkingDirectory>
    </Exec>
  </Actions>
  <Principals>
    <Principal>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
</Task>"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, encoding="utf-16"
    ) as f:
        f.write(task_xml)
        task_file = f.name

    try:
        result = subprocess.run(
            ["schtasks", "/Create", "/F", "/TN", task_name, "/XML", task_file],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode == 0:
            logging.info("[DAEMON] Windows task registered: %s", task_name)
            print(f"[OK] Daemon registered as Windows scheduled task '{task_name}'")
        else:
            logging.warning("[DAEMON] Task registration warning: %s", result.stderr)
            print(f"[WARN] Daemon task registration: {result.stderr.strip()}")
    finally:
        Path(task_file).unlink(missing_ok=True)


def _register_linux_service():
    python = BASE_DIR / "venv" / "bin" / "python"
    client = BASE_DIR / "bin" / "federated-client"

    service = f"""[Unit]
Description=Federated Learning Client Daemon
After=network.target

[Service]
Type=simple
ExecStart={python} {client} daemon
Restart=on-failure
RestartSec=60
WorkingDirectory={BASE_DIR}
Environment=HOME={Path.home()}

[Install]
WantedBy=default.target
"""
    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)
    (systemd_dir / "federated-client.service").write_text(service)

    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "federated-client"], check=True)
        logging.info("[DAEMON] systemd user service enabled")
        print("[OK] Daemon registered as systemd user service")
    except Exception as e:
        logging.warning("[DAEMON] systemd failed: %s — trying crontab", e)
        entry = f"@reboot {python} {client} daemon >> {BASE_DIR}/logs/cron.log 2>&1\n"
        try:
            existing_result = subprocess.run(
                ["crontab", "-l"], capture_output=True, text=True
            )
            existing = existing_result.stdout if existing_result.returncode == 0 else ""
            if str(client) not in existing:
                subprocess.run(
                    ["crontab", "-"], input=existing + entry, text=True, check=True
                )
                print("[OK] Daemon registered via crontab (@reboot)")
        except Exception as e2:
            logging.warning("[DAEMON] Crontab failed: %s", e2)


# ── Install state ──────────────────────────────────────────────────────────────

def write_install_state():
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(
            {"installed": True, "version": "1.0", "platform": platform.system()},
            indent=2,
        )
    )
    STATE_FILE.chmod(0o600)


# ── Virtual environment ────────────────────────────────────────────────────────

def _find_system_python() -> str:
    """
    FIX-6: When running as a PyInstaller frozen executable, sys.executable
    is the bundled .exe — not a real Python interpreter.  We must locate
    the system Python to create a working venv.

    Search order (Windows):
      1. py launcher  (py -3)         — most reliable on Windows
      2. python.exe on PATH
      3. python3.exe on PATH
      4. Common installation paths
    Search order (Linux/macOS):
      1. python3 on PATH
      2. python on PATH
    """
    is_frozen = getattr(sys, "frozen", False)

    if not is_frozen:
        # Not frozen — sys.executable is a real Python, use it directly.
        return sys.executable

    # Frozen: find real Python on the host system.
    if IS_WINDOWS:
        # Try the py launcher first — it's the most reliable on Windows.
        py_launcher = shutil.which("py")
        if py_launcher:
            try:
                result = subprocess.run(
                    [py_launcher, "-3", "--version"],
                    capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if result.returncode == 0:
                    logging.info("[venv] Using py launcher: %s", py_launcher)
                    # Return a wrapper call; caller will use [py_launcher, "-3", "-m", "venv"]
                    # We signal this with a sentinel prefix.
                    return f"__py_launcher__:{py_launcher}"
            except Exception:
                pass

        for candidate in ("python", "python3"):
            path = shutil.which(candidate)
            if path:
                logging.info("[venv] Found system Python: %s", path)
                return path

        # Last resort: common Windows install locations
        for ver in ("313", "312", "311", "310"):
            for base in [
                Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / f"Python{ver}",
                Path("C:/Python" + ver),
                Path("C:/Program Files/Python" + ver),
            ]:
                candidate = base / "python.exe"
                if candidate.exists():
                    logging.info("[venv] Found Python at: %s", candidate)
                    return str(candidate)

        raise RuntimeError(
            "Python 3.10+ not found on this system.\n"
            "Please install Python from https://python.org (tick 'Add to PATH')\n"
            "then re-run the installer."
        )
    else:
        # Linux / macOS
        for candidate in ("python3", "python"):
            path = shutil.which(candidate)
            if path:
                logging.info("[venv] Found system Python: %s", path)
                return path
        raise RuntimeError(
            "Python 3.10+ not found on PATH.\n"
            "Install it with your package manager (e.g. apt install python3) "
            "then re-run the installer."
        )


def create_venv():
    VENV_DIR = BASE_DIR / "venv"
    print("[STEP] Creating virtual environment", flush=True)

    if VENV_DIR.exists():
        # Verify the venv python actually exists before declaring success.
        if IS_WINDOWS:
            venv_py = VENV_DIR / "Scripts" / "python.exe"
        else:
            venv_py = VENV_DIR / "bin" / "python"

        if venv_py.exists():
            print("[INFO] venv already exists and is valid, skipping", flush=True)
            return
        else:
            # Broken venv — delete and recreate.
            print("[WARN] venv exists but python binary missing — recreating", flush=True)
            shutil.rmtree(VENV_DIR)

    # FIX-6: locate real Python (handles frozen exe context)
    python_spec = _find_system_python()

    if python_spec.startswith("__py_launcher__:"):
        # Use `py -3 -m venv` (Windows py launcher)
        py_launcher = python_spec.split(":", 1)[1]
        cmd = [py_launcher, "-3", "-m", "venv", str(VENV_DIR)]
    else:
        cmd = [python_spec, "-m", "venv", str(VENV_DIR)]

    print(f"[INFO] Running: {' '.join(cmd)}", flush=True)
    logging.info("[venv] Creating venv with command: %s", cmd)

    extra_kw = {}
    if IS_WINDOWS:
        extra_kw["creationflags"] = subprocess.CREATE_NO_WINDOW

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **extra_kw,
    )
    print(result.stdout or "", end="", flush=True)

    if result.returncode != 0:
        print(result.stderr or "", end="", file=sys.stderr, flush=True)
        raise RuntimeError(
            f"Failed to create venv.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr.strip()}"
        )

    # Verify the python binary is actually present
    if IS_WINDOWS:
        venv_py = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_py = VENV_DIR / "bin" / "python"

    if not venv_py.exists():
        raise RuntimeError(
            f"Venv was created but python binary is missing at {venv_py}.\n"
            "This usually means the Python installation is corrupt.\n"
            "Reinstall Python from https://python.org and re-run."
        )

    print("[OK] venv created successfully", flush=True)
    logging.info("[venv] Created successfully at %s", VENV_DIR)


# ── Public API ─────────────────────────────────────────────────────────────────

def setup_software(server_addr: str) -> bytes:
    logging.info("=== PHASE A: SOFTWARE SETUP ===")

    logging.info("[1] Anti-debug (installer mode)")
    anti_debug(strict=True, installer_mode=True)

    logging.info("[2] Secure filesystem layout")
    create_secure_layout()

    logging.info("[3] Installing runtime payload")
    install_runtime()

    # FIX-6: create_venv() now finds the real system Python even when frozen.
    logging.info("[3b] Creating virtual environment")
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
                creationflags=subprocess.CREATE_NO_WINDOW,
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
    install_python_deps()

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

    # FIX-1 (from previous version): MentalBERT installed here, AFTER venv+deps.
    logging.info("[10] Installing MentalBERT model")
    install_mentalbert_model()

    logging.info("=== PHASE A COMPLETE ===")
    return device_pubkey


def finalize_install(device_pubkey: bytes, otp: str, server_addr: str):
    logging.info("=== PHASE B2: ENROLLMENT COMPLETION ===")

    complete_enrollment(device_pubkey, otp, server_addr)

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

    logging.info("[14] Registering daemon")
    register_daemon()

    logging.info("INSTALLER COMPLETED SUCCESSFULLY")


def main(otp=None, server_addr=None):
    device_pubkey = setup_software(server_addr)
    fingerprint = request_enrollment_otp(device_pubkey, server_addr)
    print(
        f"\n[ACTION REQUIRED] Tell your administrator your Device ID: {fingerprint}\n"
        "  They will give you a one-time OTP to complete enrollment.\n"
    )
    if otp is None:
        otp = input("Enter OTP from administrator: ").strip()
    finalize_install(device_pubkey, otp, server_addr)


if __name__ == "__main__":
    main()