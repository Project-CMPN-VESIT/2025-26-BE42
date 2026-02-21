import subprocess
from .self_destruct import trigger_self_destruct
import sys

from pathlib import Path

import platform
IS_WINDOWS = platform.system().lower() == "windows"

SEALED_CTX = str(Path.home() / ".federated" / "tpm" / "sealed_secret.ctx")
BASE_DIR = Path.home() / ".federated"
TPM_DIR = BASE_DIR / "tpm"
PUBKEY_PEM = TPM_DIR / "device_pubkey.pem"

def sign_message(message: bytes) -> bytes:
    try:
        proc = subprocess.run(
            [
                "tpm2_sign",
                "-c", str(Path.home() / ".federated/tpm/device.ctx"),
                "-g", "sha256",
                "-s", "ecdsa",
                "-o", "-"
            ],
            input=message,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return proc.stdout
    except Exception:
        trigger_self_destruct("TPM signing failed")

def unseal_master_secret() -> bytes:
    try:
        output = subprocess.check_output([
            "tpm2_unseal",
            "-c", SEALED_CTX
        ])
        if not output:
            raise RuntimeError
        return output
    except Exception:
        trigger_self_destruct("TPM unseal failed (hardware state mismatch)")

def get_device_pubkey() -> bytes:
    if not PUBKEY_PEM.exists():
        sys.exit("[SECURITY] TPM identity not initialized")
    return PUBKEY_PEM.read_bytes()