"""
tpm_guard.py

BUG-1 FIX: creationflags=subprocess.CREATE_NO_WINDOW is Windows-only.
           Using it on Linux raises ValueError and crashes the process.
           Fixed by only passing creationflags on Windows.
"""

import subprocess
from .self_destruct import trigger_self_destruct
import sys
from pathlib import Path
import platform

IS_WINDOWS = platform.system().lower() == "windows"

BASE_DIR   = Path.home() / ".federated"
TPM_DIR    = BASE_DIR / "tpm"
SEALED_CTX = str(TPM_DIR / "sealed_secret.ctx")
PUBKEY_PEM = TPM_DIR / "device_pubkey.pem"

WINDOWS_SIGNER = BASE_DIR / "bin" / "windows_signer.exe"


def _subprocess_kwargs(**extra) -> dict:
    """Return subprocess kwargs with CREATE_NO_WINDOW only on Windows."""
    kw = dict(extra)
    if IS_WINDOWS:
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kw


# ── SIGN MESSAGE (ECDSA P-256) ────────────────────────────────────────────────

def sign_message(message: bytes) -> bytes:
    if not message:
        trigger_self_destruct("Empty message for signing")

    if IS_WINDOWS and not WINDOWS_SIGNER.exists():
        trigger_self_destruct("Windows TPM signer missing")

    try:
        if IS_WINDOWS:
            proc = subprocess.run(
                [str(WINDOWS_SIGNER), "--sign"],
                input=message,
                stdout=subprocess.PIPE,
                check=True,
                **_subprocess_kwargs()
            )
            return proc.stdout
        else:
            # BUG-1 FIX: no creationflags on Linux
            proc = subprocess.run(
                [
                    "tpm2_sign",
                    "-c", str(TPM_DIR / "device.ctx"),
                    "-g", "sha256",
                    "-s", "ecdsa",
                    "-o", "-"
                ],
                input=message,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return proc.stdout

    except Exception:
        trigger_self_destruct("TPM signing failed")


# ── UNSEAL MASTER SECRET ──────────────────────────────────────────────────────

def unseal_master_secret() -> bytes:
    try:
        if IS_WINDOWS:
            secret_path = BASE_DIR / "secrets" / "master.bin"
            if not secret_path.exists():
                print("[TPM] Master secret missing → creating (first run)")
                from installer.security.tpm_seal import create_master_secret_windows
                create_master_secret_windows()
            return secret_path.read_bytes()
        else:
            # BUG-1 FIX: no creationflags on Linux
            output = subprocess.check_output([
                "tpm2_unseal",
                "-c", SEALED_CTX
            ])
            if not output:
                raise RuntimeError
            return output

    except Exception:
        trigger_self_destruct("TPM unseal failed (hardware state mismatch)")


# ── GET DEVICE PUBLIC KEY ─────────────────────────────────────────────────────

def get_device_pubkey() -> bytes:
    try:
        if IS_WINDOWS:
            pubkey_file = TPM_DIR / "device_pubkey.pem"

            if not WINDOWS_SIGNER.exists():
                print("[DEBUG] Windows signer missing at:", WINDOWS_SIGNER)
                return b""

            if not pubkey_file.exists():
                subprocess.run(
                    [str(WINDOWS_SIGNER), "--pubkey", str(pubkey_file)],
                    check=True,
                    **_subprocess_kwargs()
                )

            if not pubkey_file.exists():
                print("[DEBUG] Pubkey file not created")
                return b""

            return pubkey_file.read_bytes()

        else:
            if not PUBKEY_PEM.exists():
                sys.exit("[SECURITY] TPM identity not initialized")
            return PUBKEY_PEM.read_bytes()

    except Exception as e:
        print("[DEBUG] PUBKEY ERROR:", e)
        return b""