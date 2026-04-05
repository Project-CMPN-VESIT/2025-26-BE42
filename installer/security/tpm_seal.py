"""
tpm_seal.py

BUG-2 FIX: _run() passed creationflags=subprocess.CREATE_NO_WINDOW unconditionally.
           That flag is Windows-only; it raises ValueError on Linux/macOS.
           Fixed: creationflags only added when running on Windows.
"""

import os
import sys
import subprocess
from pathlib import Path
import platform
import secrets

IS_WINDOWS = platform.system().lower() == "windows"

BASE_DIR    = Path.home() / ".federated"
TPM_DIR     = BASE_DIR / "tpm"
SECRETS_DIR = BASE_DIR / "secrets"

SEALED_OBJ  = TPM_DIR / "sealed_secret.ctx"
SECRET_PLAIN = SECRETS_DIR / "master.bin"

PCRS = "sha256:0,2,4,7"


def _run(cmd):
    """Run a subprocess, suppressing output. Uses CREATE_NO_WINDOW only on Windows."""
    kwargs = {
        "check": True,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if IS_WINDOWS:
        # BUG-2 FIX: guard with IS_WINDOWS check
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.run(cmd, **kwargs)


def seal_master_secret():
    """Generate and seal a master secret to TPM PCRs. Runs ONCE."""
    TPM_DIR.mkdir(parents=True, exist_ok=True)
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)

    if SEALED_OBJ.exists():
        print("[TPM] Secret already sealed")
        return

    print("[TPM] Generating master secret")
    secret = secrets.token_bytes(32)
    SECRET_PLAIN.write_bytes(secret)

    print("[TPM] Sealing secret to PCRs:", PCRS)
    _run([
        "tpm2_create",
        "-C", "o",
        "-u", str(TPM_DIR / "sealed.pub"),
        "-r", str(TPM_DIR / "sealed.priv"),
        "-i", str(SECRET_PLAIN),
        "-L", PCRS
    ])

    _run([
        "tpm2_load",
        "-C", "o",
        "-u", str(TPM_DIR / "sealed.pub"),
        "-r", str(TPM_DIR / "sealed.priv"),
        "-c", str(SEALED_OBJ)
    ])

    # Destroy plaintext immediately
    SECRET_PLAIN.unlink()

    for f in ["sealed.pub", "sealed.priv"]:
        p = TPM_DIR / f
        if p.exists():
            p.unlink()

    print("[TPM] Master secret sealed successfully")


def create_master_secret_windows():
    """Windows fallback: store master secret as a file (no TPM sealing)."""
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)

    if SECRET_PLAIN.exists():
        print("[TPM] Master secret already exists (Windows)")
        return

    print("[TPM] Creating master secret (Windows fallback)")
    secret = secrets.token_bytes(32)
    SECRET_PLAIN.write_bytes(secret)


def unseal_master_secret() -> bytes:
    """Unseal secret. Creates + seals on first run if missing."""
    if not SEALED_OBJ.exists():
        print("[TPM] No sealed secret found → creating new one")
        seal_master_secret()

    try:
        print("[TPM] Unsealing master secret")
        # BUG-2 FIX: no creationflags here — tpm2_unseal is Linux-only
        output = subprocess.check_output([
            "tpm2_unseal",
            "-c", str(SEALED_OBJ)
        ])
        if not output:
            raise RuntimeError("Empty TPM output")
        return output
    except Exception as e:
        print("[TPM] Unseal failed:", e)
        sys.exit("[SECURITY] TPM unseal failed")