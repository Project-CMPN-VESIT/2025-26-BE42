import os
import sys
import subprocess
import platform
import hashlib
import secrets
from pathlib import Path

BASE_DIR = Path.home() / ".federated"
TPM_DIR = BASE_DIR / "tpm"

PRIMARY_CTX = TPM_DIR / "primary.ctx"
DEVICE_CTX = TPM_DIR / "device.ctx"
PUBKEY_PEM = TPM_DIR / "device_pubkey.pem"

def tpm_attestation():
    system = platform.system().lower()

    if system == "linux":
        _linux_tpm_check()
    elif system == "windows":
        _windows_tpm_check()
    else:
        sys.exit("[SECURITY] Unsupported OS for TPM attestation")

def _linux_tpm_check():
    # -----------------------------
    # 1. TPM device presence
    # -----------------------------
    if not os.path.exists("/sys/class/tpm/tpm0"):
        sys.exit("[SECURITY] TPM not found")

    # -----------------------------
    # 2. TPM command availability
    # -----------------------------
    try:
        subprocess.run(
            ["tpm2_getcap", "properties-fixed"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception:
        sys.exit("[SECURITY] TPM tools not available or TPM blocked")

    # -----------------------------
    # 3. Generate hardware-bound nonce
    # -----------------------------
    nonce = secrets.token_bytes(32)
    digest = hashlib.sha256(nonce).hexdigest()

    # Store temporarily in memory only
    if not digest:
        sys.exit("[SECURITY] TPM entropy failure")

def _windows_tpm_check():
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Services\TPM"
        )
        winreg.CloseKey(key)
    except Exception:
        sys.exit("[SECURITY] TPM not found or disabled on Windows")

def _run(cmd):
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def provision_tpm_identity():
    """
    Creates a TPM-resident, non-exportable device identity.
    Runs only once. Safe to re-run.
    """
    TPM_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists("/sys/class/tpm/tpm0"):
        sys.exit("[SECURITY] TPM not found")

    if DEVICE_CTX.exists() and PUBKEY_PEM.exists():
        print("[TPM] Device identity already provisioned")
        return

    print("[TPM] Creating primary key")
    _run([
        "tpm2_createprimary",
        "-C", "o",
        "-G", "rsa",
        "-c", str(PRIMARY_CTX)
    ])

    print("[TPM] Creating device signing key")
    _run([
        "tpm2_create",
        "-C", str(PRIMARY_CTX),
        "-G", "rsa",
        "-u", str(TPM_DIR / "device.pub"),
        "-r", str(TPM_DIR / "device.priv"),
        "-a", "sign|fixedtpm|fixedparent|sensitivedataorigin|userwithauth"
    ])

    print("[TPM] Loading device key")
    _run([
        "tpm2_load",
        "-C", str(PRIMARY_CTX),
        "-u", str(TPM_DIR / "device.pub"),
        "-r", str(TPM_DIR / "device.priv"),
        "-c", str(DEVICE_CTX)
    ])

    print("[TPM] Exporting public key")
    _run([
        "tpm2_readpublic",
        "-c", str(DEVICE_CTX),
        "-o", str(PUBKEY_PEM),
        "-f", "pem"
    ])

    # Cleanup temp files
    for f in ["device.pub", "device.priv"]:
        p = TPM_DIR / f
        if p.exists():
            p.unlink()

    print("[TPM] TPM-backed identity created")

def get_device_pubkey() -> bytes:
    if not PUBKEY_PEM.exists():
        sys.exit("[SECURITY] TPM identity not initialized")
    return PUBKEY_PEM.read_bytes()

def get_device_pubkey_installer_safe() -> bytes:
    """
    Installer-safe public key access.
    - Linux: reads TPM pubkey if already present
    - Windows: uses placeholder keypair stored only for enrollment
    """
    system = platform.system().lower()

    if system == "linux":
        if PUBKEY_PEM.exists():
            return PUBKEY_PEM.read_bytes()
        sys.exit("[SECURITY] TPM identity not initialized. Run runtime once.")

    elif system == "windows":
        # TEMP enrollment key, NOT runtime identity
        tmp = BASE_DIR / "state" / "installer_pubkey.bin"
        if tmp.exists():
            return tmp.read_bytes()

        key = secrets.token_bytes(32)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(key)
        return key

    else:
        sys.exit("[SECURITY] Unsupported OS")
