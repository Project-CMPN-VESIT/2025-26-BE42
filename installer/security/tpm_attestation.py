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


# --------------------------------------------------
# Public entry
# --------------------------------------------------
def tpm_attestation():
    system = platform.system().lower()

    if system == "linux":
        _linux_tpm_check()
    elif system == "windows":
        _windows_tpm_check()
    else:
        sys.exit("[SECURITY] Unsupported OS for TPM attestation")


# --------------------------------------------------
# Linux TPM checks
# --------------------------------------------------
def _linux_tpm_check():
    if not os.path.exists("/sys/class/tpm/tpm0"):
        sys.exit("[SECURITY] TPM not found")

    try:
        kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "check": True
        }

        if platform.system().lower() == "windows":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        subprocess.run(["tpm2_getcap", "properties-fixed"], **kwargs)

    except Exception:
        sys.exit("[SECURITY] TPM tools not available or TPM blocked")

    nonce = secrets.token_bytes(32)
    digest = hashlib.sha256(nonce).hexdigest()
    if not digest:
        sys.exit("[SECURITY] TPM entropy failure")


# --------------------------------------------------
# Windows TPM presence check ONLY
# --------------------------------------------------
def _windows_tpm_check():
    try:
        result = subprocess.check_output(
            ["powershell", "-Command", "(Get-Tpm).TpmReady"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        if result.lower() != "true":
            sys.exit("[SECURITY] TPM not ready")

        print("[TPM] Windows TPM detected and ready")

    except Exception:
        sys.exit("[SECURITY] TPM not found or disabled on Windows")


# --------------------------------------------------
# Linux-only provisioning
# --------------------------------------------------
def _run(cmd):
    kwargs = {
        "check": True,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL
    }

    if platform.system().lower() == "windows":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    subprocess.run(cmd, **kwargs)


def provision_tpm_identity():
    system = platform.system().lower()

    # ------------------------------
    # Windows: DO NOT provision
    # ------------------------------
    if system == "windows":
        print("[TPM] Skipping TPM provisioning on Windows (runtime enforced)")
        return

    # ------------------------------
    # Linux provisioning
    # ------------------------------
    TPM_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists("/sys/class/tpm/tpm0"):
        sys.exit("[SECURITY] TPM not found")

    if DEVICE_CTX.exists() and PUBKEY_PEM.exists():
        print("[TPM] Device identity already provisioned")
        return

    print("[TPM] Creating ECC primary key")
    _run([
        "tpm2_createprimary",
        "-C", "o",
        "-G", "ecc",
        "-g", "sha256",
        "-c", str(PRIMARY_CTX)
    ])

    print("[TPM] Creating ECC device signing key")
    _run([
        "tpm2_create",
        "-C", str(PRIMARY_CTX),
        "-G", "ecc",
        "-g", "sha256",
        "-u", str(TPM_DIR / "device.pub"),
        "-r", str(TPM_DIR / "device.priv"),
        "-a", "sign|fixedtpm|fixedparent|sensitivedataorigin|userwithauth"
    ])

    print("[TPM] Loading ECC device key")
    _run([
        "tpm2_load",
        "-C", str(PRIMARY_CTX),
        "-u", str(TPM_DIR / "device.pub"),
        "-r", str(TPM_DIR / "device.priv"),
        "-c", str(DEVICE_CTX)
    ])

    print("[TPM] Exporting public key (PEM)")
    _run([
        "tpm2_readpublic",
        "-c", str(DEVICE_CTX),
        "-f", "pem",
        "-o", str(PUBKEY_PEM)
    ])


# --------------------------------------------------
# Public key access
# --------------------------------------------------
def get_device_pubkey() -> bytes:
    if not PUBKEY_PEM.exists():
        sys.exit("[SECURITY] TPM identity not initialized")
    return PUBKEY_PEM.read_bytes()


def get_device_pubkey_installer_safe() -> bytes:
    system = platform.system().lower()

    if system == "linux":
        if PUBKEY_PEM.exists():
            return PUBKEY_PEM.read_bytes()
        sys.exit("[SECURITY] TPM identity not initialized")

    elif system == "windows":

        signer = BASE_DIR / "bin" / "windows_signer.exe"
        pubkey_file = TPM_DIR / "device_pubkey.pem"

        TPM_DIR.mkdir(parents=True, exist_ok=True)

        if pubkey_file.exists():
            return pubkey_file.read_bytes()

        print("[TPM] Step 1: entering TPM provisioning")

        print("[TPM] Step 2: checking signer path")
        print("[TPM] signer:", signer)
        print("[TPM] exists:", signer.exists())

        print("[TPM] Step 3: running --init")

        proc = subprocess.Popen(
            [str(signer), "--init"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )

        proc.wait(timeout=10)

        print("[TPM] Step 4: init DONE")

        print("[TPM] Step 5: running --pubkey")

        proc = subprocess.Popen(
            [str(signer), "--pubkey", str(pubkey_file)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )

        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError("TPM pubkey export HANG detected")

        if proc.returncode != 0:
            raise RuntimeError(f"TPM pubkey export failed: {proc.returncode}")
        
        print("[TPM] Step 6: pubkey DONE")

        print("[TPM] Step 7: reading pubkey file")

        if not pubkey_file.exists():
            sys.exit("[SECURITY] Failed to obtain TPM public key")

        return pubkey_file.read_bytes()

    else:
        sys.exit("[SECURITY] Unsupported OS")
