import subprocess
from .self_destruct import trigger_self_destruct
import sys
from pathlib import Path
import platform

IS_WINDOWS = platform.system().lower() == "windows"

BASE_DIR = Path.home() / ".federated"
TPM_DIR = BASE_DIR / "tpm"
SEALED_CTX = str(TPM_DIR / "sealed_secret.ctx")
PUBKEY_PEM = TPM_DIR / "device_pubkey.pem"

WINDOWS_SIGNER = BASE_DIR / "bin" / "windows_signer.exe"


# --------------------------------------------------
# SIGN MESSAGE (ECDSA P-256)
# --------------------------------------------------

def sign_message(message: bytes) -> bytes:
    if IS_WINDOWS and not WINDOWS_SIGNER.exists(): 
        trigger_self_destruct("Windows TPM signer missing")
    
    print("[TPM] Signing message using Windows TPM signer")
    
    try:
        if IS_WINDOWS:
            # Call Windows CNG signer
            proc = subprocess.run(
                [str(WINDOWS_SIGNER), "--sign"],
                input=message,
                stdout=subprocess.PIPE,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            return proc.stdout

        else:
            # Linux TPM2 tools
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
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            return proc.stdout

    except Exception:
        trigger_self_destruct("TPM signing failed")


# --------------------------------------------------
# UNSEAL MASTER SECRET
# --------------------------------------------------

def unseal_master_secret() -> bytes:
    try:
        if IS_WINDOWS:
            # Windows: no TPM sealing yet
            # For parity, we will later implement CNG-protected secret
            # For now: read file-based sealed secret
            secret_path = BASE_DIR / "secrets" / "master.bin"
            if not secret_path.exists():
                trigger_self_destruct("Master secret missing")
            return secret_path.read_bytes()

        else:
            # Linux TPM
            output = subprocess.check_output([
                "tpm2_unseal",
                "-c", SEALED_CTX
            ])
            if not output:
                raise RuntimeError
            return output

    except Exception:
        trigger_self_destruct("TPM unseal failed (hardware state mismatch)")


# --------------------------------------------------
# GET DEVICE PUBLIC KEY
# --------------------------------------------------

def get_device_pubkey() -> bytes:
    try:
        if IS_WINDOWS:
            # Ask signer to export PEM public key
            proc = subprocess.run(
                [str(WINDOWS_SIGNER), "--export-pub"],
                stdout=subprocess.PIPE,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            return proc.stdout

        else:
            if not PUBKEY_PEM.exists():
                sys.exit("[SECURITY] TPM identity not initialized")
            return PUBKEY_PEM.read_bytes()

    except Exception:
        trigger_self_destruct("Failed to retrieve device public key")