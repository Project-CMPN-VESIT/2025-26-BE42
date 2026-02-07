import os
import sys
import subprocess
from pathlib import Path
import secrets

BASE_DIR = Path.home() / ".federated"
TPM_DIR = BASE_DIR / "tpm"
SECRETS_DIR = BASE_DIR / "secrets"

SEALED_OBJ = TPM_DIR / "sealed_secret.ctx"
SECRET_PLAIN = SECRETS_DIR / "master.bin"

# PCRs we bind to (secure defaults)
PCRS = "sha256:0,2,4,7"


def _run(cmd):
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def seal_master_secret():
    """
    Generates and seals a master secret to TPM PCRs.
    This runs ONCE.
    """
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

    # Destroy plaintext
    SECRET_PLAIN.unlink()

    for f in ["sealed.pub", "sealed.priv"]:
        p = TPM_DIR / f
        if p.exists():
            p.unlink()

    print("[TPM] Master secret sealed successfully")


def unseal_master_secret() -> bytes:
    """
    Unseals secret ONLY if PCR state matches.
    """
    if not SEALED_OBJ.exists():
        sys.exit("[SECURITY] Sealed secret missing")

    print("[TPM] Unsealing master secret")
    output = subprocess.check_output([
        "tpm2_unseal",
        "-c", str(SEALED_OBJ)
    ])

    if not output:
        sys.exit("[SECURITY] TPM unseal failed")

    return output
