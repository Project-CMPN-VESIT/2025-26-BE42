import os
import sys
import time
from pathlib import Path

from .anti_debug import anti_debug
from .integrity import verify_integrity
from .tpm_attestation import tpm_attestation
from .tpm_seal import unseal_master_secret
from .self_destruct import trigger_self_destruct

FEDERATED_DIR = Path.home() / ".federated"
RUNTIME_LOCK = FEDERATED_DIR / "state" / "runtime.lock"


def enforce():
    """
    Runtime security gate.
    MUST be called before any sensitive operation.
    """

    try:
        # --------------------------------------------------
        # 1. Anti-debug (strict, runtime mode)
        # --------------------------------------------------
        anti_debug(strict=True, installer_mode=False)

        # --------------------------------------------------
        # 2. Secure layout presence
        # --------------------------------------------------
        if not FEDERATED_DIR.exists():
            trigger_self_destruct("Missing .federated directory")

        # --------------------------------------------------
        # 3. Integrity verification
        # --------------------------------------------------
        verify_integrity()

        # --------------------------------------------------
        # 4. TPM presence check
        # --------------------------------------------------
        tpm_attestation()

        # --------------------------------------------------
        # 5. TPM secret unseal test (no use yet)
        # --------------------------------------------------
        secret = unseal_master_secret()
        if not secret or len(secret) < 16:
            trigger_self_destruct("Invalid TPM secret")

        # --------------------------------------------------
        # 6. Runtime execution lock (anti-replay / anti-fork)
        # --------------------------------------------------
        if RUNTIME_LOCK.exists():
            trigger_self_destruct("Concurrent runtime detected")

        RUNTIME_LOCK.write_text(str(os.getpid()))
        os.chmod(RUNTIME_LOCK, 0o600)

        # --------------------------------------------------
        # 7. Timing sanity (anti-attach mid-flight)
        # --------------------------------------------------
        t0 = time.time()
        time.sleep(0.01)
        t1 = time.time()

        if (t1 - t0) > 0.2:
            trigger_self_destruct("Timing anomaly detected")

    except SystemExit:
        raise
    except Exception as e:
        trigger_self_destruct(f"Runtime guard failure: {e}")
