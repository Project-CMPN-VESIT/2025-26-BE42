from installer.security.integrity import verify_integrity
from .tpm_guard import unseal_master_secret
from .self_destruct import trigger_self_destruct
import os
import platform
import time

IS_WINDOWS = platform.system().lower() == "windows"


def runtime_guard():
    # 1. Integrity verification
    time.sleep(1)
    verify_integrity()

    # 3. TPM binding
    master_secret = unseal_master_secret()
    if not master_secret:
        trigger_self_destruct("Master secret unavailable")

    # 4. Privilege sanity (Linux)
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        trigger_self_destruct("Running as root is forbidden")

    return master_secret
