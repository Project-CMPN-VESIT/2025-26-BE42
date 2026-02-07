from .integrity_guard import verify_integrity
from .tpm_guard import unseal_master_secret
from .self_destruct import trigger_self_destruct
from .validate_deps import main as validate_deps
import os
import platform

IS_WINDOWS = platform.system().lower() == "windows"


def runtime_guard():
    # 1. Integrity verification
    verify_integrity()

    # 2. Dependency validation (Windows only)
    if IS_WINDOWS:
        validate_deps()

    # 3. TPM binding
    master_secret = unseal_master_secret()
    if not master_secret:
        trigger_self_destruct("Master secret unavailable")

    # 4. Privilege sanity (Linux)
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        trigger_self_destruct("Running as root is forbidden")

    return master_secret
