# runtime_guard.py  FIXED VERSION
#
# FIX: Replaced relative imports (`from .tpm_guard`, `from .self_destruct`)
#      with absolute imports.  Relative imports require `runtime` to be loaded
#      as a package (via `import runtime.runtime_guard`), but federated_client.py
#      adds ~/.federated to sys.path and then does `from runtime.runtime_guard
#      import runtime_guard`, which works — however if runtime/__init__.py is
#      absent the relative imports inside this file still raise:
#          ImportError: attempted relative import with no known parent package
#      Using absolute imports makes this module safe regardless of how it is
#      loaded, and matches the style used everywhere else in the codebase.

from installer.security.integrity import verify_integrity
from runtime.tpm_guard import unseal_master_secret
from runtime.self_destruct import trigger_self_destruct
import os
import platform
import time

IS_WINDOWS = platform.system().lower() == "windows"


def runtime_guard():
    # 1. Integrity verification
    time.sleep(1)
    verify_integrity()

    # 2. TPM binding
    master_secret = unseal_master_secret()
    if not master_secret:
        trigger_self_destruct("Master secret unavailable")

    # 3. Privilege sanity (Linux only — Windows doesn't have geteuid)
    if not IS_WINDOWS and hasattr(os, "geteuid") and os.geteuid() == 0:
        trigger_self_destruct("Running as root is forbidden")

    return master_secret