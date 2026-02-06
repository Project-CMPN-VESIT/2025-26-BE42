import os
import stat
import hashlib
from pathlib import Path

FEDERATED_DIR = Path.home() / ".federated"

SUBDIRS = [
    "bin",
    "agents/lda",
    "agents/trainer",
    "agents/dp",
    "agents/enc",
    "state",
    "data/secure_store",
    "data/cache",
    "logs",
    "integrity",
]

def _chmod_owner_only(path: Path):
    if os.name == "posix":
        path.chmod(stat.S_IRWXU)  # 700

def create_secure_layout():
    if FEDERATED_DIR.exists():
        raise RuntimeError("[SECURITY] .federated already exists")

    FEDERATED_DIR.mkdir(mode=0o700)
    _chmod_owner_only(FEDERATED_DIR)

    for sub in SUBDIRS:
        p = FEDERATED_DIR / sub
        p.mkdir(parents=True, exist_ok=True)
        _chmod_owner_only(p)

    # install lock (used later for anti-tamper)
    lock = FEDERATED_DIR / "state" / "install.lock"
    lock.write_text("installed")
    _chmod_owner_only(lock)

    # integrity baseline placeholder
    baseline = FEDERATED_DIR / "integrity" / "baseline.sha256"
    baseline.write_text("")  # filled in Step 4
    _chmod_owner_only(baseline)

    print("[OK] Secure .federated layout created")

