import os
import stat
import platform
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
    "deps/windows",
]

def _chmod_owner_only(path: Path):
    # Linux / macOS only
    if os.name == "posix":
        path.chmod(stat.S_IRWXU)

def create_secure_layout():
    system = platform.system().lower()
    print(f"[FS] Creating secure layout at {FEDERATED_DIR}")

    # --------------------------------------------------
    # 1. Create root safely
    # --------------------------------------------------
    if not FEDERATED_DIR.exists():
        FEDERATED_DIR.mkdir(parents=True, exist_ok=True)
        _chmod_owner_only(FEDERATED_DIR)
    else:
        print("[FS] .federated already exists (continuing)")

    # --------------------------------------------------
    # 2. Create subdirectories
    # --------------------------------------------------
    for sub in SUBDIRS:
        p = FEDERATED_DIR / sub
        p.mkdir(parents=True, exist_ok=True)
        _chmod_owner_only(p)

    # --------------------------------------------------
    # 3. Install lock (idempotent)
    # --------------------------------------------------
    lock = FEDERATED_DIR / "state" / "install.lock"
    if not lock.exists():
        lock.write_text("installed")
        _chmod_owner_only(lock)

    # --------------------------------------------------
    # 4. Integrity baseline placeholder
    # --------------------------------------------------
    baseline = FEDERATED_DIR / "integrity" / "baseline.sha256"
    if not baseline.exists():
        baseline.write_text("")
        _chmod_owner_only(baseline)

    print("[FS] Secure filesystem layout created")
