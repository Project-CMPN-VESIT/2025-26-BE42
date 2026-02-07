import subprocess
import sys
from pathlib import Path

INSTALLER_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = INSTALLER_ROOT / "runtime" / "configs" / "requirements.txt"

def install_python_deps():
    if not REQ_FILE.exists():
        sys.exit("[FATAL] requirements.txt missing")

    print("[STEP 5] Installing Python dependencies")

    subprocess.run(
        ["pip", "install", "--upgrade", "pip"],
        check=True
    )

    subprocess.run(
        ["pip", "install", "-r", str(REQ_FILE)],
        check=True
    )

    print("[OK] Python dependencies installed")
