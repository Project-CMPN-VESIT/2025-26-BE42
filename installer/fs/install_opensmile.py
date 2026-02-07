import shutil
import sys
from pathlib import Path

BASE = Path.home() / ".federated"
DST = BASE / "deps" / "opensmile"

INSTALLER_ROOT = Path(__file__).resolve().parents[1]
SRC = INSTALLER_ROOT / "runtime" / "deps" / "windows" / "opensmile"

def install_opensmile():
    if DST.exists():
        print("[INFO] openSMILE already installed")
        return

    if not SRC.exists():
        sys.exit("[FATAL] openSMILE payload missing")

    print("[STEP 5] Installing openSMILE")
    shutil.copytree(SRC, DST)

    exe = list(DST.rglob("SMILExtract.exe"))
    if not exe:
        sys.exit("[FATAL] SMILExtract.exe not found")

    print("[OK] openSMILE installed")
