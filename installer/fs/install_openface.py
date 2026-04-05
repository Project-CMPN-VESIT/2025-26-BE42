"""
install_openface.py

BUG-4 FIX: SRC was pointing to the opensmile directory instead of OpenFace.
           Fixed: SRC now correctly points to the OpenFace payload.
"""

import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path.home() / ".federated"
DST  = BASE / "deps" / "windows" / "OpenFace"


def get_installer_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


INSTALLER_ROOT = get_installer_root()

# BUG-4 FIX: was pointing to "opensmile" — corrected to "OpenFace"
SRC = INSTALLER_ROOT / "runtime" / "deps" / "windows" / "OpenFace"


def install_openface():
    if DST.exists():
        print("[INFO] OpenFace already installed")
        return

    if not SRC.exists():
        sys.exit(f"[FATAL] OpenFace payload missing at: {SRC}")

    print("[STEP] Installing OpenFace")

    shutil.copytree(SRC, DST)

    # Run model download script if present (Windows only)
    ps1 = DST / "download_models.ps1"
    if ps1.exists():
        subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy", "Bypass",
                "-File", str(ps1)
            ],
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

    exe = DST / "FeatureExtraction.exe"
    if not exe.exists():
        sys.exit("[FATAL] FeatureExtraction.exe missing after OpenFace install")

    print("[OK] OpenFace installed")