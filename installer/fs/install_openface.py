import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path.home() / ".federated"
DST = BASE / "deps" / "windows" / "OpenFace"

def get_installer_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]

INSTALLER_ROOT = get_installer_root()
SRC = INSTALLER_ROOT / "runtime" / "deps" / "windows" / "opensmile"

def install_openface():
    if DST.exists():
        print("[INFO] OpenFace already installed")
        return

    if not SRC.exists():
        sys.exit("[FATAL] OpenFace payload missing")

    print("[STEP 5] Installing OpenFace")

    shutil.copytree(SRC, DST)

    ps1 = DST / "download_models.ps1"
    if ps1.exists():
        subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy", "Bypass",
                "-File", str(ps1)
            ],
            check=True
        )

    exe = DST / "FeatureExtraction.exe"
    if not exe.exists():
        sys.exit("[FATAL] FeatureExtraction.exe missing")

    print("[OK] OpenFace installed")
