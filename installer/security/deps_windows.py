import subprocess
import sys

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10

def _run(cmd):
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)

def verify_python_and_pip():
    try:
        out = _run(["python", "--version"])
        version = out.strip().split()[1]
        major, minor, *_ = map(int, version.split("."))

        if major != REQUIRED_MAJOR or minor != REQUIRED_MINOR:
            sys.exit(f"[SECURITY] Python {REQUIRED_MAJOR}.{REQUIRED_MINOR} required")

        _run(["pip", "--version"])

    except Exception as e:
        sys.exit(f"[SECURITY] Python/pip check failed: {e}")

    print("[OK] Python and pip verified")

from pathlib import Path
import sys

BASE = Path.home() / ".federated" / "deps" / "windows"

REQUIRED = [
    BASE / "OpenFace" / "FeatureExtraction.exe",
    BASE / "OpenFace" / "model",
    BASE / "opensmile" / "build" / "progsrc" / "smilextract" / "Release" / "SMILExtract.exe",
    BASE / "opensmile" / "config" / "egemaps" / "v02" / "eGeMAPSv02.conf",
]

def verify_windows_deps():
    for p in REQUIRED:
        if not p.exists():
            sys.exit(f"[INSTALLER] Missing dependency: {p}")

    print("[OK] Windows dependencies verified")
