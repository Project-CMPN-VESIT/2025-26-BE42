#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

BASE = Path.home() / ".federated"
DEPS = BASE / "deps"
REQUIRED = [
    "pydantic",
    "yaml",
    "torch",
    "transformers",
    "pandas",
    "pyarrow",
    "cv2",
]

OPTIONAL = [
    "librosa",
    "webrtcvad",
    "spacy",
    "boto3",
    "Pyfhel",
]

def check():
    import importlib

    print("\n[CHECK] Validating dependencies...\n")

    for pkg in REQUIRED:
        try:
            importlib.import_module(pkg)
            print(f"[OK] {pkg}")
        except Exception:
            print(f"[FAIL] {pkg}")
            raise RuntimeError(f"Missing REQUIRED dependency: {pkg}")

    for pkg in OPTIONAL:
        try:
            importlib.import_module(pkg)
            print(f"[OK] {pkg}")
        except Exception:
            print(f"[WARN] Optional missing: {pkg}")

def run(cmd, name):
    print(f"[TEST] {name}")
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            timeout=30
        )
        print("[OK]")
        return out
    except subprocess.CalledProcessError as e:
        print(e.output.decode(errors="ignore"))
        sys.exit(f"[FAIL] {name}")
    except Exception as e:
        sys.exit(f"[FAIL] {name}: {e}")

def main():
    # --------------------------------------------------
    # 1. Python sanity
    # --------------------------------------------------
    run(["python", "--version"], "Python available")

    # --------------------------------------------------
    # 2. OpenFace binary check
    # --------------------------------------------------
    openface = DEPS / "windows" / "OpenFace" / "FeatureExtraction.exe"
    if not openface.exists():
        sys.exit("[FAIL] FeatureExtraction.exe missing")

    run(
        [str(openface), "-h"],
        "OpenFace executable runs"
    )

    # --------------------------------------------------
    # 3. openSMILE binary check
    # --------------------------------------------------
    smile = list((DEPS / "windows" / "opensmile"/ "build" / "progsrc" / "smilextract" / "Release").rglob("SMILExtract.exe"))
    if not smile:
        sys.exit("[FAIL] SMILExtract.exe missing")

    run(
        [str(smile[0]), "-h"],
        "openSMILE executable runs"
    )
    
    print("\n[ALL CHECKS PASSED] Runtime dependencies validated")

if __name__ == "__main__":
    main()
 