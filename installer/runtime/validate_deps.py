#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

BASE = Path.home() / ".federated"
DEPS = BASE / "deps"

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
    openface = DEPS / "OpenFace" / "FeatureExtraction.exe"
    if not openface.exists():
        sys.exit("[FAIL] FeatureExtraction.exe missing")

    run(
        [str(openface), "-h"],
        "OpenFace executable runs"
    )

    # --------------------------------------------------
    # 3. openSMILE binary check
    # --------------------------------------------------
    smile = list((DEPS / "opensmile").rglob("SMILExtract.exe"))
    if not smile:
        sys.exit("[FAIL] SMILExtract.exe missing")

    run(
        [str(smile[0]), "-h"],
        "openSMILE executable runs"
    )

    print("\n[ALL CHECKS PASSED] Runtime dependencies validated")

if __name__ == "__main__":
    main()
