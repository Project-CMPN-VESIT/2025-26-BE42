import subprocess
import sys
from pathlib import Path

INSTALLER_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = INSTALLER_ROOT / "runtime" / "configs" / "requirements.txt"

def install_python_deps():
    BASE = Path.home() / ".federated"
    VENV_DIR = BASE / "venv"
    python_path = VENV_DIR / "Scripts" / "python.exe"

    print("[STEP] Installing dependencies (robust mode)", flush=True)

    # ✅ Upgrade pip (WITH OUTPUT)
    res = subprocess.run(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print(res.stdout)
    print(res.stderr)

    if res.returncode != 0:
        raise RuntimeError("pip upgrade failed")

    if not REQ_FILE.exists():
        raise RuntimeError("requirements.txt not found")

    with open(REQ_FILE, "r") as f:
        packages = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    for pkg in packages:
        print(f"\n[INSTALL] {pkg}", flush=True)

        try:
            result = subprocess.run(
                [
                    str(python_path),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--force-reinstall",
                    "--no-deps",
                    pkg
                ],
                capture_output=True,
                text=True
            )

            for line in result.stdout:
                print(line.strip())

            result.wait()

            if result.returncode != 0:
                print(f"[SKIPPED] {pkg}")
            else:
                print(f"[OK] {pkg}")

        except Exception as e:
            print(f"[ERROR - SKIPPED] {pkg}: {e}", flush=True)

    print("\n[OK] Dependency installation complete", flush=True)
    sys.stdout.flush()