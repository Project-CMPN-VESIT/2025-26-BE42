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
    print("[STEP] Upgrading pip...", flush=True)
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True,
        text=True
    )

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
            process = subprocess.Popen(
                [
                    str(python_path),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    pkg
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # ✅ STREAM OUTPUT SAFELY
            if process.stdout:
                for line in process.stdout:
                    try:
                        print(line.rstrip(), flush=True)
                    except:
                        pass  # GUI-safe

            process.wait()  # ❌ removed timeout (important)

            if process.returncode != 0:
                print(f"[SKIPPED] {pkg}", flush=True)
            else:
                print(f"[OK] {pkg}", flush=True)

        except Exception as e:
            print(f"[ERROR - SKIPPED] {pkg}: {e}", flush=True)

    print("\n[OK] Dependency installation complete", flush=True)