import subprocess
import sys
from pathlib import Path

INSTALLER_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = INSTALLER_ROOT / "runtime" / "configs" / "requirements.txt"


def _venv_python() -> Path:
    BASE = Path.home() / ".federated"
    if sys.platform == "win32":
        return BASE / "venv" / "Scripts" / "python.exe"
    return BASE / "venv" / "bin" / "python"


def install_python_deps():
    python_path = _venv_python()

    print("[STEP] Installing dependencies (robust mode)", flush=True)

    # ── Upgrade pip (capture output so GUI log receives it) ──────────────────
    res = subprocess.run(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(res.stdout or "", end="", flush=True)
    if res.returncode != 0:
        print(res.stderr or "", end="", file=sys.stderr)
        raise RuntimeError("pip upgrade failed")

    if not REQ_FILE.exists():
        raise RuntimeError(f"requirements.txt not found at {REQ_FILE}")

    with open(REQ_FILE, "r") as f:
        packages = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    failed = []

    for pkg in packages:
        print(f"\n[INSTALL] {pkg}", flush=True)

        try:
            result = subprocess.run(
                [str(python_path), "-m", "pip", "install", pkg],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # BUG-FIX: result.stdout was None before because check=True was used
            # without capture_output=True.  Now we always capture and print.
            print(result.stdout or "", end="", flush=True)

            if result.returncode != 0:
                print(f"[SKIPPED] {pkg}: {result.stderr.strip()}", flush=True)
                failed.append(pkg)
            else:
                print(f"[OK] {pkg}", flush=True)

        except Exception as e:
            print(f"[ERROR - SKIPPED] {pkg}: {e}", flush=True)
            failed.append(pkg)

    if failed:
        print(f"\n[WARN] {len(failed)} package(s) failed to install: {failed}", flush=True)

    print("\n[OK] Dependency installation complete", flush=True)
    sys.stdout.flush()