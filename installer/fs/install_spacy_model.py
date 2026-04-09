import subprocess
import sys
from pathlib import Path


def _venv_python() -> str:
    BASE = Path.home() / ".federated"
    if sys.platform == "win32":
        p = BASE / "venv" / "Scripts" / "python.exe"
    else:
        p = BASE / "venv" / "bin" / "python"
    # Fall back to system python if venv isn't ready yet
    return str(p) if p.exists() else sys.executable


def install_spacy_model():
    python_cmd = _venv_python()

    # ── Check if model is already installed in the venv ──────────────────────
    try:
        result = subprocess.run(
            [python_cmd, "-c", "import en_core_web_sm; print(en_core_web_sm.__version__)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if result.returncode == 0:
            print(f"[OK] spaCy model already installed ({result.stdout.strip()})")
            return
    except Exception:
        pass

    print("[STEP] Installing spaCy model", flush=True)

    result = subprocess.run(
        [python_cmd, "-m", "spacy", "download", "en_core_web_sm"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print(result.stdout or "", end="", flush=True)

    if result.returncode != 0:
        print(result.stderr or "", end="", file=sys.stderr)
        raise RuntimeError(f"spaCy model installation failed: {result.stderr.strip()}")

    print("[OK] spaCy model installed", flush=True)