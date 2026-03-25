import subprocess
import sys

def install_spacy_model():
    import importlib

    try:
        importlib.import_module("en_core_web_sm")
        print("[OK] spaCy model already installed")
        return
    except ImportError:
        pass

    print("[STEP] Installing spaCy model")

    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        check=True
    )