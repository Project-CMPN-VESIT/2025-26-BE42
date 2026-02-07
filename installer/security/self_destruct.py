import os
import shutil
import sys
from pathlib import Path

BASE_DIR = Path.home() / ".federated"


def _secure_delete_file(path: Path):
    try:
        if path.exists():
            size = path.stat().st_size
            with open(path, "wb") as f:
                f.write(os.urandom(size))
            path.unlink()
    except Exception:
        pass


def trigger_self_destruct(reason: str):
    print(f"[SECURITY] SELF-DESTRUCT triggered: {reason}")

    if BASE_DIR.exists():
        paths = []
        for root, dirs, files in os.walk(BASE_DIR, topdown=False):
            for name in files:
                paths.append(Path(root) / name)
            for name in dirs:
                paths.append(Path(root) / name)

        for p in paths:
            try:
                if p.is_file():
                    _secure_delete_file(p)
                else:
                    shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass

        try:
            shutil.rmtree(BASE_DIR)
        except Exception:
            pass

    try:
        os.sync()
    except Exception:
        pass

    sys.exit("[SECURITY] Client terminated")
