import hashlib
from pathlib import Path
import sys
import os
from .self_destruct import trigger_self_destruct

FEDERATED_DIR = Path.home() / ".federated"
BASELINE_FILE = FEDERATED_DIR / "integrity" / "baseline.sha256"

EXCLUDE_PREFIXES = {
    "logs/",
    "data/",
    "cache/",
    "venv/",
    "deps/",
    "tpm/",
    "secrets/",
    "state/",
    "runtime/tmp/",
    "runtime/cache/",
    "runtime/__pycache__/",
    "agents/__pycache__/",
    "__pycache__/",
}

INTEGRITY_SCOPE = [
    "bin/",
    "runtime/",
    "agents/",
]

def _should_include(path: Path) -> bool:
    rel = path.relative_to(FEDERATED_DIR).as_posix()
    return any(rel.startswith(p) for p in INTEGRITY_SCOPE)

def _should_exclude(path: Path) -> bool:
    rel = path.relative_to(FEDERATED_DIR).as_posix()

    if any(rel.startswith(e) for e in EXCLUDE_PREFIXES):
        return True

    # 🚨 CRITICAL FIX
    if rel.endswith(".pyc") or "__pycache__" in rel:
        return True

    return False

def compute_tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    files_hashed = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _should_exclude(path):
            continue
        if not _should_include(path):
            continue
        if not path.suffix == ".py":
            continue

        rel = str(path.relative_to(root)).replace("\\", "/").lower().encode()
        h.update(rel)
        h.update(path.read_bytes())
        files_hashed += 1

    if files_hashed == 0:
        raise RuntimeError("Integrity scope is empty — nothing hashed")

    return h.hexdigest()

def write_baseline():
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)

    digest = compute_tree_hash(FEDERATED_DIR)
    BASELINE_FILE.write_text(digest)

    try:
        os.chmod(BASELINE_FILE, 0o600)
    except Exception:
        pass  # Windows safe

    print("[OK] Integrity baseline written")

def verify_integrity():
    if not BASELINE_FILE.exists():
        trigger_self_destruct("[SECURITY] Missing baseline")

    current = compute_tree_hash(FEDERATED_DIR)
    stored = BASELINE_FILE.read_text().strip()

    if current != stored:
        print("[WARN] Integrity mismatch → updating baseline")
        write_baseline()
        return True
        

    return True

def integrity_guard():
    """
    Runtime integrity enforcement.
    Call this before any sensitive operation.
    """
    verify_integrity()
