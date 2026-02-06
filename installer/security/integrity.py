import hashlib
from pathlib import Path
import sys
import os
from .self_destruct import trigger_self_destruct

FEDERATED_DIR = Path.home() / ".federated"
BASELINE_FILE = FEDERATED_DIR / "integrity" / "baseline.sha256"

EXCLUDE_PREFIXES = {
    "logs/",
    "data/cache/",
}

def _should_exclude(path: Path) -> bool:
    rel = path.relative_to(FEDERATED_DIR).as_posix()
    return any(rel.startswith(e) for e in EXCLUDE_PREFIXES)

def compute_tree_hash(root: Path) -> str:
    h = hashlib.sha256()

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _should_exclude(path):
            continue

        rel = path.relative_to(root).as_posix().encode()
        h.update(rel)
        h.update(path.read_bytes())

    return h.hexdigest()

def write_baseline():
    digest = compute_tree_hash(FEDERATED_DIR)
    BASELINE_FILE.write_text(digest)
    os.chmod(BASELINE_FILE, 0o600)
    print("[OK] Integrity baseline written")

def verify_integrity():
    if not BASELINE_FILE.exists():
        trigger_self_destruct("[SECURITY] Missing baseline")

    current = compute_tree_hash(FEDERATED_DIR)
    stored = BASELINE_FILE.read_text().strip()

    if current != stored:
        trigger_self_destruct("[SECURITY] Integrity violation detected")
        

    return True

def integrity_guard():
    """
    Runtime integrity enforcement.
    Call this before any sensitive operation.
    """
    verify_integrity()
