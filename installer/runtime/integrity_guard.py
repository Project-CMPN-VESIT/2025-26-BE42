import hashlib
from pathlib import Path
from .self_destruct import trigger_self_destruct

BASE_DIR = Path.home() / ".federated"
BASELINE_FILE = BASE_DIR / "integrity" / "baseline.sha256"

EXCLUDE_PREFIXES = {
    "logs/",
    "data/cache/",
}

def _should_exclude(path: Path) -> bool:
    rel = path.relative_to(BASE_DIR).as_posix()
    return any(rel.startswith(p) for p in EXCLUDE_PREFIXES)

def compute_tree_hash() -> str:
    h = hashlib.sha256()

    for path in sorted(BASE_DIR.rglob("*")):
        if not path.is_file():
            continue
        if _should_exclude(path):
            continue

        rel = path.relative_to(BASE_DIR).as_posix().encode()
        h.update(rel)
        h.update(path.read_bytes())

    return h.hexdigest()

def verify_integrity():
    if not BASELINE_FILE.exists():
        trigger_self_destruct("Integrity baseline missing")

    expected = BASELINE_FILE.read_text().strip()
    current = compute_tree_hash()

    if current != expected:
        trigger_self_destruct("Filesystem integrity violation")
