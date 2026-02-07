#!/usr/bin/env python3
import sys
import json
from pathlib import Path

from centralized_secure_store import SecureStore

USAGE = """
Usage:
  python decrypt_secure_store.py <secure_store_root> <output_dir>

IMPORTANT:
- Run from the SAME working directory as encryption
- DO NOT use .resolve()
- DO NOT move secure_store
"""

KNOWN_AGENTS = [
    "generic",
    "client",
    "trainer-agent",
    "lda-audio",
    "lda-session-processor",
    "lda-video-processor",
    "lda-text-processor",
    "local-data-agent",
    "dp-agent",
    "encryption-agent",
]

def decrypt_all(src_root: Path, dst_root: Path):
    print(f"[info] secure_store = {src_root}")
    print(f"[info] output       = {dst_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    for enc_file in src_root.rglob("*.enc"):
        rel = enc_file.relative_to(src_root)
        out_path = dst_root / rel.with_suffix("")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        uri = f"file://{enc_file}"

        decrypted = None
        used_agent = None

        for agent in KNOWN_AGENTS:
            try:
                store = SecureStore(
                    agent=agent,
                    root=src_root,                  # ❗ NO resolve()
                    key_path=src_root / "master.key"
                )
                decrypted = store.decrypt_read(uri)
                used_agent = agent
                break
            except Exception:
                continue

        if decrypted is None:
            print(f"[FAIL] {enc_file}: no valid agent")
            continue

        try:
            text = decrypted.decode("utf-8")
            if text.strip().startswith("{") or text.strip().startswith("["):
                out_path.write_text(
                    json.dumps(json.loads(text), indent=2)
                )
            else:
                out_path.write_text(text)
        except Exception:
            out_path.write_bytes(decrypted)

        print(f"[OK] {rel} (agent={used_agent})")

    print("\n✅ Decryption complete")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(USAGE)
        sys.exit(1)

    decrypt_all(Path(sys.argv[1]), Path(sys.argv[2]))
