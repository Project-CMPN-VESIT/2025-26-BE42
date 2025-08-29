import os
import json
import base64
from pathlib import Path
from typing import List
import pandas as pd
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def read_master_key_candidates(master_key_path: str) -> List[bytes]:
    """Try multiple interpretations of the master key file."""
    candidates = []
    raw = Path(master_key_path).read_bytes()
    candidates.append(raw)

    try:
        txt = raw.decode("utf-8").strip()
        if txt:
            candidates.append(txt.encode("utf-8"))
            try:
                candidates.append(base64.b64decode(txt, validate=True))
            except Exception:
                pass
            try:
                candidates.append(bytes.fromhex(txt))
            except Exception:
                pass
    except Exception:
        pass

    seen, uniq = set(), []
    for k in candidates:
        key_sig = (len(k), k[:8])
        if key_sig not in seen:
            seen.add(key_sig)
            uniq.append(k)
    return uniq


def derive_key(master_key: bytes, context: str) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=context.encode() if context is not None else b""
    )
    return hkdf.derive(master_key)


def build_context_candidates(enc_path: Path) -> List[str]:
    contexts = set()
    parent = enc_path.parent
    abs_parent = parent.resolve()

    contexts.add(str(parent))
    contexts.add(str(abs_parent))
    contexts.add(str(parent).rstrip("/"))
    contexts.add(str(abs_parent).rstrip("/"))

    try:
        contexts.add(str(parent.parent))
        contexts.add(str(parent.parent.resolve()))
    except Exception:
        pass

    try:
        rel_cwd = parent.relative_to(Path.cwd())
        contexts.add(str(rel_cwd))
    except Exception:
        pass

    parts = list(abs_parent.parts)
    if "secure_store" in parts:
        idx = parts.index("secure_store")
        tail = Path(*parts[idx:])
        contexts.add(str(tail))
        if len(parts) - idx >= 2:
            contexts.add(str(Path(*parts[idx:-1])))

    contexts.add("")
    return [c for c in contexts if c is not None]


def decrypt_with_candidates(ct: bytes, nonce: bytes,
                            master_keys: List[bytes],
                            contexts: List[str]) -> bytes:
    for mk in master_keys:
        for ctx in contexts:
            key = derive_key(mk, ctx)
            aesgcm = AESGCM(key)
            for aad in [None, b"", ctx.encode() if ctx else b""]:
                try:
                    pt = aesgcm.decrypt(nonce, ct, aad)
                    print("SUCCESS:")
                    print(f"  - master key len: {len(mk)} bytes")
                    print(f"  - context: {repr(ctx)}")
                    print(f"  - aad: {'None' if aad is None else (aad if aad != b'' else 'EMPTY')}")
                    return pt
                except Exception:
                    pass
    raise ValueError("All decryption attempts failed.")


def main():
    encrypted_file = "/home/ritik26/Desktop/BE-Major-Project/LDA/secure_store/sess-1756473835/session/2025-08-29/13.parquet.enc"
    master_key_file = "/home/ritik26/Desktop/BE-Major-Project/LDA/secure_store/master.key"

    p = Path(encrypted_file)
    payload = json.loads(p.read_text())

    try:
        nonce = base64.b64decode(payload["nonce"], validate=True)
    except Exception:
        nonce = payload["nonce"].encode() if isinstance(payload["nonce"], str) else bytes(payload["nonce"])

    try:
        ct = base64.b64decode(payload["ct"], validate=True)
    except Exception:
        ct = payload["ct"].encode() if isinstance(payload["ct"], str) else bytes(payload["ct"])

    master_keys = read_master_key_candidates(master_key_file)
    contexts = build_context_candidates(p)

    try:
        plaintext = decrypt_with_candidates(ct, nonce, master_keys, contexts)

        # Save as parquet
        decrypted_parquet_path = str(p).replace(".enc", "")
        with open(decrypted_parquet_path, "wb") as f:
            f.write(plaintext)
        print(f"\nDecrypted parquet saved to: {decrypted_parquet_path}")

        # Load with pandas
        df = pd.read_parquet(decrypted_parquet_path)
        print("\nPreview of dataset:")
        print(df.head())

        # Save CSV
        csv_path = decrypted_parquet_path.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)
        print(f"\nCSV saved to: {csv_path}")

    except ValueError as e:
        print("\nERROR:", e)


if __name__ == "__main__":
    main()
