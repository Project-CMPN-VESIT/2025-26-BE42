# app/security/secure_store.py
import os
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
import base64
import json
import secrets

class SecureStore:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        key_path = self.root / "master.key"
        if key_path.exists():
            self.master_key = base64.b64decode(key_path.read_text())
        else:
            self.master_key = secrets.token_bytes(32)
            key_path.write_text(base64.b64encode(self.master_key).decode())

    def _derive_key(self, context: str) -> bytes:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=context.encode(),
        )
        return hkdf.derive(self.master_key)

    def encrypt_write(self, relative_path: str, data: bytes) -> str:
        out_path = self.root / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(self._derive_key(str(out_path.parent)))
        ciphertext = aesgcm.encrypt(nonce, data, None)
        payload = {"nonce": base64.b64encode(nonce).decode(), "ct": base64.b64encode(ciphertext).decode()}
        out_path.write_text(json.dumps(payload))
        return f"file://{out_path}"

    def decrypt_read(self, uri: str) -> bytes:
        assert uri.startswith("file://"), "Invalid URI scheme"
        p = Path(uri[len("file://"):])
        payload = json.loads(p.read_text())
        nonce = base64.b64decode(payload["nonce"])
        ct = base64.b64decode(payload["ct"])
        aesgcm = AESGCM(self._derive_key(str(p.parent)))
        return aesgcm.decrypt(nonce, ct, None)
