# centralized_secure_store.py
import os
import json
import base64
import secrets
from pathlib import Path
from typing import Union
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


class SecureStore:
    """
    Centralized AES-GCM encrypted store.
    Keys are derived deterministically from (agent + parent directory path)
    using HKDF, so each agent has an isolated key namespace while still
    sharing the same master key material.
    """

    def __init__(
        self,
        agent: str = "generic",
        root: Union[str, Path] = "./secure_store",
        key_path: Union[str, Path] = None
    ):
        self.agent = agent
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if key_path is None:
            key_path = self.root / "master.key"
        self.key_path = Path(key_path)
        self.master_key = self._load_or_create_master_key()
        print("[DEBUG STORE]", self.agent)

    def _load_or_create_master_key(self) -> bytes:
        """Load or generate a global master key (shared across all agents)."""
        if self.key_path.exists():
            txt = self.key_path.read_text().strip()
            try:
                return base64.b64decode(txt)
            except Exception:
                return self.key_path.read_bytes()
        else:
            k = secrets.token_bytes(32)
            self.key_path.write_text(base64.b64encode(k).decode())
            return k

    def _derive_key(self, context: str) -> bytes:
        """
        Derive a per-agent, per-directory key using HKDF.
        Context normalization ensures identical keys for same session across agents.
        """
        # Normalize context to avoid mismatch between trainer and DP agent
        if "local_updates" in context:
            context = context.split("local_updates")[0].rstrip("/")

        info = f"{self.agent}:{context}".encode()
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=info,
            backend=default_backend(),
        )
        return hkdf.derive(self.master_key)

    # ------------------ write ------------------
    def encrypt_write(self, uri: str, data: bytes) -> str:
        assert uri.startswith("file://"), "URI must start with file://"
        p = Path(uri[len("file://"):])
        p.parent.mkdir(parents=True, exist_ok=True)

        print("[DEBUG WRITE FILE]", p.resolve())
        print("[DEBUG WRITE SIZE]", len(data))

        context = str(Path(uri[len("file://"):]).parent.name)
        if "local_updates" in context:
            context = context.split("local_updates")[0].rstrip("/")

        key = self._derive_key(context)
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ct = aesgcm.encrypt(nonce, data, None)

        payload = {
            "nonce": base64.b64encode(nonce).decode(),
            "ct": base64.b64encode(ct).decode(),
        }
        p.write_text(json.dumps(payload))
        return uri  # return the file:// URI for consistency

    # ------------------ read ------------------
    def decrypt_read(self, uri: str) -> bytes:
        assert uri.startswith("file://"), "URI must start with file://"
        p = Path(uri[len("file://"):])
        print("[DEBUG READ FILE]", p.resolve())
        print("[DEBUG FILE EXISTS]", p.exists())
        print("[DEBUG FILE SIZE]", p.stat().st_size if p.exists() else "NA")

        payload = json.loads(p.read_text())
        nonce = base64.b64decode(payload["nonce"])
        ct = base64.b64decode(payload["ct"])

        context_path = p.parent.resolve()
        root_path = self.root.resolve()

        context = str(Path(uri[len("file://"):]).parent.name)
        if "local_updates" in context:
            context = context.split("local_updates")[0].rstrip("/")

        key = self._derive_key(context)
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, None)
