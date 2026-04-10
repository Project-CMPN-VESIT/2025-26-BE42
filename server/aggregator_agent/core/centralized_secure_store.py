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

# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL PATHS — single source of truth for the whole system
# All agents MUST derive from these regardless of their working directory.
# ─────────────────────────────────────────────────────────────────────────────
_FEDERATED_BASE  = Path.home() / ".federated"
_CANONICAL_ROOT  = _FEDERATED_BASE / "data" / "secure_store"
_GLOBAL_KEY_PATH = _CANONICAL_ROOT / "master.key"       # ONE key for all agents


class SecureStore:
    """
    Centralized AES-GCM encrypted store.

    master.key is ALWAYS stored at _GLOBAL_KEY_PATH so that every agent
    instance — regardless of the `root` argument — encrypts / decrypts with
    the same underlying secret.  Per-agent key isolation is achieved via HKDF
    with an (agent, context) info tag, NOT by using separate master keys.
    """

    def __init__(
        self,
        agent: str = "generic",
        root: Union[str, Path] = None,
        key_path: Union[str, Path] = None,
    ):
        self.agent = agent

        # ── root (where encrypted files live) ────────────────────────────────
        if root is None:
            root = _CANONICAL_ROOT
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

        # ── master key (ALWAYS canonical — this is the critical fix) ─────────
        if key_path is None:
            key_path = _GLOBAL_KEY_PATH
        self.key_path = Path(key_path).expanduser().resolve()
        self.key_path.parent.mkdir(parents=True, exist_ok=True)

        self.master_key = self._load_or_create_master_key()

    # ─────────────────────────────────────────────────────────────────────────
    # Key management
    # ─────────────────────────────────────────────────────────────────────────

    def _load_or_create_master_key(self) -> bytes:
        """Load or generate the global master key (shared across all agents)."""
        if self.key_path.exists():
            txt = self.key_path.read_text().strip()
            try:
                return base64.b64decode(txt)
            except Exception:
                return self.key_path.read_bytes()
        else:
            k = secrets.token_bytes(32)
            self.key_path.write_text(base64.b64encode(k).decode())
            try:
                os.chmod(self.key_path, 0o600)
            except Exception:
                pass  # Windows may not support chmod
            return k

    def _derive_key(self, context: str) -> bytes:
        """
        Derive a per-agent, per-directory key using HKDF so that each agent
        still has an isolated key namespace even though they share one master.
        """
        info = f"{self.agent}:{context}".encode()
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=info,
            backend=default_backend(),
        )
        return hkdf.derive(self.master_key)

    # ─────────────────────────────────────────────────────────────────────────
    # Context helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _uri_to_context(uri: str) -> str:
        """
        Derive a stable HKDF context from a file URI.

        Rule: use the immediate parent directory name of the file.
        Special-case: anything inside a 'local_updates' directory collapses
        to an empty string so that writer (trainer) and reader (dp_agent) agree
        even when called with slightly different sub-paths.
        """
        parent_name = Path(uri[len("file://"):]).parent.name
        if "local_updates" in parent_name:
            return ""
        return parent_name

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def encrypt_write(self, uri: str, data: bytes) -> str:
        assert uri.startswith("file://"), "URI must start with file://"
        assert data, "Refusing to encrypt empty payload"

        p = Path(uri[len("file://"):]).resolve()

        if not str(p).startswith(str(self.root)):
            raise ValueError("Access outside secure store is not allowed")
        p.parent.mkdir(parents=True, exist_ok=True)

        context = self._uri_to_context(uri)
        key = self._derive_key(context)
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ct = aesgcm.encrypt(nonce, data, None)

        payload = {
            "agent":   self.agent,
            "context": context,
            "nonce":   base64.b64encode(nonce).decode(),
            "ct":      base64.b64encode(ct).decode(),
        }
        p.write_text(json.dumps(payload))
        return uri

    def decrypt_read(self, uri: str) -> bytes:
        assert uri.startswith("file://"), "URI must start with file://"
        p = Path(uri[len("file://"):]).resolve()

        if not str(p).startswith(str(self.root)):
            raise ValueError("Access outside secure store is not allowed")

        raw = json.loads(p.read_text())
        nonce = base64.b64decode(raw["nonce"])
        ct    = base64.b64decode(raw["ct"])

        # Use the context that was stored at write time for exact key reproduction
        stored_context = raw.get("context", self._uri_to_context(uri))
        key = self._derive_key(stored_context)
        aesgcm = AESGCM(key)

        try:
            return aesgcm.decrypt(nonce, ct, None)
        except Exception:
            raise ValueError(
                f"SecureStore: decryption failed for {p}\n"
                f"  agent={self.agent!r}  context={stored_context!r}\n"
                f"  master_key_path={self.key_path}\n"
                "  Likely cause: encrypted by a different master.key."
            )