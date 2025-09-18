# centralized_receipts.py
import json
import hmac
import hashlib
import base64
import secrets
import uuid
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


class CentralReceiptManager:
    """
    Centralized Receipt Manager for all agents.
    Provides HMAC-based signing, writing, and verification of receipts.
    """

    def __init__(self, agent: Optional[str] = None,
                 key_source: Optional[str] = None, key_size: int = 32):
        self.agent = agent
        # --- existing key load logic ---
        if key_source is None:
            key_path = Path.home() / ".local_data_agent_receipt_key"
            if key_path.exists():
                self.hmac_key = base64.b64decode(key_path.read_text())
            else:
                self.hmac_key = secrets.token_bytes(key_size)
                key_path.write_text(base64.b64encode(self.hmac_key).decode())
        elif key_source.startswith("env:"):
            env_var = key_source.split(":", 1)[1]
            self.hmac_key = base64.b64decode(os.environ[env_var])
        elif key_source.startswith("file:"):
            path = Path(key_source.split(":", 1)[1])
            self.hmac_key = base64.b64decode(path.read_text())
        else:
            raise ValueError("Unsupported key_source format")

    def create_receipt(
        self,
        agent: Optional[str],
        operation: str,
        params: Dict[str, Any],
        outputs: List[str],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a standardized receipt payload.
        If agent not given here, fallback to self.agent.
        """
        payload = {
            "agent": agent or self.agent or "unknown",
            "session_id": session_id or f"sess-{uuid.uuid4().hex}",
            "operation": operation,
            "params": params,
            "outputs": outputs,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        return self.sign(payload)

    def sign(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns payload with a base64 signature.
        """
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        sig = hmac.new(self.hmac_key, payload_bytes, hashlib.sha256).digest()
        return {**payload, "signature": base64.b64encode(sig).decode()}

    def verify(self, receipt_path: str) -> bool:
        """
        Verify a receipt file.
        """
        p = Path(receipt_path)
        data = json.loads(p.read_text())
        sig_b64 = data.pop("signature", None)
        if not sig_b64:
            return False
        expected = hmac.new(
            self.hmac_key,
            json.dumps(data, sort_keys=True).encode(),
            hashlib.sha256
        ).digest()
        return hmac.compare_digest(expected, base64.b64decode(sig_b64))

    def write_receipt(
        self,
        payload: Dict[str, Any],
        out_dir: str = "./receipts",
        use_uuid: bool = True
    ) -> str:
        """
        Save a signed receipt to disk and return file:// URI.
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fname = (
            f"receipt_{uuid.uuid4().hex}.json"
            if use_uuid
            else f"{payload['operation']}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
        )
        path = Path(out_dir) / fname
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return f"file://{path.resolve()}"
