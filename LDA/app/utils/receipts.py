# app/utils/receipts.py
import json
import hashlib
import hmac
import base64
from datetime import datetime
from pathlib import Path
import secrets

class ReceiptManager:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        key_path = self.root / "receipt.key"
        if key_path.exists():
            self.hmac_key = base64.b64decode(key_path.read_text())
        else:
            self.hmac_key = secrets.token_bytes(32)
            key_path.write_text(base64.b64encode(self.hmac_key).decode())

    def create_receipt(self, operation: str, input_meta: dict, output_uri: str) -> str:
        """Generates a signed receipt for every preprocessing operation."""
        payload = {
            "operation": operation,
            "input_meta": input_meta,
            "output_uri": output_uri,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = hmac.new(self.hmac_key, payload_bytes, hashlib.sha256).digest()
        payload["signature"] = base64.b64encode(signature).decode()

        # Store receipt
        filename = f"{operation}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
        path = self.root / filename
        path.write_text(json.dumps(payload, indent=2))

        return str(path)

    def verify_receipt(self, receipt_path: str) -> bool:
        """Verifies that a receipt has not been tampered with."""
        payload = json.loads(Path(receipt_path).read_text())
        sig = base64.b64decode(payload.pop("signature"))
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        expected_sig = hmac.new(self.hmac_key, payload_bytes, hashlib.sha256).digest()
        return hmac.compare_digest(sig, expected_sig)
