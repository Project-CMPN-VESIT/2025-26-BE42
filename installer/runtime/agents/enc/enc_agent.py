import os, json, time, base64
from typing import Optional, Dict, Any

# Optional libs
try:
    from Pyfhel import Pyfhel
    HAS_PYFHEL = True
except Exception:
    HAS_PYFHEL = False

try:
    import boto3
    HAS_BOTO3 = True
except Exception:
    HAS_BOTO3 = False

# Centralized modules
from core.centralised_receipts import CentralReceiptManager
from core.centralized_secure_store import SecureStore

from installer.security.integrity import integrity_guard
integrity_guard()

class EncryptionAgent:
    def __init__(self,
                 final_store_dir: str = "secure_store/final_updates",
                 receipts_dir: str = "receipts",
                 mode: str = "aes",
                 kms_key_id: Optional[str] = None):

        self.mode = mode.lower()
        self.final_store_dir = final_store_dir
        self.receipts_dir = receipts_dir
        os.makedirs(self.final_store_dir, exist_ok=True)
        os.makedirs(self.receipts_dir, exist_ok=True)

        # Centralized SecureStore
        self.store = SecureStore(
            agent="enc-agent",
            root="./secure_store"
        )

        # AWS KMS if needed
        self.kms_key_id = kms_key_id
        if HAS_BOTO3 and self.mode == "kms_envelope":
            self.kms = boto3.client("kms")
        else:
            self.kms = None

        # Homomorphic encryption (Pyfhel CKKS)
        self.pyfhel = None
        if self.mode == "he_ckks":
            if not HAS_PYFHEL:
                raise RuntimeError("Pyfhel not available")
            self.pyfhel = Pyfhel()
            self.pyfhel.contextGen(scheme='CKKS', n=2**14, scale=2**30)
            self.pyfhel.keyGen()

        # Centralized receipts
        self.rm = CentralReceiptManager(agent="enc-agent")

    # ---------------- main entry ----------------
    def process_dp_update(self, dp_receipt_path: str) -> Dict[str, Any]:
        if dp_receipt_path.startswith("file://"):
            dp_receipt_path = dp_receipt_path[len("file://"):]

        with open(dp_receipt_path, "r") as rf:
            dp_receipt = json.load(rf)

        dp_update_uri = dp_receipt.get("outputs", [None])[0]
        if not dp_update_uri or not dp_update_uri.startswith("file://"):
            raise ValueError("dp_receipt must include file:// dp_update_uri")

        meta = {
            "scheme": "AES-GCM-SecureStore",
            "note": "DP update already encrypted at rest"
        }

        receipt = self.rm.create_receipt(
            agent="enc-agent",
            session_id=dp_receipt.get("session_id"),
            operation="finalize_update",
            params={
                "dp_receipt": dp_receipt_path,
                "dp_update_uri": dp_update_uri,
                "encryption_scheme": meta["scheme"],
                "metadata": meta,
            },
            outputs=[dp_update_uri],  # IMPORTANT: same URI
        )

        receipt_uri = self.rm.write_receipt(receipt, out_dir=self.receipts_dir)

        return {
            "receipt": receipt,
            "receipt_uri": receipt_uri,
        }
