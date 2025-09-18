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
from centralised_receipts import CentralReceiptManager
from centralized_secure_store import SecureStore


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
        self.store = SecureStore("./secure_store")

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
        self.rm = CentralReceiptManager()

    # ---------------- main entry ----------------
    def process_dp_update(self, dp_receipt_path: str) -> Dict[str, Any]:
        if dp_receipt_path.startswith("file://"):
            dp_receipt_path = dp_receipt_path[len("file://"):]

        with open(dp_receipt_path, "r") as rf:
            dp_receipt = json.load(rf)

        dp_update_uri = dp_receipt.get("outputs", [None])[0]
        if not dp_update_uri or not dp_update_uri.startswith("file://"):
            raise ValueError("dp_receipt must include file:// dp_update_uri")

        # Read raw DP update (decrypted via SecureStore)
        dp_bytes = self.store.decrypt_read(dp_update_uri)

        # Encrypt based on chosen mode
        if self.mode == "aes":
            ts = int(time.time() * 1000)
            final_uri = f"file://{os.path.join(self.final_store_dir, f'encdp_{ts}.pt.enc')}"
            self.store.encrypt_write(final_uri, dp_bytes)
            meta = {"scheme": "AES-GCM-SecureStore"}

        elif self.mode == "kms_envelope":
            if not HAS_BOTO3 or self.kms is None:
                raise RuntimeError("boto3/AWS KMS not configured")
            resp = self.kms.generate_data_key(KeyId=self.kms_key_id, KeySpec='AES_256')
            data_key_ciphertext = resp['CiphertextBlob']

            ts = int(time.time() * 1000)
            final_uri = f"file://{os.path.join(self.final_store_dir, f'encdp_kms_{ts}.pt.enc')}"
            self.store.encrypt_write(final_uri, dp_bytes)

            meta = {
                "scheme": "KMS-Envelope",
                "wrapped_key": base64.b64encode(data_key_ciphertext).decode('utf-8'),
                "kms_key_id": self.kms_key_id,
            }

        elif self.mode == "he_ckks":
            if not self.pyfhel:
                raise RuntimeError("Pyfhel not initialized")
            import numpy as np
            arr = np.frombuffer(dp_bytes, dtype=np.float32)
            ptxt = self.pyfhel.encodeFrac(arr.tolist())
            ctxt = self.pyfhel.encryptPtxt(ptxt)
            ctxt_bytes = ctxt.to_bytes()

            ts = int(time.time() * 1000)
            final_path = os.path.join(self.final_store_dir, f"encdp_ckks_{ts}.bin")
            with open(final_path, "wb") as wf:
                wf.write(ctxt_bytes)
            final_uri = "file://" + final_path
            meta = {"scheme": "CKKS-Pyfhel"}

        elif self.mode == "smpc":
            N = 3
            shares = []
            prev = dp_bytes
            for i in range(N - 1):
                r = os.urandom(len(dp_bytes))
                shares.append(base64.b64encode(r).decode('utf-8'))
                prev = bytes(x ^ y for x, y in zip(prev, r))
            shares.append(base64.b64encode(prev).decode('utf-8'))

            ts = int(time.time() * 1000)
            final_path = os.path.join(self.final_store_dir, f"encdp_shares_{ts}.json")
            with open(final_path, "w") as wf:
                json.dump({"shares": shares, "scheme": "XOR-secret-sharing-demo"}, wf)
            final_uri = "file://" + final_path
            meta = {"scheme": "XOR-secret-sharing-demo"}

        else:
            raise ValueError(f"Unknown encryption mode: {self.mode}")

        # Centralized receipt
        receipt = self.rm.create_receipt(
            agent="enc-agent",
            session_id=dp_receipt.get("session_id"),
            operation="encrypt_update",
            params={
                "dp_receipt": dp_receipt_path,
                "dp_update_uri": dp_update_uri,
                "encryption_scheme": meta.get("scheme"),
                "metadata": meta,
            },
            outputs=[final_uri],
        )
        receipt_uri = self.rm.write_receipt(receipt, out_dir=self.receipts_dir)

        return {"receipt": receipt, "receipt_uri": receipt_uri}
