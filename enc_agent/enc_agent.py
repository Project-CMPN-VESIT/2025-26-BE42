# enc_agent/enc_agent.py
import os, json, time, base64
from typing import Optional, Dict, Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.fernet import Fernet

# Optional libs
try:
    from Pyfhel import Pyfhel, PyCtxt
    HAS_PYFHEL = True
except Exception:
    HAS_PYFHEL = False

try:
    import boto3
    HAS_BOTO3 = True
except Exception:
    HAS_BOTO3 = False

# 👇 centralized receipts
from ..centralised_receipts import CentralReceiptManager


class EncryptionAgent:
    def __init__(self,
                 final_store_dir: str = "secure_store/final_updates",
                 receipts_dir: str = "receipts",
                 mode: str = "aes",
                 symmetric_key: Optional[bytes] = None,
                 kms_key_id: Optional[str] = None):

        self.mode = mode.lower()
        self.final_store_dir = final_store_dir
        self.receipts_dir = receipts_dir
        os.makedirs(self.final_store_dir, exist_ok=True)
        os.makedirs(self.receipts_dir, exist_ok=True)

        # Fernet
        if self.mode == "fernet":
            if symmetric_key is None:
                if os.path.exists("keys/fernet.key"):
                    with open("keys/fernet.key", "rb") as f:
                        symmetric_key = f.read().strip()
                else:
                    os.makedirs("keys", exist_ok=True)
                    symmetric_key = Fernet.generate_key()
                    with open("keys/fernet.key", "wb") as f:
                        f.write(symmetric_key)
            self.fernet = Fernet(symmetric_key)
        else:
            self.fernet = None

        # AES-GCM
        if symmetric_key is None and self.mode == "aes":
            symmetric_key = AESGCM.generate_key(bit_length=256)
        self.aes_key = symmetric_key

        # fallback Fernet
        if symmetric_key is None and self.mode == "fernet":
            symmetric_key = Fernet.generate_key()
        self.fernet = Fernet(symmetric_key) if (self.mode == "fernet" and symmetric_key is not None) else None

        self.kms_key_id = kms_key_id

        if HAS_BOTO3 and self.mode == "kms_envelope":
            self.kms = boto3.client("kms")
        else:
            self.kms = None

        # Pyfhel
        self.pyfhel = None
        if self.mode == "he_ckks":
            if not HAS_PYFHEL:
                raise RuntimeError("Pyfhel not available")
            self.pyfhel = Pyfhel()
            self.pyfhel.contextGen(scheme='CKKS', n=2**14, scale=2**30)
            self.pyfhel.keyGen()

        # 👇 centralized receipts
        self.rm = CentralReceiptManager()

    # ---------------- symmetric AES-GCM ----------------
    def encrypt_aes_gcm(self, plaintext: bytes) -> Dict[str, Any]:
        aes = AESGCM(self.aes_key)
        nonce = os.urandom(12)
        ct = aes.encrypt(nonce, plaintext, associated_data=None)
        return {"ciphertext": ct, "nonce": nonce, "scheme": "AES-GCM-256"}

    def decrypt_aes_gcm(self, nonce: bytes, ciphertext: bytes) -> bytes:
        aes = AESGCM(self.aes_key)
        return aes.decrypt(nonce, ciphertext, associated_data=None)

    # ---------------- fernet ----------------
    def encrypt_fernet(self, plaintext: bytes) -> Dict[str, Any]:
        token = self.fernet.encrypt(plaintext)
        return {"ciphertext": token, "nonce": None, "scheme": "Fernet"}

    def decrypt_fernet(self, token: bytes) -> bytes:
        return self.fernet.decrypt(token)

    # ---------------- KMS ----------------
    def encrypt_kms_envelope(self, plaintext: bytes) -> Dict[str, Any]:
        if not HAS_BOTO3 or self.kms is None:
            raise RuntimeError("boto3/AWS KMS not configured")

        resp = self.kms.generate_data_key(KeyId=self.kms_key_id, KeySpec='AES_256')
        data_key_plain = resp['Plaintext']
        data_key_ciphertext = resp['CiphertextBlob']

        aes = AESGCM(data_key_plain)
        nonce = os.urandom(12)
        ct = aes.encrypt(nonce, plaintext, associated_data=None)
        return {
            "ciphertext": ct,
            "nonce": nonce,
            "scheme": "KMS-Envelope-AES-GCM",
            "wrapped_key": base64.b64encode(data_key_ciphertext).decode('utf-8'),
            "kms_key_id": self.kms_key_id
        }

    def decrypt_kms_envelope(self, nonce: bytes, ciphertext: bytes, wrapped_key_b64: str) -> bytes:
        if not HAS_BOTO3 or self.kms is None:
            raise RuntimeError("boto3/AWS KMS not configured")
        wrapped = base64.b64decode(wrapped_key_b64)
        resp = self.kms.decrypt(CiphertextBlob=wrapped)
        data_key_plain = resp['Plaintext']
        aes = AESGCM(data_key_plain)
        return aes.decrypt(nonce, ciphertext, associated_data=None)

    # ---------------- Homomorphic encryption ----------------
    def encrypt_he_ckks(self, plaintext: bytes) -> Dict[str, Any]:
        if not self.pyfhel:
            raise RuntimeError("Pyfhel not initialized")
        import numpy as np
        arr = np.frombuffer(plaintext, dtype=np.float32)
        ptxt = self.pyfhel.encodeFrac(arr.tolist())
        ctxt = self.pyfhel.encryptPtxt(ptxt)
        ctxt_bytes = ctxt.to_bytes()
        return {"ciphertext": ctxt_bytes, "scheme": "CKKS-Pyfhel", "params": "pyfhel-ctx-placeholder"}

    def decrypt_he_ckks(self, ctxt_bytes: bytes) -> bytes:
        if not self.pyfhel:
            raise RuntimeError("Pyfhel not initialized")
        ctxt = PyCtxt(pyfhel=self.pyfhel, bytestring=ctxt_bytes)
        ptxt = self.pyfhel.decryptFrac(ctxt)
        import numpy as np
        arr = np.array(ptxt, dtype=np.float32)
        return arr.tobytes()

    # ---------------- SMPC demo ----------------
    def create_smpc_shares(self, plaintext: bytes) -> Dict[str, Any]:
        N = 3
        shares = []
        prev = plaintext
        for i in range(N - 1):
            r = os.urandom(len(plaintext))
            shares.append(base64.b64encode(r).decode('utf-8'))
            prev = bytes(x ^ y for x, y in zip(prev, r))
        shares.append(base64.b64encode(prev).decode('utf-8'))
        return {"shares": shares, "scheme": "XOR-secret-sharing-demo", "num_shares": N}

    # ---------------- main entry ----------------
    def process_dp_update(self, dp_receipt_path: str) -> Dict[str, Any]:
        with open(dp_receipt_path, "r") as rf:
            dp_receipt = json.load(rf)

        dp_update_uri = dp_receipt.get("local_update_uri")
        if not dp_update_uri or not dp_update_uri.startswith("file://"):
            raise ValueError("dp_receipt must include file:// local_update_uri")

        dp_path = dp_update_uri[len("file://"):]
        with open(dp_path, "rb") as f:
            dp_bytes = f.read()

        # Choose algorithm
        if self.mode == "aes":
            res = self.encrypt_aes_gcm(dp_bytes)
            ciphertext, meta = res["ciphertext"], {
                "nonce": base64.b64encode(res["nonce"]).decode('utf-8'),
                "scheme": res["scheme"],
            }
        elif self.mode == "fernet":
            res = self.encrypt_fernet(dp_bytes)
            ciphertext, meta = res["ciphertext"], {"scheme": res["scheme"]}
        elif self.mode == "kms_envelope":
            res = self.encrypt_kms_envelope(dp_bytes)
            ciphertext, meta = res["ciphertext"], {
                "nonce": base64.b64encode(res["nonce"]).decode('utf-8'),
                "scheme": res["scheme"],
                "wrapped_key": res["wrapped_key"],
                "kms_key_id": res["kms_key_id"],
            }
        elif self.mode == "he_ckks":
            res = self.encrypt_he_ckks(dp_bytes)
            ciphertext, meta = res["ciphertext"], {
                "scheme": res["scheme"], "params": res.get("params")
            }
        elif self.mode == "smpc":
            res = self.create_smpc_shares(dp_bytes)
            ciphertext, meta = None, res
        else:
            raise ValueError(f"Unknown encryption mode: {self.mode}")

        # Save encrypted update
        ts = int(time.time() * 1000)
        final_fname = f"encdp_{ts}.pt.enc2" if ciphertext is not None else f"encdp_shares_{ts}.json"
        final_path = os.path.join(self.final_store_dir, final_fname)

        if ciphertext is not None:
            with open(final_path, "wb") as wf:
                wf.write(ciphertext)
        else:
            with open(final_path, "w") as wf:
                json.dump(meta, wf)
        final_uri = "file://" + final_path

        # 👇 Use centralized receipt manager
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
