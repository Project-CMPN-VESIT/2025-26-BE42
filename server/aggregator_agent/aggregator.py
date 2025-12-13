import os
import io
import json
import numpy as np
import torch
from typing import List, Dict, Optional


class AggregatorAgent:
    """
    AggregatorAgent (Standalone Version)
    ------------------------------------
    Responsibilities:
      - Accept encrypted DP-updates (file-based)
      - Decrypt each update (via SecureStore or scheme-aware loader)
      - Convert into tensors
      - Perform robust aggregation:
          * mean
          * trimmed mean
          * coordinate-wise median
    """

    def __init__(self,
                 mode: str = "trimmed_mean",
                 trim_ratio: float = 0.1,
                 decrypt_callback=None):

        self.mode = mode
        self.trim_ratio = trim_ratio
        self.decrypt_callback = decrypt_callback or self._default_decrypt

    # ------------------------------------------------------------------
    # Default decryption callback (matches EncryptionAgent outputs)
    # ------------------------------------------------------------------
    def _default_decrypt(self,
                         enc_path: str,
                         scheme: str,
                         nonce: Optional[str]):
        """
        Matches EXACTLY the output formats used by EncryptionAgent.
        """

        # Normalize URI
        if enc_path.startswith("file://"):
            enc_path = enc_path[len("file://"):]

        # AES-GCM / KMS (SecureStore)
        if scheme.lower().startswith("aes") or scheme.lower().startswith("kms"):
            from centralized_secure_store import SecureStore

            store = SecureStore("./secure_store")
            raw = store.decrypt_read("file://" + enc_path)

            try:
                return torch.load(io.BytesIO(raw))
            except Exception:
                return np.frombuffer(raw, dtype=np.float32)

        # CKKS (Homomorphic) – not decrypted here
        if "CKKS" in scheme.upper():
            with open(enc_path, "rb") as f:
                return f.read()  # ciphertext bytes

        # SMPC (XOR demo)
        if "XOR" in scheme.upper():
            with open(enc_path, "r") as f:
                d = json.load(f)
            return d["shares"]

        raise ValueError(f"Unknown encryption scheme: {scheme}")

    # ------------------------------------------------------------------
    # Main aggregation entrypoint
    # ------------------------------------------------------------------
    def aggregate_updates(self, updates: List[Dict]):

        decrypted = []

        for u in updates:
            enc_path = u["enc_uri"]
            scheme = u.get("scheme", "aes")
            nonce = u.get("nonce")

            t = self.decrypt_callback(enc_path, scheme, nonce)

            if isinstance(t, torch.Tensor):
                decrypted.append(t.detach().cpu().numpy())
            elif isinstance(t, np.ndarray):
                decrypted.append(t)
            else:
                raise TypeError(f"Unsupported decrypted type: {type(t)}")

        arr = np.stack(decrypted, axis=0)
        return self._apply_aggregation(arr)

    # ------------------------------------------------------------------
    # Robust aggregation methods
    # ------------------------------------------------------------------
    def _apply_aggregation(self, arr: np.ndarray):

        if self.mode == "mean":
            return np.mean(arr, axis=0)

        elif self.mode == "trimmed_mean":
            lower = int(self.trim_ratio * len(arr))
            upper = len(arr) - lower
            sorted_arr = np.sort(arr, axis=0)
            return np.mean(sorted_arr[lower:upper], axis=0)

        elif self.mode in ("median", "coordinate_median"):
            return np.median(arr, axis=0)

        else:
            raise NotImplementedError(f"Unknown aggregation mode: {self.mode}")
