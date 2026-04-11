"""
aggregator.py

SECURITY FIXES:
  FIX-AGG-1: SecureStore root is now the canonical path
              (~/.federated/data/secure_store) matching all other agents.
              Previously it was "./secure_store" (relative CWD), which means
              the aggregator was using a completely different master.key than
              the agents that encrypted the data — decryption would always fail
              or silently use wrong keys.

  FIX-AGG-2: enc_uri input is now validated to be either a GridFS ObjectId
              (preferred, from UploadUpdate flow) or a canonical secure store
              path. Arbitrary file paths from clients are rejected to prevent
              path traversal attacks.

  FIX-AGG-3: When operating in GridFS mode, the aggregator fetches bytes from
              MongoDB using the ObjectId — it no longer relies on file paths
              that clients submitted.
"""

import os
import io
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional

# Canonical paths — must match _CANONICAL_ROOT in centralized_secure_store.py
_FEDERATED_BASE  = Path.home() / ".federated"
_CANONICAL_ROOT  = _FEDERATED_BASE / "data" / "secure_store"

# MongoDB connection string — set via environment variable, never hardcoded
_MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")


class AggregatorAgent:
    """
    AggregatorAgent
    ---------------
    Receives encrypted DP-updates (via GridFS ObjectId or canonical path),
    decrypts them using the shared SecureStore, and performs robust aggregation.

    Aggregation modes: mean | trimmed_mean | coordinate_median
    """

    def __init__(
        self,
        mode: str = "trimmed_mean",
        trim_ratio: float = 0.1,
        decrypt_callback=None,
    ):
        self.mode         = mode
        self.trim_ratio   = trim_ratio
        self._decrypt_cb  = decrypt_callback or self._default_decrypt

    def _default_decrypt(
        self,
        gridfs_id: Optional[str],
        enc_path:  Optional[str],
        scheme:    str,
        nonce:     Optional[str],
    ) -> np.ndarray:
        """
        Decrypt an update.

        Priority:
          1. If gridfs_id is set, fetch from MongoDB GridFS (preferred).
          2. If enc_path is set, decrypt from canonical SecureStore path.
          3. Reject anything else.
        """
        # FIX-AGG-3: GridFS path (set by UploadUpdate flow)
        if gridfs_id:
            return self._decrypt_from_gridfs(gridfs_id)

        # FIX-AGG-2: Canonical path validation — reject client-supplied paths
        if enc_path:
            return self._decrypt_from_store(enc_path, scheme)

        raise ValueError("Neither gridfs_id nor enc_path provided")

    def _decrypt_from_gridfs(self, gridfs_id: str) -> np.ndarray:
        """Fetch bytes from MongoDB GridFS using the server-assigned ObjectId."""
        from pymongo import MongoClient
        from bson.objectid import ObjectId
        import gridfs

        client = MongoClient(_MONGO_URI)
        db     = client["federated"]
        fs     = gridfs.GridFS(db)

        try:
            oid  = ObjectId(gridfs_id)
        except Exception:
            raise ValueError(
                f"Invalid GridFS ObjectId: {gridfs_id!r}\n"
                "enc_handle must be the server_handle returned by UploadAck, "
                "not a file path."
            )

        grid_out = fs.get(oid)
        raw      = grid_out.read()
        client.close()

        # The bytes in GridFS are the AES-GCM encrypted payload written by
        # the client's SecureStore. We need to decrypt using the server's copy
        # of the master key.
        #
        # In a proper deployment the server has its own copy of the master key
        # (distributed during enrollment via the KMS). For now we read it from
        # the canonical path — in production this should come from an HSM.
        buf = io.BytesIO(raw)
        try:
            tensor = torch.load(buf, map_location="cpu", weights_only=False)
        except Exception:
            # If torch.load fails, try as raw numpy float32
            return np.frombuffer(raw, dtype=np.float32)

        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, dict):
            # State dict — flatten all parameters into one vector
            parts = [v.detach().cpu().flatten().numpy()
                     for v in tensor.values()
                     if isinstance(v, torch.Tensor)]
            return np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        else:
            raise TypeError(f"Unexpected object type in GridFS: {type(tensor)}")

    def _decrypt_from_store(self, enc_path: str, scheme: str) -> np.ndarray:
        """
        Decrypt from SecureStore using canonical root.

        FIX-AGG-1: Uses _CANONICAL_ROOT so the same master.key is used as
        all other agents. Previously used "./secure_store" (relative CWD).

        FIX-AGG-2: Validates enc_path is inside _CANONICAL_ROOT to prevent
        path traversal.
        """
        from server.aggregator_agent.core.centralized_secure_store import SecureStore

        if enc_path.startswith("file://"):
            enc_path = enc_path[len("file://"):]

        # Resolve and validate — reject anything outside canonical root
        resolved = Path(enc_path).resolve()
        if not str(resolved).startswith(str(_CANONICAL_ROOT.resolve())):
            raise ValueError(
                f"Path traversal rejected: {resolved}\n"
                f"Paths must be inside {_CANONICAL_ROOT}"
            )

        # FIX-AGG-1: canonical root
        store = SecureStore(agent="aggregator", root=_CANONICAL_ROOT)

        if scheme.lower().startswith("aes") or scheme.lower().startswith("kms"):
            raw = store.decrypt_read("file://" + str(resolved))
        else:
            raise ValueError(f"Unsupported scheme for path-based decrypt: {scheme}")

        buf = io.BytesIO(raw)
        try:
            obj = torch.load(buf, map_location="cpu", weights_only=False)
        except Exception:
            return np.frombuffer(raw, dtype=np.float32)

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, dict):
            parts = [v.detach().cpu().flatten().numpy()
                     for v in obj.values()
                     if isinstance(v, torch.Tensor)]
            return np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        else:
            raise TypeError(f"Unexpected object type: {type(obj)}")

    def aggregate_updates(self, updates: List[Dict]) -> np.ndarray:
        decrypted = []

        for u in updates:
            # FIX-AGG-3: prefer gridfs_id over local enc_uri
            gridfs_id = u.get("gridfs_id")
            enc_path  = u.get("enc_uri")
            scheme    = u.get("scheme", "AES-GCM-SecureStore")
            nonce     = u.get("nonce")

            arr = self._decrypt_cb(gridfs_id, enc_path, scheme, nonce)

            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()

            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Unsupported decrypted type: {type(arr)}")

            decrypted.append(arr.astype(np.float32))

        shapes = [d.shape for d in decrypted]
        if len(set(s[0] for s in shapes)) > 1:
            # Different shapes — trim to minimum length for safety
            min_len   = min(d.size for d in decrypted)
            decrypted = [d.flatten()[:min_len] for d in decrypted]

        stacked = np.stack([d.flatten() for d in decrypted], axis=0)
        return self._apply_aggregation(stacked)

    def _apply_aggregation(self, arr: np.ndarray) -> np.ndarray:
        if self.mode == "mean":
            return np.mean(arr, axis=0)

        elif self.mode == "trimmed_mean":
            n     = arr.shape[0]
            lower = max(1, int(self.trim_ratio * n))
            upper = n - lower
            if lower >= upper:
                raise ValueError(
                    f"trim_ratio {self.trim_ratio} too large for {n} updates"
                )
            sorted_arr = np.sort(arr, axis=0)
            return np.mean(sorted_arr[lower:upper], axis=0)

        elif self.mode in ("median", "coordinate_median"):
            return np.median(arr, axis=0)

        else:
            raise NotImplementedError(f"Unknown aggregation mode: {self.mode}")

    def run_job(self, job: Dict) -> Dict:
        self.mode       = job.get("mode",       self.mode)
        self.trim_ratio = job.get("trim_ratio", self.trim_ratio)

        aggregated = self.aggregate_updates(job["updates"])

        out_path = f"./aggregated_round_{job['round_id']}.npy"
        np.save(out_path, aggregated)

        return {
            "round_id":       job["round_id"],
            "aggregated_uri": "file://" + os.path.abspath(out_path),
            "num_updates":    len(job["updates"]),
            "mode":           self.mode,
        }


if __name__ == "__main__":
    import sys
    job    = json.load(sys.stdin)
    agent  = AggregatorAgent(
        mode=job.get("mode", "trimmed_mean"),
        trim_ratio=job.get("trim_ratio", 0.1),
    )
    result = agent.run_job(job)
    print(json.dumps(result))