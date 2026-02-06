# dp_agent/dp_agent.py
import os, io, time, torch
from pathlib import Path
from typing import Optional, Dict, Any

# 👇 centralized receipts + secure store
from centralised_receipts import CentralReceiptManager
from centralized_secure_store import SecureStore

from installer.security.integrity import integrity_guard
integrity_guard()

class DPAgent:
    SUPPORTED_MECHANISMS = {
        "gaussian", "laplace", "uniform", "exponential", "student_t", "none"
    }

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        mechanism: str = "gaussian",
        secure_store_dir: str = "secure_store/local_updates",
        receipts_dir: str = "receipts",
        store: Optional['SecureStore'] = None
    ):
        self.clip = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.mechanism = mechanism.lower()
        self.secure_store_dir = secure_store_dir
        self.receipts_dir = receipts_dir
        os.makedirs(self.secure_store_dir, exist_ok=True)
        os.makedirs(self.receipts_dir, exist_ok=True)

        # ✅ Reuse centralized SecureStore if provided (avoids key mismatch)
        if store is not None:
            self.store = store
        else:
            # Default to explicit named args to avoid positional mismatch
            self.store = SecureStore(
                agent="dp-agent",
                root=Path("./secure_store")
            )

        # Centralized receipt manager
        self.rm = CentralReceiptManager(agent="dp-agent")

        if self.mechanism not in self.SUPPORTED_MECHANISMS:
            raise ValueError(
                f"Unsupported mechanism '{self.mechanism}'. "
                f"Supported: {self.SUPPORTED_MECHANISMS}"
            )

    # ---------- flatten / unflatten helpers ----------
    def flatten_state_dict(self, sd):
        tensors, meta = [], []
        for k, v in sd.items():
            t = v.detach().cpu().flatten()
            tensors.append(t)
            meta.append((k, v.size(), t.numel()))
        if len(tensors) == 0:
            return torch.tensor([]), meta
        flat = torch.cat(tensors).to(torch.float32)
        return flat, meta

    def unflatten_state_dict(self, flat, meta):
        new_sd, idx = {}, 0
        for k, shape, numel in meta:
            if numel == 0:
                new_sd[k] = torch.zeros(shape)
                continue
            seg = flat[idx: idx + numel]
            seg = seg.view(shape).clone()
            new_sd[k] = seg
            idx += numel
        return new_sd

    # ---------- noise helper ----------
    def add_noise(self, x: torch.Tensor, sensitivity: float = 1.0):
        """
        Add noise to flat tensor x according to mechanism and noise_multiplier.
        Uses small epsilon to avoid zero-scale problems.
        """
        if self.noise_multiplier == 0.0 or self.mechanism == "none":
            return x  # no noise applied

        eps = 1e-8
        scale = max(eps, self.noise_multiplier * sensitivity)

        if self.mechanism == "gaussian":
            noise = torch.normal(0.0, scale, size=x.shape)
        elif self.mechanism == "laplace":
            # torch.distributions.Laplace requires scale > 0
            noise = torch.distributions.Laplace(0.0, scale).sample(x.shape)
        elif self.mechanism == "uniform":
            # uniform in [-scale, scale]
            noise = torch.empty_like(x).uniform_(-scale, scale)
        elif self.mechanism == "exponential":
            # Exponential with scale > 0; create signed exponential
            sign = torch.randint(0, 2, x.shape, dtype=torch.float32) * 2.0 - 1.0
            # Exponential(rate) in torch uses rate parameter; torch.distributions.Exponential(scale) expects scale > 0
            noise = sign * torch.distributions.Exponential(scale).sample(x.shape)
        elif self.mechanism == "student_t":
            df = 10.0
            noise = torch.distributions.StudentT(df).sample(x.shape) * scale
        else:
            raise ValueError(f"Unknown mechanism {self.mechanism}")
        return x + noise

    # ---------- core DP processing ----------
    def process_local_update(
        self,
        local_update_uri: str,
        session_id: str,
        parent_receipt_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        local_update_uri: 'file://<path>'
        metadata: optional dict (trainer's receipt fields)
        returns: {'receipt': receipt, 'receipt_uri': receipt_uri}
        """
        metadata = metadata or {}
        assert local_update_uri.startswith("file://"), "Demo expects file:// URIs"
        path = local_update_uri[len("file://"):]

        if not os.path.exists(path):
            raise FileNotFoundError(f"DPAgent: update not found: {path}")

        # Try to decrypt; fallback to raw read
        try:
            decrypted = self.store.decrypt_read("file://" + path)
        except Exception as e:
            raise RuntimeError(
                f"[DPAgent] Failed to decrypt local update. "
                f"This indicates a SecureStore key/root mismatch.\n"
                f"Path: {path}\nError: {e}"
            )

        # Load model state dict from bytes
        buf = io.BytesIO(decrypted)
        try:
            state_dict = torch.load(
                buf,
                map_location="cpu",
                weights_only=False,   # 🔥 REQUIRED for PyTorch ≥ 2.6
            )
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load model state_dict from {path}: {e}")

        # Flatten parameters
        flat, meta = self.flatten_state_dict(state_dict)
        l2_before = float(torch.norm(flat, p=2).item()) if flat.numel() > 0 else 0.0

        # Clip gradient norm
        clipped = False
        if flat.numel() > 0 and l2_before > self.clip:
            flat = flat * (self.clip / (l2_before + 1e-12))
            clipped = True

        # Add noise
        noisy = self.add_noise(flat, sensitivity=1.0)
        l2_after = float(torch.norm(noisy, p=2).item()) if noisy.numel() > 0 else 0.0

        # Reconstruct and re-serialize model
        noisy_sd = self.unflatten_state_dict(noisy, meta)
        out_buf = io.BytesIO()
        torch.save(noisy_sd, out_buf)
        out_bytes = out_buf.getvalue()

        # Save new encrypted DP update
        ts = int(time.time() * 1000)
        out_fname = f"dp_{self.mechanism}_{ts}.pt.enc"
        out_path = os.path.join(self.secure_store_dir, out_fname)
        self.store.encrypt_write("file://" + out_path, out_bytes)

        # Create centralized receipt
        receipt = self.rm.create_receipt(
            agent="dp-agent",
            session_id=session_id,
            operation="dp_process_update",
            params={
                "clip_norm": self.clip,
                "clip_applied": clipped,
                "noise_multiplier": self.noise_multiplier,
                "mechanism": self.mechanism,
                "l2_norm_before": l2_before,
                "l2_norm_after": l2_after,
                "parent_receipt": parent_receipt_uri,  # stored as metadata, not linkage
            },
            outputs=["file://" + out_path],
        )

        receipt_uri = self.rm.write_receipt(receipt, out_dir=self.receipts_dir)
        return {
            "receipt": receipt,
            "receipt_uri": receipt_uri,
            "update_uri": "file://" + out_path,
            "l2_norm_before": l2_before,
            "l2_norm_after": l2_after,
        }
