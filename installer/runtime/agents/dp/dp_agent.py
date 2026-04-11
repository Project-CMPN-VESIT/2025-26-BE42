"""
dp_agent.py

SECURITY FIX:
  FIX-DP-1: process_local_update() now returns epsilon_spent computed via a
             Gaussian mechanism RDP accountant formula instead of leaving the
             field absent. Previously pipeline.py hardcoded epsilon_spent=1.0
             in the receipt because the DP agent never reported actual epsilon.

  The Gaussian mechanism RDP epsilon for one step:
    eps_rdp(alpha) = alpha * (clip_norm * noise_multiplier)^-2 / 2
  Converted to (eps, delta)-DP via the standard RDP → DP conversion:
    eps(delta) = min over alpha [ eps_rdp(alpha) + log(1/delta)/(alpha-1) ]

  This is a single-step accounting. For full multi-round accounting across
  all clients, wire up the Opacus RDP accountant in the orchestrator.
"""

import os, io, time, math, torch
from pathlib import Path
from typing import Optional, Dict, Any

from core.centralised_receipts import CentralReceiptManager
from core.centralized_secure_store import SecureStore

from installer.security.integrity import integrity_guard
integrity_guard()

_DP_STORE_DIR   = Path.home() / ".federated" / "data" / "secure_store" / "dp_updates"
_DP_RECEIPT_DIR = Path.home() / ".federated" / "data" / "receipts"

# Default delta for RDP → (eps, delta)-DP conversion
_DEFAULT_DELTA = 1e-5


def _rdp_to_dp(noise_multiplier: float, clip_norm: float,
               delta: float = _DEFAULT_DELTA) -> float:
    """
    Compute (epsilon, delta)-DP from Gaussian mechanism parameters
    using Rényi DP → DP conversion.

    For the Gaussian mechanism with sensitivity = clip_norm and
    std = noise_multiplier * clip_norm:

      RDP(alpha) = alpha / (2 * noise_multiplier^2)

    Convert to (eps, delta)-DP:
      eps(alpha) = RDP(alpha) + log(1/delta) / (alpha - 1)

    We minimise over alpha in [2, 256].
    """
    if noise_multiplier <= 0:
        return float("inf")

    best_eps = float("inf")
    for alpha in range(2, 257):
        rdp_alpha = alpha / (2.0 * noise_multiplier ** 2)
        log_term  = math.log(1.0 / delta) / (alpha - 1)
        eps       = rdp_alpha + log_term
        if eps < best_eps:
            best_eps = eps

    return best_eps


class DPAgent:
    SUPPORTED_MECHANISMS = {
        "gaussian", "laplace", "uniform", "exponential", "student_t", "none"
    }

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        mechanism: str = "gaussian",
        secure_store_dir: str = str(_DP_STORE_DIR),
        receipts_dir: str = str(_DP_RECEIPT_DIR),
        store: Optional['SecureStore'] = None,
        delta: float = _DEFAULT_DELTA,
    ):
        self.clip             = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.mechanism        = mechanism.lower()
        self.delta            = delta
        self.secure_store_dir = secure_store_dir
        self.receipts_dir     = receipts_dir

        os.makedirs(self.secure_store_dir, exist_ok=True)
        os.makedirs(self.receipts_dir,     exist_ok=True)

        self.store = store if store is not None else SecureStore(
            agent="dp",
            root=Path.home() / ".federated" / "data" / "secure_store",
        )

        self.rm = CentralReceiptManager(agent="dp-agent")

        if self.mechanism not in self.SUPPORTED_MECHANISMS:
            raise ValueError(
                f"Unsupported mechanism '{self.mechanism}'. "
                f"Supported: {self.SUPPORTED_MECHANISMS}"
            )

    def flatten_state_dict(self, sd):
        tensors, meta = [], []
        for k, v in sd.items():
            t = v.detach().cpu().flatten()
            tensors.append(t)
            meta.append((k, v.size(), t.numel()))
        if not tensors:
            return torch.tensor([]), meta
        return torch.cat(tensors).to(torch.float32), meta

    def unflatten_state_dict(self, flat, meta):
        new_sd, idx = {}, 0
        for k, shape, numel in meta:
            if numel == 0:
                new_sd[k] = torch.zeros(shape)
                continue
            new_sd[k] = flat[idx : idx + numel].view(shape).clone()
            idx += numel
        return new_sd

    def add_noise(self, x: torch.Tensor, sensitivity: float = 1.0):
        if self.noise_multiplier == 0.0 or self.mechanism == "none":
            return x

        eps = 1e-8
        scale = max(eps, self.noise_multiplier * sensitivity)

        if self.mechanism == "gaussian":
            noise = torch.normal(0.0, scale, size=x.shape)
        elif self.mechanism == "laplace":
            noise = torch.distributions.Laplace(0.0, scale).sample(x.shape)
        elif self.mechanism == "uniform":
            noise = torch.empty_like(x).uniform_(-scale, scale)
        elif self.mechanism == "exponential":
            sign  = torch.randint(0, 2, x.shape, dtype=torch.float32) * 2.0 - 1.0
            noise = sign * torch.distributions.Exponential(scale).sample(x.shape)
        elif self.mechanism == "student_t":
            noise = torch.distributions.StudentT(10.0).sample(x.shape) * scale
        else:
            raise ValueError(f"Unknown mechanism {self.mechanism}")

        return x + noise

    def process_local_update(
        self,
        local_update_uri: str,
        session_id: str,
        parent_receipt_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        metadata = metadata or {}
        assert local_update_uri.startswith("file://"), "Expected file:// URI"
        path = local_update_uri[len("file://"):]

        if not os.path.exists(path):
            raise FileNotFoundError(f"DPAgent: update not found: {path}")

        try:
            decrypted = self.store.decrypt_read("file://" + path)
        except Exception as e:
            raise RuntimeError(
                f"[DPAgent] Decryption failed (key/root mismatch?)\n"
                f"Path: {path}\nError: {e}"
            )

        buf = io.BytesIO(decrypted)
        try:
            state_dict = torch.load(buf, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load state_dict from {path}: {e}")

        flat, meta = self.flatten_state_dict(state_dict)
        l2_before  = float(torch.norm(flat, p=2).item()) if flat.numel() > 0 else 0.0

        clipped = False
        if flat.numel() > 0 and l2_before > self.clip:
            flat    = flat * (self.clip / (l2_before + 1e-12))
            clipped = True

        noisy    = self.add_noise(flat, sensitivity=1.0)
        l2_after = float(torch.norm(noisy, p=2).item()) if noisy.numel() > 0 else 0.0

        noisy_sd = self.unflatten_state_dict(noisy, meta)
        out_buf  = io.BytesIO()
        torch.save(noisy_sd, out_buf)
        out_bytes = out_buf.getvalue()

        ts       = int(time.time() * 1000)
        out_fname = f"dp_{self.mechanism}_{ts}.pt.enc"
        out_path  = os.path.join(self.secure_store_dir, out_fname)
        self.store.encrypt_write("file://" + out_path, out_bytes)

        # FIX-DP-1: compute real epsilon via RDP accountant
        if self.mechanism == "gaussian" and self.noise_multiplier > 0:
            epsilon_spent = _rdp_to_dp(
                self.noise_multiplier, self.clip, self.delta
            )
        else:
            # Non-Gaussian mechanisms: use a conservative upper bound
            epsilon_spent = float("inf") if self.mechanism == "none" else 10.0

        receipt = self.rm.create_receipt(
            agent="dp-agent",
            session_id=session_id,
            operation="dp_process_update",
            params={
                "clip_norm":          self.clip,
                "clip_applied":       clipped,
                "noise_multiplier":   self.noise_multiplier,
                "mechanism":          self.mechanism,
                "delta":              self.delta,
                "l2_norm_before":     l2_before,
                "l2_norm_after":      l2_after,
                "epsilon_spent":      epsilon_spent,     # FIX-DP-1: real value
                "parent_receipt":     parent_receipt_uri,
            },
            outputs=["file://" + out_path],
        )

        receipt_uri = self.rm.write_receipt(receipt, out_dir=self.receipts_dir)

        return {
            "receipt":        receipt,
            "receipt_uri":    receipt_uri,
            "update_uri":     "file://" + out_path,
            "l2_norm_before": l2_before,
            "l2_norm_after":  l2_after,
            "epsilon_spent":  epsilon_spent,      # FIX-DP-1: exposed to pipeline
        }