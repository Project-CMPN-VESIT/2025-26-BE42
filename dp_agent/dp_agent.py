import os, io, time, torch

# 👇 centralized receipts + secure store
from centralised_receipts import CentralReceiptManager
from centralized_secure_store import SecureStore


class DPAgent:
    def __init__(self,
                 clip_norm=1.0,
                 noise_multiplier=1.0,
                 secure_store_dir="secure_store/local_updates",
                 receipts_dir="receipts"):
        self.clip = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.secure_store_dir = secure_store_dir
        self.receipts_dir = receipts_dir
        os.makedirs(self.secure_store_dir, exist_ok=True)
        os.makedirs(self.receipts_dir, exist_ok=True)

        # Centralized SecureStore
        self.store = SecureStore("./secure_store")

        # Centralized receipt manager
        self.rm = CentralReceiptManager()

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

    # ---------- core DP processing ----------
    def process_local_update(self, local_update_uri: str, metadata: dict = None):
        """
        local_update_uri: 'file://<path>'
        metadata: optional dict (trainer's receipt fields)
        returns: dp_receipt dict
        """
        metadata = metadata or {}
        assert local_update_uri.startswith("file://"), "Demo expects file:// URIs"
        path = local_update_uri[len("file://"):]

        if not os.path.exists(path):
            raise FileNotFoundError(f"DPAgent: update not found: {path}")

        # 1) read encrypted trainer update and decrypt
        decrypted = self.store.decrypt_read("file://" + path)

        # 2) load state_dict from bytes
        buf = io.BytesIO(decrypted)
        state_dict = torch.load(buf, map_location="cpu")

        # 3) flatten
        flat, meta = self.flatten_state_dict(state_dict)
        l2_before = float(torch.norm(flat, p=2).item()) if flat.numel() > 0 else 0.0

        # 4) clip
        clipped = False
        if flat.numel() > 0 and l2_before > self.clip:
            flat = flat * (self.clip / l2_before)
            clipped = True

        # 5) add Gaussian noise
        noise_std = self.noise_multiplier * self.clip
        noisy = flat
        if flat.numel() > 0:
            noise = torch.normal(mean=0.0, std=noise_std, size=flat.shape)
            noisy = flat + noise

        l2_after = float(torch.norm(noisy, p=2).item()) if noisy.numel() > 0 else 0.0

        # 6) unflatten back to state_dict
        noisy_sd = self.unflatten_state_dict(noisy, meta)

        # 7) serialize
        out_buf = io.BytesIO()
        torch.save(noisy_sd, out_buf)
        out_bytes = out_buf.getvalue()

        # 8) save encrypted DP update
        ts = int(time.time() * 1000)
        out_fname = f"dp_{ts}.pt.enc"
        out_path = os.path.join(self.secure_store_dir, out_fname)
        self.store.encrypt_write("file://" + out_path, out_bytes)

        # 9) build centralized receipt
        receipt = self.rm.create_receipt(
            agent="dp-agent",
            session_id=metadata.get("session_id"),  # inherit from trainer if exists
            operation="dp_process_update",
            params={
                "clip_norm": self.clip,
                "clip_applied": clipped,
                "noise_multiplier": self.noise_multiplier,
                "l2_norm_before": l2_before,
                "l2_norm_after": l2_after,
            },
            outputs=["file://" + out_path],
        )

        # 10) write receipt to disk
        receipt_uri = self.rm.write_receipt(receipt, out_dir=self.receipts_dir)

        return {"receipt": receipt, "receipt_uri": receipt_uri}
