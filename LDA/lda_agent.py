from pathlib import Path
import json, time, hashlib, os
from typing import Dict, Any, Iterator
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from secrets import token_bytes

class SecureStore:
    def __init__(self, root: Path):
        self.root = root; self.root.mkdir(parents=True, exist_ok=True)
    def _key(self) -> bytes:
        # TODO: integrate OS keyring or TPM/HSM; this demo uses ephemeral per-session keys
        return token_bytes(32)  # AES-256 key
    def encrypt_write(self, relpath: str, payload: bytes) -> str:
        key = self._key(); aes = AESGCM(key); nonce = token_bytes(12)
        ct = aes.encrypt(nonce, payload, None)
        out = self.root / relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(nonce + ct)
        return f"local://{out}"

def sha256_bytes(b: bytes) -> str: return "sha256:" + hashlib.sha256(b).hexdigest()

class LocalDataAgent:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.store = SecureStore(Path(cfg["storage"]["root"]))
        self.session_id = f"sess-{int(time.time())}"

    # ---- Batch handlers ----
    def run_batch(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        manifest_lines = []
        # VIDEO
        if self.cfg["ingest"]["video"]["enabled"] and "video_dir" in inputs:
            for p in Path(inputs["video_dir"]).glob("*.mp4"):
                rows = self._process_video_file(p)  # list of dict rows
                manifest_lines += self._write_partitioned(rows, "video")
        # AUDIO
        if self.cfg["ingest"]["audio"]["enabled"] and "audio_dir" in inputs:
            for p in Path(inputs["audio_dir"]).glob("*.wav"):
                rows = self._process_audio_file(p)
                manifest_lines += self._write_partitioned(rows, "audio")
        # TEXT
        if self.cfg["ingest"]["text"]["enabled"] and "text_dir" in inputs:
            for p in Path(inputs["text_dir"]).glob("*.txt"):
                rows = self._process_text_file(p)
                manifest_lines += self._write_partitioned(rows, "text")

        manifest_bytes = ("\n".join(json.dumps(x) for x in manifest_lines)).encode()
        manifest_uri = self.store.encrypt_write(f"{self.session_id}/manifest.jsonl", manifest_bytes)
        receipt = self._receipt({"op":"preprocess","count":len(manifest_lines),"cfg":self.cfg})
        receipt_uri = self.store.encrypt_write(f"{self.session_id}/receipt.json", json.dumps(receipt).encode())
        return {"session_id": self.session_id, "artifact_manifest": manifest_uri, "receipts": [receipt_uri]}

    # ---- Video/audio/text processors (stubs) ----
    def _process_video_file(self, path: Path) -> list[dict]:
        # TODO: run 3D pose normalize + OpenFace; build per-window rows
        # return list of dicts with {session_id, t_start, t_end, AU features, pose, gaze, ...}
        return []

    def _process_audio_file(self, path: Path) -> list[dict]:
        # TODO: run VAD, prosody (F0/jitter/shimmer), eGeMAPS, wav2vec2 pooling
        return []

    def _process_text_file(self, path: Path) -> list[dict]:
        # TODO: NER PII scrub; sentence embeddings; metrics
        return []

    # ---- Writer helpers ----
    def _write_partitioned(self, rows: list[dict], modality: str) -> list[dict]:
        if not rows: return []
        payload = "\n".join(json.dumps(r) for r in rows).encode()  # (use parquet in real impl)
        ts = time.strftime("%Y-%m-%d/%H")
        uri = self.store.encrypt_write(f"{self.session_id}/{modality}/{ts}.jsonl.enc", payload)
        out = []
        for i, r in enumerate(rows):
            out.append({
                "session_id": self.session_id,
                "window_id": r.get("window_id", f"{modality}-{i}"),
                "modality": modality,
                "path": f"{uri}#row={i}",
                "hash": sha256_bytes(json.dumps(r, sort_keys=True).encode()),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            })
        return out

    def _receipt(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: sign with ed25519; this is a stub
        payload["sig"] = "ed25519:stub"
        payload["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        payload["agent"] = "local-data-agent"
        payload["session_id"] = self.session_id
        return payload
