"""
Audio pipeline that computes a single wav2vec2 embedding per audio file.
Now uses centralized SecureStore + CentralReceiptManager for receipts/storage.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import wave, contextlib, logging, json

# centralized imports
from centralized_secure_store import SecureStore
from centralised_receipts import CentralReceiptManager

log = logging.getLogger(__name__)

# optional heavy libs
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    librosa = None
    _HAS_LIBROSA = False

try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import torch
    _HAS_TRANSFORMERS = True
except Exception:
    Wav2Vec2Processor = None
    Wav2Vec2Model = None
    torch = None
    _HAS_TRANSFORMERS = False

# Global cached model/processor to avoid reloading
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _wav_duration(path: str) -> float:
    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        if _HAS_LIBROSA:
            try:
                y, sr = librosa.load(path, sr=None, mono=True)
                return len(y) / float(sr)
            except Exception:
                pass
    return 0.0


def _safe_load_audio(path: str, sr: int = 16000):
    if _HAS_LIBROSA:
        try:
            y, s = librosa.load(path, sr=sr, mono=True)
            return y, s
        except Exception as e:
            log.debug("librosa.load failed, fallback: %s", e)

    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            s = wf.getframerate()
            nchan = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if sampwidth == 2:
            import numpy as np, struct
            fmt = "<{}h".format(int(len(frames) / 2))
            samples = struct.unpack(fmt, frames)
            arr = np.array(samples, dtype=np.float32)
            if nchan > 1:
                arr = arr.reshape(-1, nchan).mean(axis=1)
            arr /= float(max(1.0, np.max(np.abs(arr))))
            return arr, s
    except Exception as e:
        log.debug("wave fallback load failed: %s", e)

    return None, None


def _ensure_model(model_name: str, device: Optional[str] = None):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]["processor"], _MODEL_CACHE[model_name]["model"], _MODEL_CACHE[model_name]["device"]

    if not _HAS_TRANSFORMERS:
        raise RuntimeError("transformers/torch not available")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()

    _MODEL_CACHE[model_name] = {"processor": processor, "model": model, "device": device}
    return processor, model, device


def _chunk_audio(y, sr: int, chunk_s: float):
    if chunk_s <= 0:
        yield 0, len(y), y
        return
    chunk_frames = int(chunk_s * sr)
    for start in range(0, len(y), chunk_frames):
        yield start, min(len(y), start + chunk_frames), y[start:start + chunk_frames]


def _compute_wav2vec2_embedding_for_waveform(
    y, sr: int, model_name="facebook/wav2vec2-base-960h",
    pool="mean", device=None, chunk_seconds=20.0
) -> List[float]:
    if not _HAS_TRANSFORMERS:
        raise RuntimeError("transformers/torch not available")

    processor, model, model_device = _ensure_model(model_name, device)

    vecs = []
    for _, _, chunk in _chunk_audio(y, sr, chunk_seconds):
        if not len(chunk):
            continue
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            last_hidden = model(**inputs).last_hidden_state
            if pool == "mean":
                vec = last_hidden.mean(dim=1).squeeze(0).cpu().numpy()
            else:
                vec = last_hidden[:, 0, :].squeeze(0).cpu().numpy()
        vecs.append(vec)

    if not vecs:
        raise RuntimeError("No embeddings computed")

    import numpy as np
    return np.mean(np.stack(vecs, axis=0), axis=0).tolist()


def process_audio_file(p: Path, cfg: dict, session_id: str) -> List[Dict[str, Any]]:
    p = Path(p)
    out_rows: List[Dict[str, Any]] = []

    store = SecureStore(cfg["storage"]["root"])
    rm = CentralReceiptManager()

    dur = _wav_duration(str(p))
    features: Dict[str, Any] = {"duration": dur}

    enabled = cfg.get("ingest", {}).get("audio", {}).get("features", {}).get("wav2vec2", {}).get("enabled", True)
    if not enabled:
        processed_uri = str(p)
        receipt = rm.create_receipt(
            agent="lda-audio",
            session_id=session_id,
            operation="audio_preprocessing",
            params={"source": str(p), "duration": dur},
            outputs=[processed_uri],
        )
        rrel = f"{session_id}/receipts/audio_{p.stem}.json.enc"
        receipt_uri = store.encrypt_write(f"file://{store.root / rrel}", json.dumps(receipt).encode())
        out_rows.append({
            "session_id": session_id,
            "modality": "audio",
            "source": str(p),
            "filename": p.name,
            "processed_uri": processed_uri,
            "receipt_uri": receipt_uri,
            "features": features
        })
        return out_rows

    # load audio
    target_sr = cfg.get("ingest", {}).get("audio", {}).get("sr", 16000)
    y, sr = _safe_load_audio(str(p), sr=target_sr)
    if y is None:
        processed_uri = str(p)
        receipt = rm.create_receipt(
            agent="lda-audio",
            session_id=session_id,
            operation="audio_preprocessing",
            params={"source": str(p), "duration": dur, "error": "load_failed"},
            outputs=[processed_uri],
        )
        rrel = f"{session_id}/receipts/audio_{p.stem}.json.enc"
        receipt_uri = store.encrypt_write(f"file://{store.root / rrel}", json.dumps(receipt).encode())
        out_rows.append({
            "session_id": session_id,
            "modality": "audio",
            "source": str(p),
            "filename": p.name,
            "processed_uri": processed_uri,
            "receipt_uri": receipt_uri,
            "features": features
        })
        return out_rows

    # compute embedding
    try:
        emb = _compute_wav2vec2_embedding_for_waveform(
            y, sr,
            model_name=cfg["ingest"]["audio"]["features"]["wav2vec2"].get("model", "facebook/wav2vec2-base-960h"),
            pool=cfg["ingest"]["audio"]["features"]["wav2vec2"].get("pool", "mean"),
            device=cfg["ingest"]["audio"]["features"]["wav2vec2"].get("device"),
            chunk_seconds=float(cfg["ingest"]["audio"]["features"]["wav2vec2"].get("chunk_seconds", 20.0))
        )
        features["w2v2_embedding"] = emb
        features["w2v2_dim"] = len(emb)
    except Exception as e:
        log.exception("Embedding computation failed: %s", e)

    processed_uri = str(p)
    receipt = rm.create_receipt(
        agent="lda-audio",
        session_id=session_id,
        operation="audio_preprocessing",
        params={"source": str(p), "duration": dur, "w2v2_dim": features.get("w2v2_dim")},
        outputs=[processed_uri],
    )
    rrel = f"{session_id}/receipts/audio_{p.stem}.json.enc"
    receipt_uri = store.encrypt_write(f"file://{store.root / rrel}", json.dumps(receipt).encode())

    row = {
        "session_id": session_id,
        "modality": "audio",
        "source": str(p),
        "filename": p.name,
        "processed_uri": processed_uri,
        "receipt_uri": receipt_uri,
        "features": features
    }
    out_rows.append(row)
    return out_rows
