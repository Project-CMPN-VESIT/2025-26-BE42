import os
import subprocess
import tempfile
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, Any, List

from core.centralized_secure_store import SecureStore
from core.centralised_receipts import CentralReceiptManager


def _extract_wav2vec2_features(wav_path: str, model_id: str, pool: str, max_dim: int) -> Dict[str, Any]:
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
    except ImportError:
        return {"_status": "unavailable"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2Model.from_pretrained(model_id).to(device)

    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    input_values = processor(
        waveform.squeeze().numpy(),
        return_tensors="pt",
        sampling_rate=16000
    ).input_values.to(device)

    with torch.no_grad():
        hidden = model(input_values).last_hidden_state.cpu().numpy()

    pooled = np.mean(hidden, axis=1).squeeze() if pool == "mean" else hidden.squeeze()

    if pooled.shape[0] > max_dim:
        pooled = pooled[:max_dim]

    return {"vector": pooled.tolist(), "_status": "ok"}

def _extract_opensmile_features(
    wav_path: str,
    opensmile_bin: str,
    opensmile_config: str
) -> Dict[str, Any]:
    """
    Run openSMILE (eGeMAPS) via subprocess and parse output CSV.
    """
    if not os.path.exists(opensmile_bin):
        print(f"⚠️ openSMILE binary not found at {opensmile_bin}. Skipping eGeMAPS.")
        return {}
    if not os.path.exists(opensmile_config):
        print(f"⚠️ openSMILE config not found at {opensmile_config}. Skipping eGeMAPS.")
        return {}

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpout:
        tmp_csv = tmpout.name

    cmd = [
        opensmile_bin,
        "-C", opensmile_config,
        "-I", wav_path,
        "-O", tmp_csv,
        "-nologfile",
        "-noconsoleoutput"
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = np.genfromtxt(tmp_csv, delimiter=';', names=True, dtype=None, encoding='utf-8')
        if data.ndim == 0:  # single row
            feature_dict = {name: float(data[name]) for name in data.dtype.names}
        else:
            feature_dict = {name: float(np.mean(data[name])) for name in data.dtype.names}
    except Exception as e:
        print(f"⚠️ openSMILE extraction failed: {e}")
        feature_dict = {}
    finally:
        try:
            os.remove(tmp_csv)
        except OSError:
            pass

    return {"egemaps": feature_dict}


def _compute_basic_prosody(wav_path: str) -> Dict[str, Any]:
    """
    Compute basic energy, pitch, and zero-crossing features using torchaudio.
    """
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono
    energy = torch.mean(waveform ** 2).item()
    zcr = torch.mean((waveform[:, 1:] * waveform[:, :-1]) < 0).item()

    try:
        pitch = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate=sr)
        pitch_mean = torch.mean(pitch[pitch > 0]).item()
    except Exception:
        pitch_mean = 0.0

    return {"energy": energy, "pitch_mean": pitch_mean, "zcr": zcr}


def process_audio_file(
    audio_path: str,
    cfg: Dict[str, Any],
    session_id: str
) -> List[Dict[str, Any]]:
    """
    Extracts multimodal audio features (prosody + wav2vec2 + eGeMAPS).
    Writes encrypted outputs and returns a structured rows list.
    """

    audio_path = str(audio_path)
    print(f"🎧 Processing audio: {audio_path}")
    rows = []

    storage = SecureStore(
        agent="lda-audio",
        root=Path(cfg["storage"]["root"]).resolve()
    )
    rm = CentralReceiptManager(agent="lda-audio")

    features_cfg = cfg["audio_pipe"]["features"]

    # ---- feature containers ----
    derived: Dict[str, Any] = {}
    feature_status: Dict[str, str] = {}

    # ---- Prosody ----
    if features_cfg.get("prosody", False):
        try:
            prosody = _compute_basic_prosody(audio_path)
            derived["prosody"] = prosody
            feature_status["prosody"] = "ok"
        except Exception as e:
            feature_status["prosody"] = f"failed: {type(e).__name__}"

    # ---- eGeMAPS (openSMILE) ----
    egemaps_cfg = features_cfg.get("egemaps", {})
    if egemaps_cfg.get("enabled", False):
        try:
            opensmile_bin = egemaps_cfg.get("opensmile_binary")
            opensmile_conf = egemaps_cfg.get("opensmile_config")
            egemaps = _extract_opensmile_features(
                audio_path,
                opensmile_bin,
                opensmile_conf
            )
            if egemaps:
                derived["egemaps"] = egemaps
                feature_status["egemaps"] = "ok"
            else:
                feature_status["egemaps"] = "unavailable"
        except Exception as e:
            feature_status["egemaps"] = f"failed: {type(e).__name__}"

    # ---- wav2vec2 ----
    wav2vec_cfg = features_cfg.get("wav2vec2", {})
    if wav2vec_cfg.get("enabled", False):
        try:
            max_dim = wav2vec_cfg.get("max_dim", 512)
            w2v = _extract_wav2vec2_features(
                audio_path,
                model_id=wav2vec_cfg.get("model", "facebook/wav2vec2-base-960h"),
                pool=wav2vec_cfg.get("pool", "mean"),
            )

            # enforce dimensionality cap if vector exists
            if isinstance(w2v, dict) and "wav2vec2" in w2v:
                vec = w2v["wav2vec2"]
                if isinstance(vec, list) and len(vec) > max_dim:
                    vec = vec[:max_dim]
                derived["wav2vec2"] = vec
                feature_status["wav2vec2"] = "ok"
            else:
                feature_status["wav2vec2"] = "unavailable"

        except Exception as e:
            feature_status["wav2vec2"] = f"failed: {type(e).__name__}"

    # ---- Package + Secure Write ----
    record = {
        "session_id": session_id,
        "path": audio_path,
        "features": {
            "audio": derived
        },
        "derived": {
            "num_features": sum(
                len(v) if isinstance(v, dict) else 1
                for v in derived.values()
            ),
            "feature_status": feature_status,
        },
    }

    fname = Path(audio_path).stem + "_audio_features.json"
    uri = storage.encrypt_write(
        f"file://{storage.root / session_id / 'audio' / fname}",
        json.dumps(record).encode(),
    )

    receipt = rm.create_receipt(
        session_id=session_id,
        operation="audio_process",
        params={
            "input": audio_path,
            "features_requested": list(features_cfg.keys()),
            "features_extracted": list(derived.keys()),
        },
        outputs=[uri],
    )

    receipt_uri = storage.encrypt_write(
        f"file://{storage.root / session_id / 'receipts' / (Path(audio_path).stem + '_audio.json.enc')}",
        json.dumps(receipt).encode(),
    )

    rows.append({
        "uri": uri,
        "receipt_uri": receipt_uri,
        **record
    })

    print(f"✅ Audio processed and stored for {audio_path}")
    return rows
