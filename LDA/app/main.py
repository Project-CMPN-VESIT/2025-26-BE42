# app/main.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Dict, Any, List
from pathlib import Path
import yaml, json, time
from security.secure_store import SecureStore
from utils.receipts import make_receipt
from pipelines.video import process_video_file
from pipelines.audio import process_audio_file
from pipelines.text import process_text_sources
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

app = FastAPI(title="Local Data Agent (Privacy)", version="1.0.0")

class PreprocessRequest(BaseModel):
    mode: str  # "batch" | "interactive" | "continuous"
    inputs: Dict[str, str]
    config_uri: str

def _load_config(uri: str) -> dict:
    assert uri.startswith("file://"), "Only file:// URIs are supported for config"
    p = Path(uri[len("file://"):])
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _write_parquet_encrypted(store: SecureStore, session_id: str, modality: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    payload = buf.getvalue().to_pybytes()
    rel = f"{session_id}/{modality}/{datetime.utcnow().strftime('%Y-%m-%d/%H')}.parquet.enc"
    return store.encrypt_write(rel, payload)

@app.post("/local/preprocess")
def preprocess(req: PreprocessRequest):
    cfg = _load_config(req.config_uri)
    store = SecureStore(cfg["storage"]["root"])
    session_id = f"sess-{int(time.time())}"
    outputs = []
    manifest = []

    # VIDEO
    if cfg.get("ingest", {}).get("video", {}).get("enabled", False) and req.inputs.get("video_dir"):
        vdir = Path(req.inputs["video_dir"])
        for p in sorted(vdir.glob("*.mp4")):
            rows = process_video_file(p, cfg, session_id)
            uri = _write_parquet_encrypted(store, session_id, "video", rows)
            if uri:
                outputs.append(uri)
                for i, _ in enumerate(rows):
                    manifest.append({"session_id": session_id, "modality": "video", "uri": uri, "row": i})

    # AUDIO
    if cfg.get("ingest", {}).get("audio", {}).get("enabled", False) and req.inputs.get("audio_dir"):
        adir = Path(req.inputs["audio_dir"])
        for p in sorted(adir.glob("*.wav")):
            rows = process_audio_file(p, cfg, session_id)
            uri = _write_parquet_encrypted(store, session_id, "audio", rows)
            if uri:
                outputs.append(uri)
                for i, _ in enumerate(rows):
                    manifest.append({"session_id": session_id, "modality": "audio", "uri": uri, "row": i})

    # TEXT
    if cfg.get("ingest", {}).get("text", {}).get("enabled", False) and req.inputs.get("text_dir"):
        tdir = Path(req.inputs["text_dir"])
        rows = process_text_sources(tdir, cfg, session_id)
        uri = _write_parquet_encrypted(store, session_id, "text", rows)
        if uri:
            outputs.append(uri)
            for i, _ in enumerate(rows):
                manifest.append({"session_id": session_id, "modality": "text", "uri": uri, "row": i})

    # Write manifest & receipt (encrypted)
    manifest_bytes = "\n".join(json.dumps(m) for m in manifest).encode()
    mrel = f"{session_id}/manifest/{datetime.utcnow().strftime('%Y-%m-%d/%H')}.jsonl.enc"
    manifest_uri = store.encrypt_write(mrel, manifest_bytes)

    receipt = make_receipt(
        agent="local-data-agent",
        session_id=session_id,
        op="preprocess",
        params={"mode": req.mode, "config_uri": req.config_uri},
        outputs=[manifest_uri] + outputs
    )
    rrel = f"{session_id}/receipts/{datetime.utcnow().strftime('%Y-%m-%d/%H')}.json.enc"
    receipt_uri = store.encrypt_write(rrel, json.dumps(receipt).encode())

    return {
        "session_id": session_id,
        "artifact_manifest": manifest_uri,
        "receipts": [receipt_uri],
        "count": len(manifest)
    }
