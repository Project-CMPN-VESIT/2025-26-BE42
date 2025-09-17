# lda/main.py

from pydantic import BaseModel
from typing import Dict, Any, List
from pathlib import Path
import yaml, json, time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

from app.security.secure_store import SecureStore
from app.pipelines.video import process_video_file
from app.pipelines.audio import process_audio_file
from app.pipelines.text import process_text_file
from app.pipelines.session_processor import process_session_file

# 👇 centralized receipts
from ...centralised_receipts import CentralReceiptManager


class PreprocessRequest(BaseModel):
    mode: str  # "batch" | "interactive" | "continuous" | "session" | "text"
    inputs: Dict[str, str]
    config_uri: str


def _load_config(uri: str) -> dict:
    assert uri.startswith("file://"), "Only file:// URIs are supported for config"
    p = Path(uri[len("file://"):])
    with open(p, "r") as f:
        return yaml.safe_load(f)


def _write_parquet_encrypted(
    store: SecureStore, session_id: str, modality: str, rows: List[Dict[str, Any]]
) -> str:
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    payload = buf.getvalue().to_pybytes()
    rel = f"{session_id}/{modality}/{datetime.utcnow().strftime('%Y-%m-%d/%H')}.parquet.enc"
    return store.encrypt_write(f"file://{store.root / rel}", payload)


def preprocess(req: PreprocessRequest) -> Dict[str, Any]:
    """
    Pure function (no FastAPI).
    Processes inputs according to req.mode.
    """
    cfg = _load_config(req.config_uri)
    store = SecureStore(cfg["storage"]["root"])
    openface_bin = cfg["ingest"]["video"]["params"]["openface"]["binary_path"]
    haar_path = cfg["ingest"]["video"]["params"]["openface"].get("haar_path")
    session_id = f"sess-{int(time.time())}"
    outputs, manifest = [], []

    mode = req.mode.lower()

    # -----------------------------
    # SESSION / CONTINUOUS
    # -----------------------------
    if mode in ("session", "continuous"):
        if req.inputs.get("video_dir"):
            vdir = Path(req.inputs["video_dir"])
            for p in sorted(vdir.glob("*.mp4")):
                work_dir = (
                    Path(cfg.get("storage", {}).get("root", "./secure_store"))
                    / session_id
                    / p.stem
                )
                work_dir.mkdir(parents=True, exist_ok=True)

                audio_path = req.inputs.get("audio_dir")
                audio_file = None
                if audio_path:
                    candidate = Path(audio_path) / f"{p.stem}.wav"
                    if candidate.exists():
                        audio_file = str(candidate)

                text_input = None
                text_dir = req.inputs.get("text_dir")
                if text_dir:
                    candidate_txt = Path(text_dir) / f"{p.stem}.txt"
                    if candidate_txt.exists():
                        text_input = candidate_txt.read_text(encoding="utf-8")

                rows, artifacts, receipts = process_session_file(
                    session_id=session_id,
                    cfg=cfg,
                    work_dir=work_dir,
                    video_path=str(p),
                    audio_path=audio_file,
                    text_input=text_input,
                    mode=mode,
                    roles=None,
                )

                uri = _write_parquet_encrypted(store, session_id, "session", rows)
                if uri:
                    outputs.append(uri)
                    for i, _ in enumerate(rows):
                        manifest.append(
                            {
                                "session_id": session_id,
                                "modality": "session",
                                "uri": uri,
                                "row": i,
                            }
                        )
        else:
            raise RuntimeError(
                "mode 'session' or 'continuous' requires 'video_dir' in inputs"
            )

    # -----------------------------
    # BATCH
    # -----------------------------
    elif mode == "batch":
        # VIDEO
        if cfg.get("ingest", {}).get("video", {}).get("enabled", False) and req.inputs.get("video_dir"):
            vdir = Path(req.inputs["video_dir"])
            for p in sorted(vdir.glob("*.mp4")):
                rows = process_video_file(str(p), cfg, session_id, openface_bin, haar_path)
                uri = _write_parquet_encrypted(store, session_id, "video", rows)
                if uri:
                    outputs.append(uri)
                    for i, _ in enumerate(rows):
                        manifest.append(
                            {
                                "session_id": session_id,
                                "modality": "video",
                                "uri": uri,
                                "row": i,
                            }
                        )

        # AUDIO
        if cfg.get("ingest", {}).get("audio", {}).get("enabled", False) and req.inputs.get("audio_dir"):
            adir = Path(req.inputs["audio_dir"])
            for p in sorted(adir.glob("*.wav")):
                rows = process_audio_file(p, cfg, session_id)
                uri = _write_parquet_encrypted(store, session_id, "audio", rows)
                if uri:
                    outputs.append(uri)
                    for i, _ in enumerate(rows):
                        manifest.append(
                            {
                                "session_id": session_id,
                                "modality": "audio",
                                "uri": uri,
                                "row": i,
                            }
                        )

        # TEXT
        if cfg.get("ingest", {}).get("text", {}).get("enabled", False) and req.inputs.get("text_dir"):
            tdir = Path(req.inputs["text_dir"])
            rows = process_text_file(tdir, cfg, session_id)
            uri = _write_parquet_encrypted(store, session_id, "text", rows)
            if uri:
                outputs.append(uri)
                for i, _ in enumerate(rows):
                    manifest.append(
                        {
                            "session_id": session_id,
                            "modality": "text",
                            "uri": uri,
                            "row": i,
                        }
                    )

    # -----------------------------
    # TEXT ONLY
    # -----------------------------
    elif mode == "text":
        if req.inputs.get("text_dir"):
            tdir = Path(req.inputs["text_dir"])
            rows = process_text_file(tdir, cfg, session_id)
            uri = _write_parquet_encrypted(store, session_id, "text", rows)
            if uri:
                outputs.append(uri)
                for i, _ in enumerate(rows):
                    manifest.append(
                        {
                            "session_id": session_id,
                            "modality": "text",
                            "uri": uri,
                            "row": i,
                        }
                    )
        else:
            raise RuntimeError("mode 'text' requires 'text_dir' in inputs")

    else:
        raise RuntimeError(f"Unknown mode '{req.mode}'")

    # -----------------------------
    # MANIFEST + RECEIPT
    # -----------------------------
    manifest_bytes = "\n".join(json.dumps(m) for m in manifest).encode()
    mrel = f"{session_id}/manifest/{datetime.utcnow().strftime('%Y-%m-%d/%H')}.jsonl.enc"
    manifest_uri = store.encrypt_write(f"file://{store.root / mrel}", manifest_bytes)

    rm = CentralReceiptManager()
    receipt = rm.create_receipt(
        agent="local-data-agent",
        session_id=session_id,
        operation="preprocess",
        params={"mode": req.mode, "config_uri": req.config_uri},
        outputs=[manifest_uri] + outputs,
    )
    rrel = f"{session_id}/receipts/{datetime.utcnow().strftime('%Y-%m-%d/%H')}.json.enc"
    receipt_uri = store.encrypt_write(
        f"file://{store.root / rrel}", json.dumps(receipt).encode()
    )

    return {
        "session_id": session_id,
        "artifact_manifest": manifest_uri,
        "receipts": [receipt_uri],
        "count": len(manifest),
    }
