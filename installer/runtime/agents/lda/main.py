# lda/app/main.py

from pydantic import BaseModel
from typing import Dict, Any, List, Tuple
from pathlib import Path
import yaml, json, time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

# 👇 centralized imports
from centralized_secure_store import SecureStore
from centralised_receipts import CentralReceiptManager

# local pipelines
from LDA.app.pipelines.video import process_video_file
from LDA.app.pipelines.audio import process_audio_file
from LDA.app.pipelines.text import process_text_file
from LDA.app.pipelines.session_processor import process_session_file

from installer.security.integrity import integrity_guard
integrity_guard()

class PreprocessRequest(BaseModel):
    mode: str  # "batch" | "interactive" | "continuous" | "session" | "text"
    inputs: Dict[str, str]
    config_uri: str

def _load_config(uri: str) -> dict:
    assert uri.startswith("file://"), "Only file:// URIs are supported for config"
    p = Path(uri[len("file://"):]).expanduser().resolve()

    with open(p, "r") as f:
        cfg = yaml.safe_load(f)

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            if obj.startswith("~"):
                return str(Path(obj).expanduser().resolve())
            return obj
        return obj

    return _expand(cfg)


def _write_parquet_encrypted(
    store: SecureStore, rm: CentralReceiptManager, session_id: str,
    modality: str, rows: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """
    Writes rows as encrypted parquet into SecureStore and creates a receipt.
    Ensures dict/list fields (e.g., features, derived) are JSON-serialized.
    Returns: (uri, receipt_uri)
    """
    if not rows:
        return "", ""

    # --- normalize rows ---
    norm_rows = []
    for r in rows:
        r = dict(r)  # shallow copy
        for k in ("features", "derived"):
            if isinstance(r.get(k), (dict, list)):
                try:
                    r[k] = json.dumps(r[k])
                except Exception:
                    # fallback: repr to avoid crashes
                    r[k] = repr(r[k])
        norm_rows.append(r)

    # --- DataFrame -> Parquet ---
    df = pd.DataFrame(norm_rows)
    table = pa.Table.from_pandas(df)
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    payload = buf.getvalue().to_pybytes()

    # --- write encrypted parquet ---
    rel = f"{session_id}/{modality}/{datetime.utcnow().strftime('%Y-%m-%d/%H')}.parquet.enc"
    uri = store.encrypt_write(f"file://{store.root / rel}", payload)

    # --- create receipt ---
    receipt = rm.create_receipt(
        agent="local-data-agent",
        session_id=session_id,
        operation=f"preprocess_{modality}",
        params={"rows": len(rows)},
        outputs=[uri],
    )
    rrel = f"{session_id}/receipts/{modality}_{datetime.utcnow().strftime('%Y-%m-%d/%H')}.json.enc"
    receipt_uri = store.encrypt_write(
        f"file://{store.root / rrel}", json.dumps(receipt).encode()
    )

    return uri, receipt_uri

def preprocess(req: PreprocessRequest) -> Dict[str, Any]:
    """
    Pure function (no FastAPI).
    Processes inputs according to req.mode.
    """
    cfg = _load_config(req.config_uri)
    # create SecureStore with explicit named argument 'root'
    store = SecureStore(
        agent="lda-session-processor",
        root=Path(cfg["storage"]["root"]).resolve()
    )
    rm = CentralReceiptManager(agent="lda-session-processor")

    openface_bin = cfg["ingest"]["video"]["params"]["openface"]["binary_path"]
    haar_path = cfg["ingest"]["video"]["params"]["openface"].get("haar_path")
    session_id = f"sess-{int(time.time())}"
    outputs, receipts, manifest = [], [], []

    mode = req.mode.lower()

    # -----------------------------
    # SESSION / CONTINUOUS
    # -----------------------------
    if mode in ("session", "continuous"):
        if req.inputs.get("video_dir"):
            vdir = Path(req.inputs["video_dir"])
            for p in sorted(vdir.glob("*.mp4")):
                work_dir = Path(cfg["storage"]["root"]) / session_id / p.stem
                work_dir.mkdir(parents=True, exist_ok=True)

                audio_file, text_input = None, None
                if req.inputs.get("audio_dir"):
                    candidate = Path(req.inputs["audio_dir"]) / f"{p.stem}.wav"
                    if candidate.exists():
                        audio_file = str(candidate)
                if req.inputs.get("text_dir"):
                    candidate_txt = Path(req.inputs["text_dir"]) / f"{p.stem}.txt"
                    if candidate_txt.exists():
                        text_input = candidate_txt.read_text(encoding="utf-8")

                rows, artifacts, rlist = process_session_file(
                    session_id=session_id,
                    cfg=cfg,
                    work_dir=work_dir,
                    video_path=str(p),
                    audio_path=audio_file,
                    text_input=text_input,
                    mode=mode,
                    roles=None,
                )

                uri, ruri = _write_parquet_encrypted(store, rm, session_id, "session", rows)
                if uri:
                    outputs.append(uri)
                    receipts.append(ruri)
                    for i, _ in enumerate(rows):
                        manifest.append({"session_id": session_id, "modality": "session", "uri": uri, "row": i})
        else:
            raise RuntimeError("mode 'session' or 'continuous' requires 'video_dir' in inputs")

    # -----------------------------
    # BATCH
    # -----------------------------
    elif mode == "batch":
        # VIDEO
        if cfg.get("ingest", {}).get("video", {}).get("enabled", False) and req.inputs.get("video_dir"):
            vdir = Path(req.inputs["video_dir"])
            for p in sorted(vdir.glob("*.mp4")):
                rows = process_video_file(str(p), cfg, session_id, openface_bin, haar_path)
                uri, ruri = _write_parquet_encrypted(store, rm, session_id, "video", rows)
                if uri:
                    outputs.append(uri)
                    receipts.append(ruri)
                    for i, _ in enumerate(rows):
                        manifest.append({"session_id": session_id, "modality": "video", "uri": uri, "row": i})

        # AUDIO
        if cfg.get("ingest", {}).get("audio", {}).get("enabled", False) and req.inputs.get("audio_dir"):
            adir = Path(req.inputs["audio_dir"])
            for p in sorted(adir.glob("*.wav")):
                rows = process_audio_file(p, cfg, session_id)
                uri, ruri = _write_parquet_encrypted(store, rm, session_id, "audio", rows)
                if uri:
                    outputs.append(uri)
                    receipts.append(ruri)
                    for i, _ in enumerate(rows):
                        manifest.append({"session_id": session_id, "modality": "audio", "uri": uri, "row": i})

        # TEXT
        if cfg.get("ingest", {}).get("text", {}).get("enabled", False) and req.inputs.get("text_dir"):
            tdir = Path(req.inputs["text_dir"])
            # pass the SecureStore instance and an out_dir where text preprocessor will write
            out_dir_for_text = Path(cfg["storage"]["root"]) / session_id / "text"
            rows = process_text_file(str(tdir), store, str(out_dir_for_text), session_id=session_id)
            uri, ruri = _write_parquet_encrypted(store, rm, session_id, "text", rows)
            if uri:
                outputs.append(uri)
                receipts.append(ruri)
                for i, _ in enumerate(rows):
                    manifest.append({"session_id": session_id, "modality": "text", "uri": uri, "row": i})

        # Similarly in 'text' mode:
        elif mode == "text":
            if req.inputs.get("text_dir"):
                tdir = Path(req.inputs["text_dir"])
                out_dir_for_text = Path(cfg["storage"]["root"]) / session_id / "text"
                rows = process_text_file(str(tdir), store, str(out_dir_for_text), session_id=session_id)
                uri, ruri = _write_parquet_encrypted(store, rm, session_id, "text", rows)
                if uri:
                    outputs.append(uri)
                    receipts.append(ruri)
                    for i, _ in enumerate(rows):
                        manifest.append({"session_id": session_id, "modality": "text", "uri": uri, "row": i})
            else:
                raise RuntimeError("mode 'text' requires 'text_dir' in inputs")

    # -----------------------------
    # TEXT ONLY
    # -----------------------------
    elif mode == "text":
        if req.inputs.get("text_dir"):
            tdir = Path(req.inputs["text_dir"])
            rows = process_text_file(tdir, cfg, session_id)
            uri, ruri = _write_parquet_encrypted(store, rm, session_id, "text", rows)
            if uri:
                outputs.append(uri)
                receipts.append(ruri)
                for i, _ in enumerate(rows):
                    manifest.append({"session_id": session_id, "modality": "text", "uri": uri, "row": i})
        else:
            raise RuntimeError("mode 'text' requires 'text_dir' in inputs")

    else:
        raise RuntimeError(f"Unknown mode '{req.mode}'")

    # -----------------------------
    # MANIFEST + FINAL RECEIPT
    # -----------------------------
    manifest_bytes = "\n".join(json.dumps(m) for m in manifest).encode()
    mrel = f"{session_id}/manifest/{datetime.utcnow().strftime('%Y-%m-%d/%H')}.jsonl.enc"
    manifest_uri = store.encrypt_write(f"file://{store.root / mrel}", manifest_bytes)

    final_receipt = rm.create_receipt(
        agent="lda-session-processor",
        session_id=session_id,
        operation="preprocess_complete",
        params={"mode": req.mode, "config_uri": req.config_uri},
        outputs=[manifest_uri] + outputs,
    )
    rrel = f"{session_id}/receipts/final_{datetime.utcnow().strftime('%Y-%m-%d/%H')}.json.enc"
    final_receipt_uri = store.encrypt_write(
        f"file://{store.root / rrel}", json.dumps(final_receipt).encode()
    )

    return {
        "session_id": session_id,
        "artifact_manifest": manifest_uri,
        "receipts": receipts + [final_receipt_uri],
        "count": len(manifest),
    }
