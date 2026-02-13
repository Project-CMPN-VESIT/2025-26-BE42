# LDA/app/pipelines/text.py
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    USE_SPACY = True
except Exception:
    USE_SPACY = False

from core.centralized_secure_store import SecureStore
from core.centralised_receipts import CentralReceiptManager


class TextPreprocessor:
    def __init__(self, storage: SecureStore, out_dir: Path, agent: str = "lda-text-processor"):
        self.storage = storage
        # ensure storage has agent set consistently for encryption key derivation
        if not getattr(self.storage, "agent", None):
            # try set attribute if not present
            try:
                self.storage.agent = agent
            except Exception:
                pass
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.receipt_mgr = CentralReceiptManager(agent=agent)

    def scrub_pii(self, text: str) -> str:
        """Remove obvious PII using regex, optionally reinforce with spaCy NER."""
        # basic regex scrubs
        text = re.sub(r"\b\d{10}\b", "[PHONE]", text)   # phone numbers
        text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL]", text)

        if USE_SPACY:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "GPE", "ORG"]:
                    text = text.replace(ent.text, f"[{ent.label_}]")
        return text

    def process_text(self, text: str, session_id: str, source: str = "user") -> Dict[str, Any]:
        """Process a single text, save encrypted copy and return receipt + metadata row."""
        clean_text = self.scrub_pii(text)
        record = {
            "session_id": session_id,
            "source": source,
            "text": clean_text
        }

        # deterministic filename hash for storage
        h = hashlib.sha256(clean_text.encode()).hexdigest()[:16]
        fpath = self.out_dir / f"{h}.json"

        # Save encrypted JSON (returns file:// URI)
        uri = self.storage.encrypt_write(f"file://{fpath}", json.dumps(record).encode())

        # Create receipt (local receipt manager)
        receipt = self.receipt_mgr.create_receipt(
            session_id=session_id,
            operation="text_process",
            params={"source": source},
            outputs=[uri]
        )

        # Also save receipt into secure store (so other agents can find it)
        receipt_rel = self.out_dir / f"{h}_text.json.enc"
        receipt_uri = f"file://{receipt_rel}"
        self.storage.encrypt_write(receipt_uri, json.dumps(receipt).encode())

        # Return both row (suitable for parquet) and receipt info
        row = {
            "session_id": session_id,
            "modality": "text",
            "source": source,
            "text": clean_text,
            "encrypted_uri": uri,
            "receipt_uri": receipt_uri
        }

        return {"row": row, "receipt": receipt, "receipt_uri": receipt_uri}

    def process_asr_output(self, asr_result: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Hook for ASR pipeline.
        Input: {
          "text": "transcript string",
          "confidence": float (optional),
          "segments": [...] (optional, per-chunk metadata)
        }
        """
        text = asr_result.get("text", "")
        result = self.process_text(text, session_id=session_id, source="asr")
        result["receipt"]["confidence"] = asr_result.get("confidence")
        return result


def process_text_file(
    text_path: str,
    storage: SecureStore,
    out_dir: str,
    session_id: Optional[str] = None,
    from_asr: Optional[bool] = False
) -> List[Dict[str, Any]]:
    """
    Process a single text file _or_ a directory of text files and return list of rows (dictionaries)
    suitable for writing to parquet by the main pipeline.
    - If text_path is a directory: iterate over *.txt / *.csv files and produce rows
    - If text_path is a file: process it and return a single-row list
    """
    tp = TextPreprocessor(storage, Path(out_dir))
    p = Path(text_path)
    rows = []

    # if session_id not provided, derive one deterministically from path
    if session_id is None:
        session_id = hashlib.sha256(str(p).encode()).hexdigest()[:12]

    if p.is_dir():
        # find likely transcript/text files
        for f in sorted(p.glob("*")):
            if f.is_file() and f.suffix.lower() in {".txt", ".csv", ".json"}:
                try:
                    if from_asr and f.suffix.lower() == ".json":
                        asr_result = json.loads(f.read_text(encoding="utf-8"))
                        res = tp.process_asr_output(asr_result, session_id=session_id)
                    else:
                        raw = f.read_text(encoding="utf-8")
                        res = tp.process_text(raw, session_id=session_id, source="file")
                    rows.append(res["row"])
                except Exception:
                    continue
    else:
        # single file
        if from_asr and p.suffix.lower() == ".json":
            asr_result = json.loads(p.read_text(encoding="utf-8"))
            res = tp.process_asr_output(asr_result, session_id=session_id)
        else:
            raw = p.read_text(encoding="utf-8")
            res = tp.process_text(raw, session_id=session_id, source="file")
        rows.append(res["row"])

    return rows
