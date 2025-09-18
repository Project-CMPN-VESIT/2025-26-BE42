import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    USE_SPACY = True
except Exception:
    USE_SPACY = False

from centralized_secure_store import SecureStore
from centralised_receipts import CentralReceiptManager


class TextPreprocessor:
    def __init__(self, storage: SecureStore, out_dir: Path, agent: str = "lda-text-processor"):
        self.storage = storage
        storage.agent = agent
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

    def process_text(self, text: str, source: str = "user") -> Dict[str, Any]:
        """Process raw text or ASR transcript and save securely."""
        clean_text = self.scrub_pii(text)
        record = {
            "source": source,
            "text": clean_text
        }

        # filename hash for deterministic storage
        h = hashlib.sha256(clean_text.encode()).hexdigest()[:16]
        fpath = self.out_dir / f"{h}.json"

        # Save encrypted JSON
        uri = self.storage.encrypt_write(f"file://{fpath}", json.dumps(record).encode())

        # Create + encrypt receipt
        receipt = self.receipt_mgr.create_receipt(
            session_id=h,
            operation="text_process",
            params={"source": source},
            outputs=[uri]
        )

        receipt_uri = f"file://{self.storage.root / 'receipts' / f'{h}_text.json.enc'}"
        self.storage.encrypt_write(receipt_uri, json.dumps(receipt).encode())

        return {"receipt_uri": receipt_uri, "receipt": receipt}

    def process_asr_output(self, asr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook for ASR pipeline.
        Input: {
          "text": "transcript string",
          "confidence": float (optional),
          "segments": [...] (optional, per-chunk metadata)
        }
        """
        text = asr_result.get("text", "")
        result = self.process_text(text, source="asr")
        result["receipt"]["confidence"] = asr_result.get("confidence")
        return result


def process_text_file(
    text_path: str,
    storage: SecureStore,
    out_dir: str,
    from_asr: Optional[bool] = False
) -> Dict[str, Any]:
    """Convenience wrapper to process a raw .txt or ASR JSON file."""
    tp = TextPreprocessor(storage, Path(out_dir))
    p = Path(text_path)

    if from_asr:
        asr_result = json.loads(p.read_text())
        return tp.process_asr_output(asr_result)
    else:
        raw = p.read_text()
        return tp.process_text(raw, source="file")
