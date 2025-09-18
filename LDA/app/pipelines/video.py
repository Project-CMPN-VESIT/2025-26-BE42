import cv2
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import json

from centralized_secure_store import SecureStore
from centralised_receipts import CentralReceiptManager


class VideoProcessor:
    def __init__(self, storage: SecureStore, out_dir: Path, openface_bin: str, haar_xml: str):
        self.storage = storage
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.openface_bin = Path(openface_bin)
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        self.receipt_mgr = CentralReceiptManager(agent="lda-video-processor")

    def detect_faces(self, frame):
        """Detect faces in a single frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces

    def blur_faces(self, frame, faces):
        """Apply blur on detected faces."""
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            roi_blur = cv2.GaussianBlur(roi, (99, 99), 30)
            frame[y:y+h, x:x+w] = roi_blur
        return frame

    def process_video(self, video_path: str) -> Dict[str, Any]:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        tmp_dir = Path(tempfile.mkdtemp())
        cropped_dir = tmp_dir / "crops"
        cropped_dir.mkdir(parents=True, exist_ok=True)

        out_path = self.out_dir / f"{video_path.stem}_blurred.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        frame_idx = 0
        face_found = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.detect_faces(frame)
            if len(faces) > 0:
                face_found = True
                # crop detected faces into temp folder for OpenFace
                for i, (x, y, w_, h_) in enumerate(faces):
                    crop_file = cropped_dir / f"frame{frame_idx}_face{i}.png"
                    cv2.imwrite(str(crop_file), frame[y:y+h_, x:x+w_])

            # blur before saving
            frame = self.blur_faces(frame, faces)
            out_writer.write(frame)
            frame_idx += 1

        cap.release()
        out_writer.release()

        # run OpenFace on crops or fallback to original video
        feature_out = tmp_dir / "openface_out"
        feature_out.mkdir(parents=True, exist_ok=True)

        if face_found:
            cmd = [
                str(self.openface_bin),
                "-fdir", str(cropped_dir),
                "-out_dir", str(feature_out)
            ]
        else:
            cmd = [
                str(self.openface_bin),
                "-f", str(video_path),
                "-out_dir", str(feature_out)
            ]

        subprocess.run(cmd, check=True)

        # Encrypt blurred video and OpenFace features
        uri_video = self.storage.encrypt_write(f"file://{out_path}", out_path.read_bytes())

        features_csv = next(feature_out.glob("*.csv"), None)
        uri_features = None
        if features_csv:
            uri_features = self.storage.encrypt_write(
                f"file://{features_csv}", features_csv.read_bytes()
            )

        shutil.rmtree(tmp_dir)

        # Create + encrypt receipt
        receipt = self.receipt_mgr.create_receipt(
            agent="lda-video-processor",
            session_id=video_path.stem,
            operation="video_process",
            params={"faces_detected": face_found},
            outputs=[uri_video, uri_features] if uri_features else [uri_video],
        )
        receipt_uri = f"file://{self.storage.root / video_path.stem / 'receipts' / 'video.json.enc'}"
        self.storage.encrypt_write(receipt_uri, json.dumps(receipt).encode())

        return {"receipt_uri": receipt_uri, "receipt": receipt}


def process_video_file(video_path: str, storage: SecureStore, out_dir: str, openface_bin: str, haar_xml: str):
    vp = VideoProcessor(storage, Path(out_dir), openface_bin, haar_xml)
    return vp.process_video(video_path)
