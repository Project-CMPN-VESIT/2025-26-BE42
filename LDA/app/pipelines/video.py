# app/pipelines/video.py
import cv2
import numpy as np
from pathlib import Path
from ..utils.receipts import ReceiptManager
import mediapipe as mp

class VideoPreprocessor:
    def __init__(self, output_dir: str, receipt_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.receipts = ReceiptManager(receipt_dir)
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_video(self, video_path: str) -> str:
        """Process video: detect face, frontalize, blur identifiable regions, store anonymized output."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video {video_path}")

        # Prepare output file
        out_path = self.output_dir / f"processed_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb_frame)

            # Blur entire face region for anonymity
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    points = [(int(landmark.x * width), int(landmark.y * height))
                              for landmark in face_landmarks.landmark]
                    hull = cv2.convexHull(np.array(points))
                    cv2.fillConvexPoly(mask, hull, 255)

                    blurred = cv2.GaussianBlur(frame, (99, 99), 30)
                    frame = np.where(mask[:, :, None] == 255, blurred, frame)

            writer.write(frame)

        cap.release()
        writer.release()

        # Generate receipt for audit
        receipt_path = self.receipts.create_receipt(
            operation="video_preprocessing",
            input_meta={"source": video_path, "fps": fps, "resolution": (width, height)},
            output_uri=str(out_path)
        )

        return str(out_path), receipt_path
