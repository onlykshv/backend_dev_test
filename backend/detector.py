"""
detector.py - YOLOv8n object detection module.

Loads the YOLOv8n model and runs inference on video frames.
Filters detections to only the required classes:
person, car, motorcycle, bicycle, bus, truck.

Crops detected objects and saves them to persistent storage.
"""

import os
import cv2
import threading
import logging
from datetime import datetime
from typing import List, Dict, Optional
# Configure PyTorch 2.6+ to allow loading Ultralytics model weights
try:
    import torch
    import ultralytics.nn.tasks
    # Monkey-patch torch.load to set weights_only=False by default for loading YOLO models
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs or kwargs['weights_only'] is True:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except Exception as e:
    logging.getLogger(__name__).warning(f"Could not patch torch.load for PyTorch 2.6+: {e}")

from ultralytics import YOLO


from database import DetectionDB

logger = logging.getLogger(__name__)

# Project root (one level up from backend/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory for saving cropped detection images
DETECTIONS_DIR = os.environ.get("DETECTIONS_DIR", os.path.join(PROJECT_ROOT, "detections", "images"))

# Minimum confidence threshold for detections
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

# Minimum time between saving detections of the same class from the same camera (seconds)
# This prevents flooding the database with duplicate detections
COOLDOWN_SECONDS = float(os.environ.get("DETECTION_COOLDOWN", "5"))

# COCO class IDs for the required object classes
# Reference: https://docs.ultralytics.com/datasets/detect/coco/
ALLOWED_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class ObjectDetector:
    """
    YOLOv8n-based object detector.

    Runs inference on frames, filters for allowed classes,
    crops detected objects, and persists results to the database.
    """

    def __init__(self):
        """Initialize the detector with YOLOv8n model."""
        self._model = None
        self._model_lock = threading.Lock()
        self._db = DetectionDB()
        self._cooldown_tracker: Dict[str, datetime] = {}
        self._cooldown_lock = threading.Lock()

        # Ensure detection images directory exists
        os.makedirs(DETECTIONS_DIR, exist_ok=True)

        logger.info(
            f"ObjectDetector initialized | threshold={CONFIDENCE_THRESHOLD} "
            f"| cooldown={COOLDOWN_SECONDS}s"
        )

    @property
    def model(self):
        """Lazy-load the YOLO model (thread-safe)."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    logger.info("Loading YOLOv8n model...")
                    self._model = YOLO("yolov8n.pt")
                    logger.info("YOLOv8n model loaded successfully.")
        return self._model

    def _is_on_cooldown(self, camera_name: str, class_name: str) -> bool:
        """
        Check if a detection of the same class from the same camera
        is within the cooldown period.
        """
        key = f"{camera_name}:{class_name}"
        now = datetime.now()

        with self._cooldown_lock:
            last_time = self._cooldown_tracker.get(key)
            if last_time and (now - last_time).total_seconds() < COOLDOWN_SECONDS:
                return True
            self._cooldown_tracker[key] = now
            return False

    def detect_and_save(
        self, frame, camera_name: str
    ) -> List[Dict]:
        """
        Run object detection on a frame and save valid detections.

        Args:
            frame: OpenCV image (numpy array) to run detection on
            camera_name: Name of the source camera

        Returns:
            List of detection result dictionaries containing:
                - class_name, confidence, bbox, image_path, timestamp
        """
        if frame is None or frame.size == 0:
            return []

        # Run YOLOv8 inference
        results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])

                # Filter: only process allowed classes
                if class_id not in ALLOWED_CLASSES:
                    continue

                class_name = ALLOWED_CLASSES[class_id]
                confidence = float(box.conf[0])

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Clamp coordinates to frame boundaries
                h, w = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Skip if bounding box is too small
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                # Check cooldown to avoid duplicate detections
                if self._is_on_cooldown(camera_name, class_name):
                    # Still return for drawing bounding boxes, but don't save
                    detections.append(
                        {
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": (x1, y1, x2, y2),
                            "saved": False,
                        }
                    )
                    continue

                # Crop detected object from frame
                cropped = frame[y1:y2, x1:x2].copy()

                # Generate unique filename
                timestamp = datetime.now()
                ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{camera_name}_{class_name}_{ts_str}.jpg"
                filepath = os.path.join(DETECTIONS_DIR, filename)

                # Save cropped image
                cv2.imwrite(filepath, cropped)

                # Relative path for web serving
                relative_path = f"/detections/images/{filename}"

                # Store in database
                detection_id = self._db.add_detection(
                    timestamp=timestamp.isoformat(),
                    class_name=class_name,
                    confidence=round(confidence, 4),
                    camera_name=camera_name,
                    image_path=relative_path,
                    bbox=(x1, y1, x2, y2),
                )

                detections.append(
                    {
                        "id": detection_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                        "image_path": relative_path,
                        "timestamp": timestamp.isoformat(),
                        "saved": True,
                    }
                )

                logger.info(
                    f"Detection saved: {class_name} ({confidence:.2%}) "
                    f"from {camera_name} -> {filename}"
                )

        return detections

    def annotate_frame(self, frame, detections: List[Dict]):
        """
        Draw bounding boxes and labels on a frame for live preview.

        Args:
            frame: OpenCV image to annotate (modified in-place)
            detections: List of detection dicts with bbox, class_name, confidence
        """
        # Color palette for different classes
        colors = {
            "person": (0, 255, 0),       # Green
            "bicycle": (255, 165, 0),     # Orange
            "car": (255, 0, 0),           # Blue (BGR)
            "motorcycle": (0, 255, 255),  # Yellow
            "bus": (255, 0, 255),         # Magenta
            "truck": (0, 165, 255),       # Orange-red
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            color = colors.get(class_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{class_name} {confidence:.0%}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1,
            )
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame
