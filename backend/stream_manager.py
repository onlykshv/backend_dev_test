"""
stream_manager.py - RTSP stream manager with multi-threaded processing.

Manages multiple RTSP camera streams, each in its own thread.
Features:
- Automatic reconnection on stream disconnect
- Thread-safe frame access for MJPEG streaming
- Per-stream status tracking (online/offline, last frame timestamp)
- Configurable detection interval
"""

import os
import cv2
import time
import threading
import logging
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field

from detector import ObjectDetector

logger = logging.getLogger(__name__)

# How many seconds to wait before attempting reconnection
RECONNECT_DELAY = int(os.environ.get("RECONNECT_DELAY", "5"))

# Run detection every N frames (to reduce CPU load)
DETECTION_INTERVAL = int(os.environ.get("DETECTION_INTERVAL", "15"))


@dataclass
class StreamInfo:
    """Holds metadata and state for a single RTSP stream."""
    name: str
    url: str
    status: str = "offline"             # "online" or "offline"
    last_frame_time: Optional[str] = None  # ISO format timestamp
    frame_count: int = 0
    error: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _latest_frame: Optional[object] = None  # Latest captured frame (numpy array)
    _latest_annotated: Optional[object] = None  # Latest annotated frame

    def set_frame(self, frame, annotated=None):
        """Thread-safe frame update."""
        with self._lock:
            self._latest_frame = frame
            self._latest_annotated = annotated if annotated is not None else frame
            self.last_frame_time = datetime.now().isoformat()
            self.frame_count += 1

    def get_frame(self, annotated: bool = True):
        """Thread-safe frame retrieval."""
        with self._lock:
            if annotated and self._latest_annotated is not None:
                return self._latest_annotated.copy()
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def to_dict(self) -> dict:
        """Serialize stream info for API responses."""
        return {
            "name": self.name,
            "url": self.url,
            "status": self.status,
            "last_frame_time": self.last_frame_time,
            "frame_count": self.frame_count,
            "error": self.error,
        }


class StreamManager:
    """
    Manages multiple RTSP streams with automatic reconnection.

    Each stream runs in a dedicated daemon thread that:
    1. Connects to the RTSP URL
    2. Reads frames continuously
    3. Runs object detection at configured intervals
    4. Stores annotated frames for MJPEG streaming
    5. Automatically reconnects on failure
    """

    def __init__(self):
        self._streams: Dict[str, StreamInfo] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._detector = ObjectDetector()
        self._running = True
        self._lock = threading.Lock()

    def add_stream(self, name: str, url: str):
        """
        Register and start processing an RTSP stream.

        Args:
            name: Human-readable camera identifier
            url: RTSP stream URL
        """
        with self._lock:
            if name in self._streams:
                logger.warning(f"Stream '{name}' already exists, skipping.")
                return

            stream_info = StreamInfo(name=name, url=url)
            self._streams[name] = stream_info

            # Start stream processing in a dedicated thread
            thread = threading.Thread(
                target=self._stream_worker,
                args=(stream_info,),
                daemon=True,
                name=f"stream-{name}",
            )
            self._threads[name] = thread
            thread.start()

            logger.info(f"Stream '{name}' registered: {url}")

    def _stream_worker(self, stream: StreamInfo):
        """
        Worker thread for a single RTSP stream.
        Handles connection, frame reading, detection, and reconnection.
        """
        while self._running:
            cap = None
            try:
                logger.info(f"[{stream.name}] Connecting to {stream.url}")
                stream.status = "connecting"
                stream.error = None

                # Configure OpenCV capture for RTSP
                cap = cv2.VideoCapture(stream.url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Set timeout for RTSP connection
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)

                if not cap.isOpened():
                    raise ConnectionError(
                        f"Failed to open RTSP stream: {stream.url}"
                    )

                stream.status = "online"
                logger.info(f"[{stream.name}] Stream connected successfully.")
                frame_idx = 0
                consecutive_failures = 0

                while self._running and cap.isOpened():
                    ret, frame = cap.read()

                    if not ret:
                        consecutive_failures += 1
                        if consecutive_failures > 30:
                            raise ConnectionError("Too many consecutive read failures")
                        time.sleep(0.1)
                        continue

                    consecutive_failures = 0
                    frame_idx += 1

                    # Run detection at configured interval
                    detections = []
                    if frame_idx % DETECTION_INTERVAL == 0:
                        try:
                            detections = self._detector.detect_and_save(
                                frame, stream.name
                            )
                        except Exception as e:
                            logger.error(
                                f"[{stream.name}] Detection error: {e}"
                            )

                    # Annotate frame with detection bounding boxes
                    annotated = frame.copy()
                    if detections:
                        self._detector.annotate_frame(annotated, detections)

                    # Store frame for MJPEG streaming
                    stream.set_frame(frame, annotated)

                    # Small sleep to prevent excessive CPU usage
                    time.sleep(0.03)

            except Exception as e:
                stream.status = "offline"
                stream.error = str(e)
                logger.error(
                    f"[{stream.name}] Stream error: {e}. "
                    f"Reconnecting in {RECONNECT_DELAY}s..."
                )

            finally:
                if cap is not None:
                    cap.release()

            # Wait before reconnection attempt
            if self._running:
                time.sleep(RECONNECT_DELAY)

    def get_stream(self, name: str) -> Optional[StreamInfo]:
        """Get stream info by name."""
        return self._streams.get(name)

    def get_all_streams(self) -> Dict[str, StreamInfo]:
        """Get all registered streams."""
        return dict(self._streams)

    def get_streams_list(self) -> list:
        """Get serialized list of all streams for API responses."""
        return [s.to_dict() for s in self._streams.values()]

    def generate_mjpeg(self, stream_name: str):
        """
        Generator that yields MJPEG frames for HTTP streaming.

        Args:
            stream_name: Name of the stream to serve

        Yields:
            MJPEG frame bytes suitable for multipart/x-mixed-replace response
        """
        stream = self._streams.get(stream_name)
        if stream is None:
            return

        while self._running:
            frame = stream.get_frame(annotated=True)

            if frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )

            time.sleep(0.066)  # ~15 FPS for MJPEG output

    def stop(self):
        """Stop all stream processing threads."""
        logger.info("Stopping all streams...")
        self._running = False


def load_streams_from_env() -> list:
    """
    Parse RTSP stream configuration from environment variables.

    Supported formats:
        RTSP_STREAMS=name1,url1;name2,url2
        Or individual: RTSP_URL_1=rtsp://..., RTSP_NAME_1=Camera1, etc.

    Returns:
        List of (name, url) tuples
    """
    streams = []

    # Format 1: RTSP_STREAMS="Camera1,rtsp://...;Camera2,rtsp://..."
    rtsp_streams = os.environ.get("RTSP_STREAMS", "")
    if rtsp_streams:
        for entry in rtsp_streams.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(",", 1)
            if len(parts) == 2:
                name, url = parts[0].strip(), parts[1].strip()
                streams.append((name, url))
            else:
                # If only URL provided, generate a name
                streams.append((f"Camera_{len(streams)+1}", parts[0].strip()))

    # Format 2: RTSP_URL_1, RTSP_NAME_1, RTSP_URL_2, RTSP_NAME_2, ...
    idx = 1
    while True:
        url = os.environ.get(f"RTSP_URL_{idx}")
        if url is None:
            break
        name = os.environ.get(f"RTSP_NAME_{idx}", f"Camera_{idx}")
        streams.append((name, url))
        idx += 1

    if not streams:
        logger.warning(
            "No RTSP streams configured. Set RTSP_STREAMS or "
            "RTSP_URL_1/RTSP_NAME_1 environment variables."
        )

    return streams
