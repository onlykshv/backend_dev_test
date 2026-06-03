"""
app.py - FastAPI application entry point.

Serves the web dashboard, MJPEG video streams, REST APIs,
and static detection images using Jinja2 templates.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional

try:
    from dotenv import load_dotenv
    # Find .env in project root (one level up from backend/)
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(dotenv_path)
except ImportError:
    pass


from fastapi import FastAPI, Request, Query
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add backend directory to path so modules can be imported
sys.path.insert(0, os.path.dirname(__file__))

# Project root (one level up from backend/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from stream_manager import StreamManager, load_streams_from_env
from database import DetectionDB

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Globals ──────────────────────────────────────────────────
stream_manager: Optional[StreamManager] = None
db: Optional[DetectionDB] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler: startup and shutdown."""
    global stream_manager, db

    logger.info("=" * 60)
    logger.info("  RTSP Object Detection System - Starting Up")
    logger.info("=" * 60)

    # Initialize database
    db = DetectionDB()
    logger.info("Database initialized.")

    # Initialize stream manager and load configured streams
    stream_manager = StreamManager()
    streams = load_streams_from_env()

    if not streams:
        logger.warning("No RTSP streams configured! Add streams to .env file.")
    else:
        for name, url in streams:
            stream_manager.add_stream(name, url)
            logger.info(f"  ├── {name}: {url}")
        logger.info(f"  └── Total: {len(streams)} stream(s) registered")

    logger.info("System ready. Dashboard: http://0.0.0.0:8000")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down stream manager...")
    if stream_manager:
        stream_manager.stop()
    logger.info("Shutdown complete.")


# ── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title="RTSP Object Detection System",
    description="Real-time RTSP stream monitoring with YOLOv8 object detection",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Static Files & Templates ────────────────────────────────
# Serve detection images from persistent volume
DETECTIONS_DIR = os.environ.get("DETECTIONS_DIR", os.path.join(PROJECT_ROOT, "detections", "images"))
os.makedirs(DETECTIONS_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "detections"), exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "static")), name="static")
app.mount(
    "/detections/images",
    StaticFiles(directory=DETECTIONS_DIR),
    name="detection_images",
)

# Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "templates"))


# ═══════════════════════════════════════════════════════════
#  WEB ROUTES
# ═══════════════════════════════════════════════════════════


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Main dashboard page.
    Displays all camera streams, status, and detection gallery.
    """
    streams = stream_manager.get_streams_list() if stream_manager else []
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "streams": streams,
        },
    )


@app.get("/video/{stream_name}")
async def video_feed(stream_name: str):
    """
    MJPEG video stream endpoint.
    Returns a multipart response with continuous JPEG frames.
    """
    if stream_manager is None:
        return JSONResponse(
            {"error": "Stream manager not initialized"}, status_code=503
        )

    stream = stream_manager.get_stream(stream_name)
    if stream is None:
        return JSONResponse(
            {"error": f"Stream '{stream_name}' not found"}, status_code=404
        )

    return StreamingResponse(
        stream_manager.generate_mjpeg(stream_name),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ═══════════════════════════════════════════════════════════
#  REST API ENDPOINTS
# ═══════════════════════════════════════════════════════════


@app.get("/api/streams")
async def api_get_streams():
    """
    GET /api/streams - Retrieve status of all configured streams.

    Returns:
        JSON list of stream objects with:
        - name: Camera identifier
        - url: RTSP URL
        - status: "online" | "offline" | "connecting"
        - last_frame_time: ISO timestamp of last received frame
        - frame_count: Total frames received
        - error: Error message if offline
    """
    if stream_manager is None:
        return JSONResponse({"streams": []})

    return JSONResponse({"streams": stream_manager.get_streams_list()})


@app.get("/api/detections")
async def api_get_detections(
    limit: int = Query(default=50, ge=1, le=500, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Skip N results"),
    camera: Optional[str] = Query(default=None, description="Filter by camera"),
    class_name: Optional[str] = Query(
        default=None, description="Filter by class name"
    ),
):
    """
    GET /api/detections - Retrieve detection records.

    Query Parameters:
        - limit (int): Maximum number of results (1-500, default 50)
        - offset (int): Pagination offset (default 0)
        - camera (str): Filter by camera name
        - class_name (str): Filter by object class

    Returns:
        JSON object with:
        - detections: List of detection records
        - total: Total matching records
        - limit: Applied limit
        - offset: Applied offset
    """
    if db is None:
        return JSONResponse({"detections": [], "total": 0})

    detections = db.get_detections(
        limit=limit, offset=offset, camera=camera, class_name=class_name
    )
    total = db.get_detection_count(camera=camera, class_name=class_name)

    return JSONResponse(
        {
            "detections": detections,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@app.get("/api/detections/{detection_id}")
async def api_get_detection(detection_id: int):
    """
    GET /api/detections/{id} - Retrieve a single detection by ID.
    """
    if db is None:
        return JSONResponse({"error": "Database not initialized"}, status_code=503)

    detection = db.get_detection_by_id(detection_id)
    if detection is None:
        return JSONResponse(
            {"error": f"Detection {detection_id} not found"}, status_code=404
        )

    return JSONResponse({"detection": detection})


@app.get("/api/health")
async def health_check():
    """Health check endpoint for Docker and monitoring."""
    stream_count = len(stream_manager.get_all_streams()) if stream_manager else 0
    online = sum(
        1 for s in (stream_manager.get_all_streams().values() if stream_manager else [])
        if s.status == "online"
    )

    return JSONResponse(
        {
            "status": "healthy",
            "streams": {"total": stream_count, "online": online},
            "detections": db.get_detection_count() if db else 0,
        }
    )


# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("APP_PORT", "8000"))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
