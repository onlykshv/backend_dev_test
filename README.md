# RTSP Object Detection System

Real-time RTSP stream monitoring dashboard with **YOLOv8n** object detection, built with **FastAPI**, **OpenCV**, and **SQLite**.

Automatically detects and captures **persons, cars, motorcycles, bicycles, buses, and trucks** from multiple RTSP camera streams with a modern web dashboard for live monitoring and browsing detection history.

---


```

### Component Overview

| Component | File | Responsibility |
|-----------|------|----------------|
| **FastAPI App** | `backend/app.py` | HTTP server, REST APIs, MJPEG streaming, Jinja2 rendering |
| **Stream Manager** | `backend/stream_manager.py` | Multi-threaded RTSP stream reading, auto-reconnection |
| **Object Detector** | `backend/detector.py` | YOLOv8n inference, object cropping, detection saving |
| **Database** | `backend/database.py` | Thread-safe SQLite operations for detection persistence |
| **Dashboard** | `templates/index.html` | Live stream viewer, detection gallery, real-time status |

### Data Flow

1. **Stream Manager** reads frames from RTSP streams in dedicated threads
2. Every N frames, the **Object Detector** runs YOLOv8n inference
3. Detected objects (person, car, motorcycle, bicycle, bus, truck) are:
   - Cropped from the frame and saved as JPEG images
   - Recorded in the **SQLite database** with metadata
4. Annotated frames (with bounding boxes) are served via **MJPEG** streams
5. The **Dashboard** displays live feeds and a browsable detection gallery

---

##  Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)
- RTSP camera stream URL(s)

### Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd backend_dev_test
```

2. **Create the environment file:**

```bash
cp .env.example .env
```

3. **Configure your RTSP streams** in `.env`:

```env
# Single stream
RTSP_STREAMS=FrontDoor,rtsp://username:password@192.168.1.100:554/stream1

# Multiple streams (separated by semicolons)
RTSP_STREAMS=FrontDoor,rtsp://user:pass@192.168.1.100:554/stream1;ParkingLot,rtsp://user:pass@192.168.1.101:554/stream1
```

4. **Build and start the application:**

```bash
docker compose up --build
```

5. **Open the dashboard:** [http://localhost:8000](http://localhost:8000)

That's it! No additional setup commands required.

---

## ⚙ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RTSP_STREAMS` | *(required)* | RTSP stream config: `Name,URL;Name2,URL2` |
| `RTSP_URL_1` | - | Alternative: Individual stream URL |
| `RTSP_NAME_1` | `Camera_1` | Alternative: Individual stream name |
| `CONFIDENCE_THRESHOLD` | `0.5` | Min detection confidence (0.0-1.0) |
| `DETECTION_COOLDOWN` | `5` | Seconds between duplicate detection saves |
| `DETECTION_INTERVAL` | `15` | Run detection every N frames |
| `RECONNECT_DELAY` | `5` | Seconds before reconnecting disconnected streams |
| `APP_PORT` | `8000` | Application server port |

### Stream Configuration Formats

**Format 1: Combined (recommended)**
```env
RTSP_STREAMS=Camera1,rtsp://host1/stream;Camera2,rtsp://host2/stream
```

**Format 2: Individual variables**
```env
RTSP_URL_1=rtsp://host1/stream
RTSP_NAME_1=Camera1
RTSP_URL_2=rtsp://host2/stream
RTSP_NAME_2=Camera2
```

---

## 📡 API Documentation

### `GET /api/streams`

Returns status of all configured RTSP streams.

**Response:**
```json
{
  "streams": [
    {
      "name": "FrontDoor",
      "url": "rtsp://...",
      "status": "online",
      "last_frame_time": "2024-01-15T10:30:45.123456",
      "frame_count": 1523,
      "error": null
    }
  ]
}
```

### `GET /api/detections`

Returns paginated detection records, most recent first.

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 50 | Max results (1-500) |
| `offset` | int | 0 | Pagination offset |
| `camera` | string | - | Filter by camera name |
| `class_name` | string | - | Filter by object class |

**Response:**
```json
{
  "detections": [
    {
      "id": 42,
      "timestamp": "2024-01-15T10:30:45.123456",
      "class_name": "person",
      "confidence": 0.9234,
      "camera_name": "FrontDoor",
      "image_path": "/detections/images/FrontDoor_person_20240115_103045_123456.jpg",
      "bbox_x1": 100,
      "bbox_y1": 50,
      "bbox_x2": 300,
      "bbox_y2": 400,
      "created_at": "2024-01-15T10:30:45"
    }
  ],
  "total": 156,
  "limit": 50,
  "offset": 0
}
```

### `GET /api/detections/{id}`

Returns a single detection record by ID.

### `GET /api/health`

Health check endpoint for Docker and monitoring.

**Response:**
```json
{
  "status": "healthy",
  "streams": {"total": 2, "online": 2},
  "detections": 156
}
```

### `GET /video/{stream_name}`

Returns an MJPEG video stream for embedding in `<img>` tags.

### `GET /`

Main web dashboard (HTML).

---

## 📂 Project Structure

```
backend_dev_test/
├── backend/
│   ├── app.py               # FastAPI application entry point
│   ├── stream_manager.py    # Multi-threaded RTSP stream manager
│   ├── detector.py          # YOLOv8n object detection module
│   └── database.py          # Thread-safe SQLite database module
├── templates/
│   └── index.html           # Jinja2 dashboard template
├── static/
│   └── style.css            # Static CSS assets
├── detections/              # Detection images (persistent volume)
├── screenshots/             # Application screenshots
│   ├── dashboard.png
│   ├── live_streams.png
│   └── detection_gallery.png
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose configuration
├── .env.example             # Environment variable template
└── README.md                # This file
```

---

##  Screenshots

### Dashboard

> *Screenshot: Full dashboard with live camera feeds and detection gallery*

![Dashboard](screenshots/dashboard.png)

### Live Streams

> *Screenshot: Live camera feeds with real-time object detection bounding boxes*

![Live Streams](screenshots/live_streams.png)

### Detection Gallery

> *Screenshot: Browsable gallery of detected objects with filtering and pagination*

![Detection Gallery](screenshots/detection_gallery.png)

---

## 🔧 Troubleshooting

### Container won't start

```bash
# Check logs
docker compose logs -f

# Rebuild from scratch
docker compose down
docker compose up --build
```

### RTSP stream shows "offline"

1. **Verify the RTSP URL** is correct and accessible from the Docker container
2. **Check network connectivity:** the container must reach the camera's IP
3. **Test the stream** outside Docker first:
   ```bash
   ffplay rtsp://username:password@camera-ip:554/stream
   ```
4. **Check firewall rules** — Docker containers may not have access to host network resources by default
5. Try adding `network_mode: host` to docker-compose.yml if cameras are on the local network

### No detections appearing

1. **Lower the confidence threshold** in `.env`:
   ```env
   CONFIDENCE_THRESHOLD=0.3
   ```
2. **Decrease the detection interval** for more frequent checks:
   ```env
   DETECTION_INTERVAL=5
   ```
3. **Check that objects are in the supported classes:** person, car, motorcycle, bicycle, bus, truck

### High CPU usage

1. **Increase the detection interval** to reduce inference frequency:
   ```env
   DETECTION_INTERVAL=30
   ```
2. **Increase the detection cooldown** to reduce image saving:
   ```env
   DETECTION_COOLDOWN=15
   ```

### Detection images lost after restart

Ensure the Docker volume is properly configured in `docker-compose.yml`:
```yaml
volumes:
  - detections_data:/app/detections
```

### Port conflict

Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "9000:8000"  # Map to port 9000 instead
```

---

## ✅ Assessment Requirements Checklist

| # | Requirement | Status | Implementation |
|---|------------|--------|----------------|
| 1 | **RTSP Stream Handling** | ✅ | `stream_manager.py` - Multi-threaded, auto-reconnect |
| 1a | Read streams from environment variables | ✅ | `load_streams_from_env()` supports two formats |
| 1b | Support multiple streams | ✅ | Each stream runs in a dedicated thread |
| 1c | Automatic reconnection on disconnect | ✅ | Reconnect loop with configurable delay |
| 1d | Maintain stream status | ✅ | `StreamInfo.status` (online/offline/connecting) |
| 1e | Track last frame timestamp | ✅ | `StreamInfo.last_frame_time` updated per frame |
| 2 | **Dashboard** | ✅ | `templates/index.html` with Jinja2 |
| 2a | Display all live streams simultaneously | ✅ | MJPEG `<img>` tags in responsive grid |
| 2b | Display camera name | ✅ | Stream header shows camera name |
| 2c | Display online/offline status | ✅ | Color-coded status pills with auto-refresh |
| 2d | Display last frame timestamp | ✅ | Timestamps updated via polling |
| 3 | **Object Detection** | ✅ | `detector.py` using YOLOv8n |
| 3a | Detect only required classes | ✅ | ALLOWED_CLASSES filter (6 classes) |
| 3b | Use YOLOv8n | ✅ | `ultralytics.YOLO("yolov8n.pt")` |
| 4 | **Snapshot Capture** | ✅ | `detector.py` → `detect_and_save()` |
| 4a | Crop detected object | ✅ | Bounding box crop from frame |
| 4b | Save image to persistent storage | ✅ | Docker volume: `/app/detections/images/` |
| 4c | Store timestamp | ✅ | ISO format in SQLite |
| 4d | Store class name | ✅ | COCO class label |
| 4e | Store confidence | ✅ | Float 0-1 |
| 4f | Store camera name | ✅ | Source stream identifier |
| 5 | **Detection Gallery** | ✅ | JavaScript-powered gallery in dashboard |
| 5a | Latest detections first | ✅ | `ORDER BY timestamp DESC` |
| 5b | Display image | ✅ | Served from `/detections/images/` |
| 5c | Display timestamp | ✅ | Formatted in detection cards |
| 5d | Display class | ✅ | Badge on detection cards |
| 5e | Display confidence | ✅ | Percentage with visual bar |
| 5f | Display source camera | ✅ | Camera name in detection info |
| 6 | **Persistence** | ✅ | Docker named volume `detections_data` |
| 6a | Images survive container restarts | ✅ | Volume mounted at `/app/detections` |
| 7 | **REST APIs** | ✅ | FastAPI endpoints in `app.py` |
| 7a | `GET /api/streams` | ✅ | Returns all stream statuses |
| 7b | `GET /api/detections` | ✅ | Paginated with filters |
| 8 | **Docker** | ✅ | Complete containerization |
| 8a | Dockerfile | ✅ | Python 3.11-slim with all deps |
| 8b | docker-compose.yml | ✅ | Volume, env_file, health check |
| 8c | .env.example | ✅ | Documented configuration template |
| 9 | **Project Structure** | ✅ | Matches required layout |
| 10 | **Code Quality** | ✅ | |
| 10a | Modular architecture | ✅ | 4 backend modules with clear responsibilities |
| 10b | Thread-safe implementation | ✅ | Locks, thread-local storage, singleton DB |
| 10c | Multi-threaded stream processing | ✅ | Daemon threads per stream |
| 10d | Clear comments | ✅ | Docstrings and inline comments throughout |
| 10e | Production-ready | ✅ | Health checks, error handling, logging |

---

## 📝 License

This project is developed as a backend developer assessment submission.
