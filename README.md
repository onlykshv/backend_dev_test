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

##  Project Structure

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

![Live Streams](screenshots/live_streams_1.png)

![Live Streams](screenshots/live_streams_2.png)

### Detection Gallery

> *Screenshot: Browsable gallery of detected objects with filtering and pagination*

![Detection Gallery](screenshots/live_feed.png)


This project is developed as a backend developer assessment submission.
