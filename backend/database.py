"""
database.py - SQLite database module for persisting object detections.

Provides thread-safe database operations using SQLite WAL mode.
Stores detection metadata including timestamp, class, confidence,
camera source, and image path.
"""

import sqlite3
import threading
import os
from datetime import datetime
from typing import List, Dict, Optional


# Project root (one level up from backend/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database file path - stored in persistent volume
DB_PATH = os.environ.get("DB_PATH", os.path.join(PROJECT_ROOT, "detections", "detections.db"))


class DetectionDB:
    """Thread-safe SQLite database wrapper for detection records."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure single DB connection pool."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent read/write performance
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.connection = conn
        return self._local.connection

    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                camera_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_timestamp
            ON detections(timestamp DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_camera
            ON detections(camera_name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_class
            ON detections(class_name)
        """)
        conn.commit()

    def add_detection(
        self,
        timestamp: str,
        class_name: str,
        confidence: float,
        camera_name: str,
        image_path: str,
        bbox: Optional[tuple] = None,
    ) -> int:
        """
        Insert a new detection record into the database.

        Args:
            timestamp: ISO format timestamp of detection
            class_name: Detected object class (person, car, etc.)
            confidence: Detection confidence score (0-1)
            camera_name: Source camera identifier
            image_path: Path to the cropped detection image
            bbox: Optional bounding box coordinates (x1, y1, x2, y2)

        Returns:
            The ID of the inserted record
        """
        conn = self._get_connection()
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox if bbox else (None, None, None, None)

        cursor = conn.execute(
            """
            INSERT INTO detections
                (timestamp, class_name, confidence, camera_name, image_path,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                class_name,
                confidence,
                camera_name,
                image_path,
                bbox_x1,
                bbox_y1,
                bbox_x2,
                bbox_y2,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_detections(
        self,
        limit: int = 50,
        offset: int = 0,
        camera: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve detections, ordered by most recent first.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip (for pagination)
            camera: Optional filter by camera name
            class_name: Optional filter by object class

        Returns:
            List of detection records as dictionaries
        """
        conn = self._get_connection()
        query = "SELECT * FROM detections WHERE 1=1"
        params = []

        if camera:
            query += " AND camera_name = ?"
            params.append(camera)

        if class_name:
            query += " AND class_name = ?"
            params.append(class_name)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_detection_count(
        self,
        camera: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> int:
        """Get total count of detections with optional filters."""
        conn = self._get_connection()
        query = "SELECT COUNT(*) FROM detections WHERE 1=1"
        params = []

        if camera:
            query += " AND camera_name = ?"
            params.append(camera)

        if class_name:
            query += " AND class_name = ?"
            params.append(class_name)

        cursor = conn.execute(query, params)
        return cursor.fetchone()[0]

    def get_detection_by_id(self, detection_id: int) -> Optional[Dict]:
        """Retrieve a single detection by its ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM detections WHERE id = ?", (detection_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
