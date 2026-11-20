"""Core entities produced by the tracking pipeline.

These are pure dataclasses with no dependency on ultralytics/OpenCV, so the
domain layer stays testable and the tracking backend stays swappable
(Clean Architecture: entities at the centre, frameworks at the edge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    """A single tracked object observation on one frame.

    Attributes:
        frame_index: Zero-based index of the source frame.
        timestamp: Seconds since the start of the video (frame_index / fps).
        track_id: Stable identity assigned by the MOT algorithm (ByteTrack).
        class_id: COCO class index (0 == person).
        class_name: Human-readable class label.
        confidence: Detector confidence in [0, 1].
        bbox_xyxy: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
    """

    frame_index: int
    timestamp: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def centroid(self) -> tuple[float, float]:
        """Bounding-box centre (x, y) — the point fed to the miner."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class TrackPoint:
    """One (x, y, t) sample of a trajectory — the atomic unit the
    spatiotemporal miner consumes."""

    track_id: int
    x: float
    y: float
    timestamp: float
    frame_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "t": round(self.timestamp, 4),
            "frame": self.frame_index,
        }


@dataclass
class Trajectory:
    """The full movement path of a single tracked identity."""

    track_id: int
    class_name: str
    points: list[TrackPoint] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Dwell time of the identity inside the scene."""
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].timestamp - self.points[0].timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "duration_seconds": round(self.duration_seconds, 4),
            "points": [p.to_dict() for p in self.points],
        }
