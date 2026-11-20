"""Analytics entities produced by the spatiotemporal miners.

Pure dataclasses — no pandas/torch/sklearn imports — so the domain layer
stays at the centre of the Clean Architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Hotspot:
    """A spatial cluster of activity (e.g. a K-Means centroid over all
    trajectory points, weighted by dwell).

    Attributes:
        center_x / center_y: Cluster centre in pixel coordinates.
        radius: Dispersion of member points (e.g. mean distance to centre).
        member_count: Number of trajectory points assigned to the cluster.
        intensity: Normalised activity share in [0, 1] across all hotspots.
    """

    center_x: float
    center_y: float
    radius: float
    member_count: int
    intensity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": [round(self.center_x, 2), round(self.center_y, 2)],
            "radius": round(self.radius, 2),
            "member_count": self.member_count,
            "intensity": round(self.intensity, 4),
        }


@dataclass(frozen=True)
class CongestionForecast:
    """Predicted crowd dynamics over a future time window.

    Attributes:
        window_start / window_end: Forecast window in seconds relative to
            the end of the observed video.
        expected_event_count: Predicted number of point-process events
            (object appearances) in the window.
        expected_occupancy: Predicted simultaneous object count.
        congestion_level: Qualitative label ("LOW" | "MEDIUM" | "HIGH").
    """

    window_start: float
    window_end: float
    expected_event_count: float
    expected_occupancy: float
    congestion_level: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": [round(self.window_start, 2), round(self.window_end, 2)],
            "expected_event_count": round(self.expected_event_count, 3),
            "expected_occupancy": round(self.expected_occupancy, 3),
            "congestion_level": self.congestion_level,
        }


@dataclass
class AnalyticsReport:
    """Aggregate output of one mining run — the payload the Go gateway
    serves at GET /api/v1/analytics/:video_id."""

    video_id: str
    miner_name: str
    observed_track_count: int
    mean_dwell_seconds: float
    hotspots: list[Hotspot] = field(default_factory=list)
    forecasts: list[CongestionForecast] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "miner": self.miner_name,
            "observed_track_count": self.observed_track_count,
            "mean_dwell_seconds": round(self.mean_dwell_seconds, 3),
            "hotspots": [h.to_dict() for h in self.hotspots],
            "forecasts": [f.to_dict() for f in self.forecasts],
        }
