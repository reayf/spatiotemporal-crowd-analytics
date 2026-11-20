"""Environment-driven settings for the CV engine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration.

    Attributes:
        data_dir: Root of the shared JSON file buffer (must match the
            gateway's DATA_DIR so both services see the same directories).
        poll_interval_seconds: How often the job consumer polls the queue
            directory when it is empty.
        model_path: YOLOv8 weights to load (auto-downloaded by ultralytics).
        device: Inference device ("cpu", "cuda:0", ...).
        sample_rate: Process every N-th frame (1 = every frame).
    """

    data_dir: Path
    poll_interval_seconds: float
    model_path: str
    device: str
    sample_rate: int

    @property
    def tracking_queue_dir(self) -> Path:
        return self.data_dir / "queue" / "tracking"

    @property
    def mining_queue_dir(self) -> Path:
        return self.data_dir / "queue" / "mining"

    @property
    def trajectories_dir(self) -> Path:
        return self.data_dir / "trajectories"

    @staticmethod
    def from_env() -> "Settings":
        """Build settings from environment variables with dev defaults.

        The default data_dir assumes the service is started from the
        cv-engine/ directory (see README).
        """
        return Settings(
            data_dir=Path(os.getenv("DATA_DIR", "../data")).resolve(),
            poll_interval_seconds=float(os.getenv("POLL_INTERVAL_SECONDS", "2.0")),
            model_path=os.getenv("YOLO_MODEL", "yolov8n.pt"),
            device=os.getenv("DEVICE", "cpu"),
            sample_rate=int(os.getenv("SAMPLE_RATE", "1")),
        )
