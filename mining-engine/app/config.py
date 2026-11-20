"""Environment-driven settings for the mining engine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration.

    Attributes:
        data_dir: Root of the shared JSON file buffer (must match the
            gateway's DATA_DIR).
        poll_interval_seconds: Queue polling cadence when idle.
        forecast_horizon_seconds: How far ahead ``predict`` should forecast.
        miner: Mining strategy to use ("kmeans" or "deepstpp").
    """

    data_dir: Path
    poll_interval_seconds: float
    forecast_horizon_seconds: float
    miner: str

    @property
    def mining_queue_dir(self) -> Path:
        return self.data_dir / "queue" / "mining"

    @property
    def trajectories_dir(self) -> Path:
        return self.data_dir / "trajectories"

    @property
    def analytics_dir(self) -> Path:
        return self.data_dir / "analytics"

    @staticmethod
    def from_env() -> "Settings":
        """Build settings from environment variables with dev defaults.

        The default data_dir assumes the service is started from the
        mining-engine/ directory (see README).
        """
        return Settings(
            data_dir=Path(os.getenv("DATA_DIR", "../data")).resolve(),
            poll_interval_seconds=float(os.getenv("POLL_INTERVAL_SECONDS", "2.0")),
            forecast_horizon_seconds=float(os.getenv("FORECAST_HORIZON_SECONDS", "60.0")),
            miner=os.getenv("MINER", "kmeans").lower(),
        )
