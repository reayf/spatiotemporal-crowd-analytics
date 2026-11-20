"""Use case: process one mining job end-to-end."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.infrastructure.json_buffer import ReportWriter
from app.miners.base_miner import SpatiotemporalMiner

logger = logging.getLogger(__name__)

_REQUIRED_MANIFEST_FIELDS = ("job_id", "video_id", "trajectory_path")


class MiningService:
    """Orchestrates a single mining job:

    1. Resolve the trajectory log path from the CV engine's job manifest.
    2. Run the injected :class:`SpatiotemporalMiner` (Strategy) over it.
    3. Persist the :class:`AnalyticsReport` where the gateway serves it.

    Depends only on the miner abstraction — swapping K-Means for DeepSTPP
    is a one-line change in the composition root (main.py).
    """

    def __init__(
        self,
        miner: SpatiotemporalMiner,
        report_writer: ReportWriter,
        forecast_horizon_seconds: float,
    ) -> None:
        self.miner = miner
        self.report_writer = report_writer
        self.forecast_horizon_seconds = forecast_horizon_seconds

    def handle_job(self, manifest: dict[str, Any]) -> None:
        """Execute the mining pipeline for one job manifest.

        Args:
            manifest: Parsed job JSON from the CV engine; must contain
                ``job_id``, ``video_id`` and ``trajectory_path``.

        Raises:
            KeyError: If the manifest is missing required fields.
            FileNotFoundError: If the trajectory log does not exist.
            ValueError: If the log contains too little data to mine.
        """
        missing = [f for f in _REQUIRED_MANIFEST_FIELDS if not manifest.get(f)]
        if missing:
            raise KeyError(f"manifest missing required fields: {missing}")

        job_id: str = manifest["job_id"]
        video_id: str = manifest["video_id"]
        trajectory_path = Path(manifest["trajectory_path"])
        if not trajectory_path.exists():
            raise FileNotFoundError(f"trajectory log not found: {trajectory_path}")

        logger.info("job %s: mining %s with %s", job_id, trajectory_path.name, self.miner.name)
        report = self.miner.mine(
            video_id=video_id,
            trajectory_path=trajectory_path,
            horizon_seconds=self.forecast_horizon_seconds,
        )
        report_path = self.report_writer.write(report)
        logger.info(
            "job %s: report written to %s (%d tracks, %d hotspots, %d forecasts)",
            job_id, report_path, report.observed_track_count,
            len(report.hotspots), len(report.forecasts),
        )
