"""Use case: process one tracking job end-to-end."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.infrastructure.json_buffer import JsonJobPublisher, TrajectoryWriter
from app.infrastructure.video_probe import probe_video
from app.trackers.abstract_tracker import AbstractTracker

logger = logging.getLogger(__name__)

_REQUIRED_MANIFEST_FIELDS = ("job_id", "video_id", "video_path")


class TrackingService:
    """Orchestrates a single tracking job:

    1. Resolve the video path from the gateway's job manifest.
    2. Run the injected :class:`AbstractTracker` over the video.
    3. Persist trajectories via :class:`TrajectoryWriter`.
    4. Publish a follow-up mining job so the analytics engine picks it up.

    The service depends only on the tracker *abstraction* — swapping
    YOLOv8+ByteTrack for another backend requires zero changes here.
    """

    def __init__(
        self,
        tracker: AbstractTracker,
        writer: TrajectoryWriter,
        mining_publisher: JsonJobPublisher,
        sample_rate: int = 1,
    ) -> None:
        self.tracker = tracker
        self.writer = writer
        self.mining_publisher = mining_publisher
        self.sample_rate = sample_rate
        self._model_ready = False

    def handle_job(self, manifest: dict[str, Any]) -> None:
        """Execute the tracking pipeline for one gateway job manifest.

        Args:
            manifest: Parsed job JSON from the gateway; must contain
                ``job_id``, ``video_id`` and ``video_path``.

        Raises:
            KeyError: If the manifest is missing required fields.
            FileNotFoundError: If the referenced video does not exist.
            ValueError: If the video cannot be decoded.
        """
        missing = [f for f in _REQUIRED_MANIFEST_FIELDS if not manifest.get(f)]
        if missing:
            raise KeyError(f"manifest missing required fields: {missing}")

        job_id: str = manifest["job_id"]
        video_id: str = manifest["video_id"]
        video_path = Path(manifest["video_path"])
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")

        # Weights load once per process, not per job.
        if not self._model_ready:
            self.tracker.load_model()
            self._model_ready = True

        metadata = probe_video(video_path)
        logger.info(
            "job %s: tracking %s (%d frames @ %.2f fps)",
            job_id, video_path.name, metadata["total_frames"], metadata["fps"],
        )

        trajectories = self.tracker.process_video(video_path, sample_rate=self.sample_rate)
        logger.info("job %s: extracted %d trajectories", job_id, len(trajectories))

        metadata["sample_rate"] = self.sample_rate
        log_path = self.writer.write(video_id, trajectories, metadata)

        # Hand off to the mining engine — the next hop of the pipeline.
        self.mining_publisher.publish(
            job_id,
            {
                "job_id": job_id,
                "video_id": video_id,
                "trajectory_path": str(log_path),
            },
        )
        logger.info("job %s: published mining job for video %s", job_id, video_id)
