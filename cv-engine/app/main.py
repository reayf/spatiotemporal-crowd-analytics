"""CV engine entrypoint: composition root + polling worker loop.

Run from the cv-engine/ directory::

    python -m app.main
"""

from __future__ import annotations

import logging
import time

from app.config import Settings
from app.infrastructure.json_buffer import (
    JsonJobConsumer,
    JsonJobPublisher,
    TrajectoryWriter,
)
from app.services.tracking_service import TrackingService
from app.trackers.yolo_bytetrack import YoloByteTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [cv-engine] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def build_service(settings: Settings) -> TrackingService:
    """Composition root: wire concrete adapters into the use case.

    This is the single place that knows about YoloByteTracker; everything
    downstream depends on the AbstractTracker interface.
    """
    tracker = YoloByteTracker(model_path=settings.model_path, device=settings.device)
    return TrackingService(
        tracker=tracker,
        writer=TrajectoryWriter(settings.trajectories_dir),
        mining_publisher=JsonJobPublisher(settings.mining_queue_dir),
        sample_rate=settings.sample_rate,
    )


def run() -> None:
    """Poll the tracking queue forever, processing one job at a time."""
    settings = Settings.from_env()
    consumer = JsonJobConsumer(settings.tracking_queue_dir)
    service = build_service(settings)

    logger.info("polling %s every %.1fs", settings.tracking_queue_dir, settings.poll_interval_seconds)
    while True:
        manifest = consumer.claim()
        if manifest is None:
            time.sleep(settings.poll_interval_seconds)
            continue
        job_id = manifest.get("job_id", "<unknown>")
        logger.info("claimed job %s", job_id)
        try:
            service.handle_job(manifest)
            logger.info("job %s completed", job_id)
        except Exception:
            logger.exception("job %s failed", job_id)


if __name__ == "__main__":
    run()
