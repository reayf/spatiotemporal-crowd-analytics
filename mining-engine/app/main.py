"""Mining engine entrypoint: composition root + polling worker loop.

Run from the mining-engine/ directory::

    python -m app.main
"""

from __future__ import annotations

import logging
import time

from app.config import Settings
from app.infrastructure.json_buffer import JsonJobConsumer, ReportWriter
from app.miners.base_miner import SpatiotemporalMiner
from app.miners.hotspot_miner import KMeansHotspotMiner
from app.services.mining_service import MiningService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [mining-engine] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def build_miner(name: str) -> SpatiotemporalMiner:
    """Resolve the mining strategy selected via the MINER env var.

    DeepSTPPMiner is imported lazily so the default K-Means configuration
    does not require torch to be installed.
    """
    if name == "deepstpp":
        from app.miners.stpp_miner import DeepSTPPMiner

        return DeepSTPPMiner()
    if name == "kmeans":
        return KMeansHotspotMiner()
    raise ValueError(f"unknown miner {name!r}; expected 'kmeans' or 'deepstpp'")


def build_service(settings: Settings) -> MiningService:
    """Composition root: wire the chosen mining strategy into the use case."""
    return MiningService(
        miner=build_miner(settings.miner),
        report_writer=ReportWriter(settings.analytics_dir),
        forecast_horizon_seconds=settings.forecast_horizon_seconds,
    )


def run() -> None:
    """Poll the mining queue forever, processing one job at a time."""
    settings = Settings.from_env()
    consumer = JsonJobConsumer(settings.mining_queue_dir)
    service = build_service(settings)

    logger.info("polling %s every %.1fs", settings.mining_queue_dir, settings.poll_interval_seconds)
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
