"""Adapters over the shared JSON file buffer for the mining engine.

Intentionally mirrors the CV engine's adapters rather than sharing a
package: each microservice owns its infrastructure code and could be
deployed/versioned independently.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.domain.entities import AnalyticsReport


class JsonJobConsumer:
    """Polls the mining queue directory and claims job manifests."""

    def __init__(self, queue_dir: Path) -> None:
        self.queue_dir = queue_dir
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def claim(self) -> dict[str, Any] | None:
        """Atomically claim the oldest pending manifest, if any.

        Rename-before-read acts as the broker "ack", so concurrent replicas
        never process the same job.

        Returns:
            The parsed manifest dict, or ``None`` when the queue is empty.
        """
        pending = sorted(
            self.queue_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        for manifest_path in pending:
            claimed_path = manifest_path.with_suffix(".claimed")
            try:
                os.rename(manifest_path, claimed_path)
            except OSError:
                continue  # another worker claimed it first
            try:
                manifest: dict[str, Any] = json.loads(
                    claimed_path.read_text(encoding="utf-8")
                )
            finally:
                claimed_path.unlink(missing_ok=True)
            return manifest
        return None


class ReportWriter:
    """Persists analytics reports where the Go gateway serves them from."""

    def __init__(self, analytics_dir: Path) -> None:
        self.analytics_dir = analytics_dir
        self.analytics_dir.mkdir(parents=True, exist_ok=True)

    def write(self, report: AnalyticsReport) -> Path:
        """Atomically write ``data/analytics/<video_id>.json``.

        The atomic tmp+rename matters here: the gateway may read this file
        at any moment, and must never observe a partial report.
        """
        final_path = self.analytics_dir / f"{report.video_id}.json"
        tmp_path = final_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp_path, final_path)
        return final_path
