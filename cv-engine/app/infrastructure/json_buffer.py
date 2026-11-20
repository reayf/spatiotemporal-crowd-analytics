"""Adapters over the shared JSON file buffer.

The `data/` directory tree simulates a message broker:

* :class:`JsonJobConsumer` "subscribes" to a queue directory by polling and
  atomically claiming manifest files (rename = broker ack, so two engine
  replicas never process the same job).
* :class:`JsonJobPublisher` "produces" by atomically writing manifests.
* :class:`TrajectoryWriter` persists the pipeline's payload output.

These classes are deliberately dumb IO adapters — no business logic —
so they sit at the outer ring of the Clean Architecture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.domain.entities import Trajectory


class JsonJobConsumer:
    """Polls a queue directory and claims job manifests one at a time."""

    def __init__(self, queue_dir: Path) -> None:
        self.queue_dir = queue_dir
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def claim(self) -> dict[str, Any] | None:
        """Atomically claim the oldest pending manifest, if any.

        The manifest file is renamed to ``*.claimed`` before it is read;
        the rename is the atomic "message ack" that prevents double
        consumption. The claimed file is deleted after a successful parse.

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


class JsonJobPublisher:
    """Publishes job manifests into a downstream queue directory."""

    def __init__(self, queue_dir: Path) -> None:
        self.queue_dir = queue_dir
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def publish(self, job_id: str, payload: dict[str, Any]) -> Path:
        """Atomically write ``payload`` as ``<job_id>.json`` (tmp + rename
        so consumers never observe a half-written manifest)."""
        final_path = self.queue_dir / f"{job_id}.json"
        tmp_path = final_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, final_path)
        return final_path


class TrajectoryWriter:
    """Persists extracted trajectories as the CV engine's output artifact."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        video_id: str,
        trajectories: list[Trajectory],
        metadata: dict[str, Any],
    ) -> Path:
        """Write ``data/trajectories/<video_id>.json``.

        Args:
            video_id: Stable identifier assigned by the gateway.
            trajectories: One entry per tracked identity.
            metadata: Source facts the miner needs (``fps``, ``frame_width``,
                ``frame_height``, ``total_frames``, ...).

        Returns:
            The path of the written log.
        """
        document = {
            "video_id": video_id,
            "metadata": metadata,
            "trajectories": [t.to_dict() for t in trajectories],
        }
        final_path = self.output_dir / f"{video_id}.json"
        tmp_path = final_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        os.replace(tmp_path, final_path)
        return final_path
