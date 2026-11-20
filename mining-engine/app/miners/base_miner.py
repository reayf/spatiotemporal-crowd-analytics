"""Abstract spatiotemporal mining contract.

Every mining strategy (K-Means hotspot clustering, DeepSTPP, ARIMA, ...)
implements this base class so the service layer can run any of them — or
several in sequence — without knowing the algorithm (Strategy pattern +
Dependency Inversion).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from app.domain.entities import AnalyticsReport, CongestionForecast, Hotspot


class SpatiotemporalMiner(ABC):
    """Base class for spatiotemporal point-process / clustering miners.

    Pipeline contract, driven by the template method :meth:`mine`::

        load_trajectories  raw JSON log      -> tidy DataFrame (track_id, x, y, t)
        transform          tidy DataFrame    -> model-ready features
        fit                features          -> estimated model parameters
        predict            horizon (seconds) -> congestion forecasts
        extract_hotspots   tidy DataFrame    -> spatial hotspots (optional hook)

    Subclasses implement ``transform``/``fit``/``predict`` (and optionally
    override ``extract_hotspots``); the loading, orchestration and report
    assembly are shared here.
    """

    #: Human-readable strategy name written into the analytics report.
    name: str = "abstract"

    # ------------------------------------------------------------------ #
    # Shared, concrete steps
    # ------------------------------------------------------------------ #

    def load_trajectories(self, trajectory_path: Path) -> pd.DataFrame:
        """Parse a CV-engine trajectory log into a tidy event frame.

        Args:
            trajectory_path: Path to ``data/trajectories/<video_id>.json``
                as produced by the CV engine's TrajectoryWriter.

        Returns:
            DataFrame with one row per observation and columns
            ``track_id`` (int), ``x`` (float), ``y`` (float), ``t`` (float,
            seconds), ``frame`` (int), sorted by ``t``.

        Raises:
            FileNotFoundError: If the log does not exist.
            ValueError: If the log contains no trajectory points.
        """
        document = json.loads(trajectory_path.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = [
            {
                "track_id": trajectory["track_id"],
                "x": point["x"],
                "y": point["y"],
                "t": point["t"],
                "frame": point["frame"],
            }
            for trajectory in document.get("trajectories", [])
            for point in trajectory.get("points", [])
        ]
        if not rows:
            raise ValueError(f"no trajectory points in {trajectory_path}")
        return pd.DataFrame(rows).sort_values("t", ignore_index=True)

    def mine(
        self,
        video_id: str,
        trajectory_path: Path,
        horizon_seconds: float = 60.0,
    ) -> AnalyticsReport:
        """Template method: run the full mining pipeline for one video.

        Composes the abstract steps into the standard flow and assembles the
        final :class:`AnalyticsReport`. Subclasses normally do NOT override
        this method.
        """
        events = self.load_trajectories(trajectory_path)
        features = self.transform(events)
        self.fit(features)

        dwell = events.groupby("track_id")["t"].agg(lambda s: s.max() - s.min())
        return AnalyticsReport(
            video_id=video_id,
            miner_name=self.name,
            observed_track_count=int(events["track_id"].nunique()),
            mean_dwell_seconds=float(dwell.mean()),
            hotspots=self.extract_hotspots(events),
            forecasts=self.predict(horizon_seconds),
        )

    # ------------------------------------------------------------------ #
    # Strategy-specific steps
    # ------------------------------------------------------------------ #

    @abstractmethod
    def transform(self, events: pd.DataFrame) -> Any:
        """Turn the tidy event frame into model-ready features.

        Typical outputs: binned occupancy time-series, sliding windows of
        (x, y, dt) event sequences, or normalised coordinate tensors.

        Args:
            events: Output of :meth:`load_trajectories`.

        Returns:
            Whatever representation :meth:`fit` expects (kept ``Any`` on
            purpose — the base class never inspects it).
        """

    @abstractmethod
    def fit(self, features: Any) -> None:
        """Estimate model parameters from the transformed features.

        Must leave the miner in a state where :meth:`predict` can be called.

        Raises:
            ValueError: If the features are insufficient to fit the model
                (e.g. video shorter than one time window).
        """

    @abstractmethod
    def predict(self, horizon_seconds: float) -> list[CongestionForecast]:
        """Forecast crowd dynamics ``horizon_seconds`` past the observed data.

        Args:
            horizon_seconds: Length of the future period to forecast.

        Returns:
            One forecast per prediction window covering the horizon.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """

    def extract_hotspots(self, events: pd.DataFrame) -> list[Hotspot]:
        """Optional hook: derive spatial hotspots from the raw events.

        The default implementation returns no hotspots; clustering miners
        override this.
        """
        return []
