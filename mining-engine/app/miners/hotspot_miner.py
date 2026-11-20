"""K-Means hotspot clustering + occupancy time-series regression miner."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

from app.domain.entities import CongestionForecast, Hotspot
from app.miners.base_miner import SpatiotemporalMiner

logger = logging.getLogger(__name__)


class KMeansHotspotMiner(SpatiotemporalMiner):
    """Classical baseline miner.

    Spatial dimension — K-Means over all (x, y) trajectory points to find
    activity hotspots, ranked by their share of total activity.

    Temporal dimension — bins events into fixed windows to build an
    occupancy time-series, then fits a Ridge regressor over lagged
    occupancy and rolls it forward recursively to forecast congestion.
    Videos too short to fit the regressor fall back to a mean-occupancy
    (persistence) forecast instead of failing.

    Args:
        n_clusters: Number of hotspot centroids for K-Means.
        time_bin_seconds: Width of the occupancy time-series bins.
        n_lags: How many past bins condition each occupancy prediction.
        random_state: Seed for reproducible clustering.
    """

    name = "kmeans_hotspot"

    def __init__(
        self,
        n_clusters: int = 5,
        time_bin_seconds: float = 2.0,
        n_lags: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.time_bin_seconds = time_bin_seconds
        self.n_lags = n_lags
        self.random_state = random_state
        self._occupancy_model: Ridge | None = None
        self._history: np.ndarray | None = None
        self._event_ratio: float = 1.0  # observed events per occupancy-unit per bin

    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        """Bin events into a gap-free occupancy time-series.

        See :meth:`SpatiotemporalMiner.transform`.

        Returns:
            DataFrame with one row per ``time_bin_seconds`` bin (empty bins
            included as zeros) and columns ``bin_start``, ``occupancy``
            (distinct simultaneous track ids) and ``event_count``.
        """
        bin_s = self.time_bin_seconds
        t_max = float(events["t"].max())

        bin_index = np.floor(events["t"].to_numpy(dtype=float) / bin_s).astype(int)
        binned = (
            events.assign(bin=bin_index)
            .groupby("bin")
            .agg(occupancy=("track_id", "nunique"), event_count=("t", "size"))
        )
        # Reindex over the full observation span so quiet periods appear as
        # zeros — otherwise the regressor never learns that lulls exist.
        n_bins = int(np.floor(t_max / bin_s)) + 1
        binned = binned.reindex(range(n_bins), fill_value=0)

        return (
            binned.reset_index()
            .assign(bin_start=lambda d: d["bin"] * bin_s)
            [["bin_start", "occupancy", "event_count"]]
        )

    def fit(self, features: pd.DataFrame) -> None:
        """Fit the lagged-occupancy Ridge regressor.

        See :meth:`SpatiotemporalMiner.fit`.
        """
        occupancy = features["occupancy"].to_numpy(dtype=float)
        self._history = occupancy
        self._event_ratio = float(features["event_count"].sum()) / max(
            float(occupancy.sum()), 1.0
        )

        if len(occupancy) <= self.n_lags:
            # Too few bins to build even one lag window; persistence fallback.
            self._occupancy_model = None
            logger.warning(
                "only %d occupancy bins (< %d lags): using mean-occupancy fallback",
                len(occupancy), self.n_lags + 1,
            )
            return

        lagged = np.stack(
            [occupancy[i : i + self.n_lags] for i in range(len(occupancy) - self.n_lags)]
        )
        targets = occupancy[self.n_lags :]
        self._occupancy_model = Ridge(alpha=1.0).fit(lagged, targets)
        logger.info("fitted Ridge on %d lag windows", len(targets))

    def predict(self, horizon_seconds: float) -> list[CongestionForecast]:
        """Extrapolate occupancy recursively, one bin at a time.

        See :meth:`SpatiotemporalMiner.predict`.
        """
        if self._history is None:
            raise RuntimeError("fit() must be called before predict()")

        bin_s = self.time_bin_seconds
        n_windows = max(1, int(np.ceil(horizon_seconds / bin_s)))
        rolling = list(self._history)
        observed_peak = max(float(np.max(self._history)), 1.0)
        fallback_mean = float(np.mean(self._history))

        forecasts: list[CongestionForecast] = []
        for i in range(n_windows):
            if self._occupancy_model is not None:
                window = np.asarray(rolling[-self.n_lags :], dtype=float).reshape(1, -1)
                predicted = float(self._occupancy_model.predict(window)[0])
            else:
                predicted = fallback_mean
            predicted = max(0.0, predicted)
            rolling.append(predicted)

            forecasts.append(
                CongestionForecast(
                    window_start=i * bin_s,
                    window_end=min((i + 1) * bin_s, horizon_seconds),
                    expected_event_count=predicted * self._event_ratio,
                    expected_occupancy=predicted,
                    congestion_level=self._congestion_level(predicted, observed_peak),
                )
            )
        return forecasts

    def extract_hotspots(self, events: pd.DataFrame) -> list[Hotspot]:
        """Cluster (x, y) points with K-Means and rank clusters by activity.

        See :meth:`SpatiotemporalMiner.extract_hotspots`.
        """
        points = events[["x", "y"]].to_numpy(dtype=float)
        n_clusters = min(self.n_clusters, len(points))
        if n_clusters < 1:
            return []

        kmeans = KMeans(
            n_clusters=n_clusters, random_state=self.random_state, n_init=10
        ).fit(points)

        hotspots: list[Hotspot] = []
        for cluster_id, center in enumerate(kmeans.cluster_centers_):
            members = points[kmeans.labels_ == cluster_id]
            distances = np.linalg.norm(members - center, axis=1)
            hotspots.append(
                Hotspot(
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    radius=float(distances.mean()) if len(members) else 0.0,
                    member_count=int(len(members)),
                    intensity=len(members) / len(points),
                )
            )
        return sorted(hotspots, key=lambda h: h.intensity, reverse=True)

    @staticmethod
    def _congestion_level(occupancy: float, observed_peak: float) -> str:
        """Map predicted occupancy to a label relative to the observed peak."""
        ratio = occupancy / observed_peak
        if ratio >= 0.75:
            return "HIGH"
        if ratio >= 0.4:
            return "MEDIUM"
        return "LOW"
