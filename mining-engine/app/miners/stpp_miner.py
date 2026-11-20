"""Simplified Deep Spatiotemporal Point Process (DeepSTPP) miner.

Models the trajectory log as a marked point process: each observation is an
event (x, y, t). A GRU encodes the recent event history into a latent state
from which two heads are decoded:

* a **temporal head** emitting the conditional intensity ``lambda`` of the
  next event (exponential waiting-time likelihood), and
* a **spatial head** emitting a diagonal Gaussian over the next event's
  displacement (dx, dy).

Training maximises the joint log-likelihood; forecasting rolls the model
forward event-by-event, accumulating expected event counts per window.

Reference: Zhou et al., "Neural Point Process for Learning Spatiotemporal
Event Dynamics" (DeepSTPP), simplified to a fixed-length history encoder
and closed-form exponential/Gaussian likelihoods for portfolio scope.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from app.domain.entities import CongestionForecast
from app.miners.base_miner import SpatiotemporalMiner

logger = logging.getLogger(__name__)

#: Safety cap on simulated events during forecasting, so a degenerate model
#: predicting near-zero waiting times cannot loop forever.
_MAX_SIMULATED_EVENTS = 5000


class _IntensityNetwork(nn.Module):
    """Encodes a fixed-length event history and emits intensity parameters.

    Input: (batch, history_len, 3) tensors of z-normalised (dx, dy, dt)
    event deltas. Output: the positive temporal intensity ``lambda`` and the
    spatial Gaussian parameters (mu_x, mu_y, log_sigma_x, log_sigma_y).
    """

    def __init__(self, hidden_dim: int = 64, history_len: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.embed = nn.Linear(3, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.spatial_head = nn.Linear(hidden_dim, 4)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map (batch, history_len, 3) delta sequences to intensity params.

        Returns:
            ``(rate, spatial)`` where ``rate`` is (batch,) strictly positive
            event intensities and ``spatial`` is (batch, 4) Gaussian params
            (mu_x, mu_y, log_sigma_x, log_sigma_y) in normalised delta space.
        """
        embedded = torch.relu(self.embed(history))
        _, final_state = self.encoder(embedded)
        latent = final_state[-1]  # (batch, hidden_dim)
        rate = nn.functional.softplus(self.temporal_head(latent)).squeeze(-1) + 1e-6
        spatial = self.spatial_head(latent)
        return rate, spatial


class DeepSTPPMiner(SpatiotemporalMiner):
    """Neural point-process miner over the (x, y, t) event stream.

    Args:
        history_len: Number of past events conditioning each prediction.
            Automatically shrunk for short videos (see :meth:`transform`).
        hidden_dim: Width of the history encoder.
        epochs: Training epochs per video (the model is fit per-video).
        learning_rate: Adam step size.
        device: Torch device string.
        seed: Manual torch seed so per-video fits are reproducible.
    """

    name = "deep_stpp"

    def __init__(
        self,
        history_len: int = 32,
        hidden_dim: int = 64,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.seed = seed

        self._network: _IntensityNetwork | None = None
        self._effective_history: int = history_len
        self._delta_mean: np.ndarray | None = None
        self._delta_std: np.ndarray | None = None
        self._last_history: torch.Tensor | None = None
        self._observed_event_rate: float = 0.0   # events / second
        self._observed_mean_occupancy: float = 0.0

    def transform(
        self, events: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build normalised sliding windows of (dx, dy, dt) event deltas.

        The history length shrinks automatically when the video yields fewer
        deltas than ``history_len`` requires, so short clips still train.

        See :meth:`SpatiotemporalMiner.transform`.

        Returns:
            ``(histories, dt_targets, xy_targets)`` where ``histories`` is
            (n_windows, effective_history, 3) normalised deltas,
            ``dt_targets`` is (n_windows,) raw waiting times in seconds and
            ``xy_targets`` is (n_windows, 2) normalised displacements.

        Raises:
            ValueError: If the log yields fewer than 4 event deltas.
        """
        ordered = events.sort_values("t", ignore_index=True)
        coords = ordered[["x", "y", "t"]].to_numpy(dtype=np.float32)
        deltas = np.diff(coords, axis=0)  # (n-1, 3): dx, dy, dt

        if len(deltas) < 4:
            raise ValueError(
                f"only {len(deltas)} event deltas — too few to fit a point process"
            )
        self._effective_history = min(self.history_len, len(deltas) - 1)

        self._delta_mean = deltas.mean(axis=0)
        self._delta_std = np.maximum(deltas.std(axis=0), 1e-6)
        normalised = (deltas - self._delta_mean) / self._delta_std

        h = self._effective_history
        histories = np.stack([normalised[i : i + h] for i in range(len(deltas) - h)])
        dt_targets = deltas[h:, 2]          # raw seconds (exponential support)
        xy_targets = normalised[h:, :2]     # normalised space (Gaussian)

        # Kept for forecasting: the model rolls forward from the latest state.
        self._last_history = torch.from_numpy(normalised[-h:]).unsqueeze(0).to(self.device)

        duration = max(float(ordered["t"].iloc[-1] - ordered["t"].iloc[0]), 1e-6)
        self._observed_event_rate = len(deltas) / duration
        self._observed_mean_occupancy = float(
            ordered.groupby(np.floor(ordered["t"]))["track_id"].nunique().mean()
        )

        return (
            torch.from_numpy(histories).to(self.device),
            torch.from_numpy(dt_targets.copy()).to(self.device),
            torch.from_numpy(xy_targets.copy()).to(self.device),
        )

    def fit(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        """Train the intensity network by minimising the joint negative
        log-likelihood (exponential temporal + diagonal Gaussian spatial).

        See :meth:`SpatiotemporalMiner.fit`.
        """
        histories, dt_targets, xy_targets = features
        torch.manual_seed(self.seed)

        network = _IntensityNetwork(
            hidden_dim=self.hidden_dim, history_len=self._effective_history
        ).to(self.device)
        optimiser = torch.optim.Adam(network.parameters(), lr=self.learning_rate)

        network.train()
        for epoch in range(self.epochs):
            optimiser.zero_grad()
            rate, spatial = network(histories)

            # Exponential waiting-time NLL: -log lambda + lambda * dt
            temporal_nll = (-torch.log(rate) + rate * dt_targets).mean()

            # Diagonal Gaussian NLL over normalised (dx, dy)
            mu = spatial[:, :2]
            log_sigma = spatial[:, 2:].clamp(-5.0, 5.0)
            spatial_nll = (
                0.5 * (((xy_targets - mu) / torch.exp(log_sigma)) ** 2)
                + log_sigma
            ).sum(dim=-1).mean() + math.log(2 * math.pi)

            loss = temporal_nll + spatial_nll
            loss.backward()
            optimiser.step()

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.debug(
                    "epoch %d/%d  nll=%.4f (temporal=%.4f spatial=%.4f)",
                    epoch + 1, self.epochs, loss.item(),
                    temporal_nll.item(), spatial_nll.item(),
                )

        self._network = network
        logger.info(
            "trained DeepSTPP on %d windows (history=%d) for %d epochs",
            len(histories), self._effective_history, self.epochs,
        )

    def predict(self, horizon_seconds: float) -> list[CongestionForecast]:
        """Roll the learned process forward, accumulating expected event
        counts per window; occupancy is estimated by scaling the observed
        occupancy/event-rate ratio.

        See :meth:`SpatiotemporalMiner.predict`.
        """
        if self._network is None or self._last_history is None:
            raise RuntimeError("fit() must be called before predict()")
        assert self._delta_mean is not None and self._delta_std is not None

        n_windows = max(1, min(12, int(np.ceil(horizon_seconds / 5.0))))
        window_s = horizon_seconds / n_windows
        window_counts = [0.0] * n_windows

        self._network.eval()
        history = self._last_history.clone()
        elapsed = 0.0
        with torch.no_grad():
            for _ in range(_MAX_SIMULATED_EVENTS):
                rate, spatial = self._network(history)
                # Expected waiting time of an exponential arrival: 1/lambda.
                dt = float(1.0 / rate.item())
                dt = min(max(dt, 1e-3), horizon_seconds)
                elapsed += dt
                if elapsed >= horizon_seconds:
                    break
                window_counts[min(int(elapsed // window_s), n_windows - 1)] += 1.0

                # Feed the predicted event back as the newest history entry.
                mu = spatial[0, :2].cpu().numpy()
                raw_delta = np.array(
                    [
                        mu[0] * self._delta_std[0] + self._delta_mean[0],
                        mu[1] * self._delta_std[1] + self._delta_mean[1],
                        dt,
                    ],
                    dtype=np.float32,
                )
                normalised = (raw_delta - self._delta_mean) / self._delta_std
                next_entry = torch.from_numpy(normalised).view(1, 1, 3).to(self.device)
                history = torch.cat([history[:, 1:], next_entry], dim=1)

        observed_rate = max(self._observed_event_rate, 1e-6)
        occupancy_per_rate = self._observed_mean_occupancy / observed_rate

        forecasts: list[CongestionForecast] = []
        for i, count in enumerate(window_counts):
            predicted_rate = count / window_s
            forecasts.append(
                CongestionForecast(
                    window_start=i * window_s,
                    window_end=(i + 1) * window_s,
                    expected_event_count=count,
                    expected_occupancy=predicted_rate * occupancy_per_rate,
                    congestion_level=self._congestion_level(predicted_rate, observed_rate),
                )
            )
        return forecasts

    @staticmethod
    def _congestion_level(predicted_rate: float, observed_rate: float) -> str:
        """Label a window by its intensity relative to the observed rate."""
        ratio = predicted_rate / observed_rate
        if ratio >= 1.2:
            return "HIGH"
        if ratio >= 0.6:
            return "MEDIUM"
        return "LOW"
