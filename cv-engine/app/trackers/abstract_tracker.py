"""Abstract multi-object tracking contract.

Any MOT backend (YOLOv8+ByteTrack, DeepSORT, ...) must implement this
interface so the service layer depends only on the abstraction
(Dependency Inversion Principle) and backends stay hot-swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from app.domain.entities import Detection, TrackPoint, Trajectory


class AbstractTracker(ABC):
    """Base class for frame-by-frame multi-object trackers.

    Lifecycle contract::

        tracker.load_model()
        trajectories = tracker.process_video(Path("video.mp4"))
        tracker.release()

    ``process_video`` is a template method: it owns the video iteration and
    trajectory assembly, delegating per-frame inference to ``track_frame``,
    which is the only method that touches the underlying CV framework.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load detection weights and initialise tracker state.

        Must be called once before any ``track_frame``/``process_video`` call.

        Raises:
            RuntimeError: If the model weights cannot be loaded on the
                configured device.
        """

    @abstractmethod
    def track_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> list[Detection]:
        """Run detection + identity association on a single BGR frame.

        Args:
            frame: HxWx3 uint8 image in BGR channel order (OpenCV native).
            frame_index: Zero-based index of the frame within the video.
            timestamp: Seconds since the start of the video.

        Returns:
            All confirmed tracked objects visible on this frame. Objects
            without an assigned track id (unconfirmed by ByteTrack) must be
            excluded.
        """

    @abstractmethod
    def release(self) -> None:
        """Free model/GPU resources and reset internal tracker state.

        Must be idempotent: calling it twice is a no-op.
        """

    def on_video_start(self) -> None:
        """Hook invoked before the first frame of every video.

        Backends that keep association state across calls (ByteTrack's
        track buffer) override this to reset identities, so track ids never
        leak between videos. Default is a no-op.
        """

    def process_video(
        self,
        video_path: Path,
        sample_rate: int = 1,
    ) -> list[Trajectory]:
        """Template method: run the full tracking pass over a video file.

        Iterates frames with OpenCV, calls :meth:`track_frame` on every
        ``sample_rate``-th frame, and groups the resulting detections into
        per-identity :class:`Trajectory` objects ordered by timestamp.

        Args:
            video_path: Path to a readable video file (mp4/avi/mkv).
            sample_rate: Process every N-th frame (1 = every frame). Higher
                values trade temporal resolution for throughput.

        Returns:
            One trajectory per unique track id, points sorted by time.

        Raises:
            FileNotFoundError: If ``video_path`` does not exist.
            ValueError: If ``sample_rate`` < 1 or the video cannot be decoded.
        """
        if sample_rate < 1:
            raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot decode video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0  # some containers report 0; assume a sane default

        self.on_video_start()
        trajectories: dict[int, Trajectory] = {}
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % sample_rate == 0:
                    timestamp = frame_index / fps
                    for detection in self.track_frame(frame, frame_index, timestamp):
                        x, y = detection.centroid
                        trajectory = trajectories.get(detection.track_id)
                        if trajectory is None:
                            trajectory = Trajectory(
                                track_id=detection.track_id,
                                class_name=detection.class_name,
                            )
                            trajectories[detection.track_id] = trajectory
                        trajectory.points.append(
                            TrackPoint(
                                track_id=detection.track_id,
                                x=x,
                                y=y,
                                timestamp=timestamp,
                                frame_index=frame_index,
                            )
                        )
                frame_index += 1
        finally:
            capture.release()

        # track_frame is called in frame order, so points are already sorted
        # by time; sort trajectories by id for deterministic output.
        return [trajectories[track_id] for track_id in sorted(trajectories)]
