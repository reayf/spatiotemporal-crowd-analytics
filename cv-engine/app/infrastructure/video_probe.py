"""Video metadata probing, kept in infrastructure so the service layer
never imports OpenCV directly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


def probe_video(video_path: Path) -> dict[str, Any]:
    """Read container-level facts the miner needs to interpret trajectories.

    Args:
        video_path: Path to a readable video file.

    Returns:
        Dict with ``fps``, ``frame_width``, ``frame_height`` and
        ``total_frames``. ``fps`` falls back to 30.0 when the container
        reports 0 (matching AbstractTracker.process_video's assumption).

    Raises:
        ValueError: If the video cannot be opened.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        return {
            "fps": float(fps) if fps and fps > 0 else 30.0,
            "frame_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "frame_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        capture.release()
