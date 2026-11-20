"""YOLOv8 + ByteTrack implementation of the AbstractTracker contract."""

from __future__ import annotations

import logging

import numpy as np

from app.domain.entities import Detection
from app.trackers.abstract_tracker import AbstractTracker

logger = logging.getLogger(__name__)


class YoloByteTracker(AbstractTracker):
    """Multi-object tracker backed by Ultralytics YOLOv8 detection with the
    native ByteTrack association (``model.track(..., tracker="bytetrack.yaml",
    persist=True)``).

    Suitable for pedestrian tracking on MOT17/MOT20 and Oxford Town Centre
    style CCTV footage.

    Args:
        model_path: YOLOv8 weights file or model name (auto-downloaded by
            ultralytics, e.g. ``"yolov8n.pt"``).
        tracker_config: Ultralytics tracker YAML (``"bytetrack.yaml"``).
        confidence_threshold: Minimum detector confidence to keep a box.
        device: Inference device (``"cpu"``, ``"cuda:0"``, ...).
        target_classes: COCO class ids to track; ``(0,)`` == persons only.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.3,
        device: str = "cpu",
        target_classes: tuple[int, ...] = (0,),
    ) -> None:
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.target_classes = target_classes
        self._model = None  # ultralytics.YOLO, loaded lazily

    def load_model(self) -> None:
        """Load YOLOv8 weights onto the configured device and warm up.

        See :meth:`AbstractTracker.load_model`.
        """
        # Imported here, not at module top, so the abstraction layer can be
        # imported (and unit-tested) without ultralytics/torch installed.
        from ultralytics import YOLO

        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            # Warm-up inference so the first real frame doesn't absorb the
            # graph/kernel initialisation latency.
            model.predict(
                np.zeros((640, 640, 3), dtype=np.uint8),
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # ultralytics raises assorted exception types
            raise RuntimeError(
                f"failed to load YOLO model {self.model_path!r} on {self.device!r}: {exc}"
            ) from exc

        self._model = model
        logger.info("loaded %s on %s", self.model_path, self.device)

    def track_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> list[Detection]:
        """Detect + associate objects on one frame via ``model.track``.

        See :meth:`AbstractTracker.track_frame`.

        Raises:
            RuntimeError: If called before :meth:`load_model`.
        """
        if self._model is None:
            raise RuntimeError("load_model() must be called before track_frame()")

        results = self._model.track(
            frame,
            persist=True,  # keep ByteTrack state across successive frames
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            classes=list(self.target_classes),
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        # boxes.id is None while ByteTrack has no confirmed tracks yet;
        # unconfirmed detections are excluded per the interface contract.
        if boxes is None or boxes.id is None:
            return []

        names = results[0].names
        detections: list[Detection] = []
        for track_id, box, confidence, class_id in zip(
            boxes.id.int().tolist(),
            boxes.xyxy.tolist(),
            boxes.conf.tolist(),
            boxes.cls.int().tolist(),
        ):
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                Detection(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    track_id=int(track_id),
                    class_id=int(class_id),
                    class_name=str(names.get(class_id, str(class_id))),
                    confidence=float(confidence),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return detections

    def on_video_start(self) -> None:
        """Reset ByteTrack's track buffer so identities never leak across
        videos processed by the same tracker instance.

        See :meth:`AbstractTracker.on_video_start`.
        """
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def release(self) -> None:
        """Drop the model reference and free CUDA memory if applicable.

        See :meth:`AbstractTracker.release`.
        """
        if self._model is None:
            return
        self._model = None
        if "cuda" in self.device:
            import torch

            torch.cuda.empty_cache()
        logger.info("released %s", self.model_path)
