"""Lightweight face tracker using OpenCV YuNet (FaceDetectorYN).

Works on ARM64/aarch64 where MediaPipe is unavailable.
Implements the HeadTracker protocol used by CameraWorker.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerResult

logger = logging.getLogger(__name__)

# YuNet model is expected alongside this file
_MODEL_PATH = Path(__file__).parent / "face_detection_yunet_2023mar.onnx"

# Haar cascade fallback (bundled with OpenCV)
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class OpenCVFaceTracker:
    """Face tracker using OpenCV's built-in YuNet detector with Haar fallback.

    Implements the ``HeadTracker`` protocol so it can be plugged directly
    into :class:`CameraWorker`.
    """

    def __init__(
        self,
        input_width: int = 320,
        input_height: int = 240,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ) -> None:
        """Initialize the face tracker.

        Args:
            input_width: Detector input width (smaller = faster).
            input_height: Detector input height.
            score_threshold: Minimum detection confidence.
            nms_threshold: Non-maximum suppression threshold.
        """
        self._input_w = input_width
        self._input_h = input_height
        self._use_yunet = False
        self._haar_cascade: Any = None

        # Try YuNet first
        if _MODEL_PATH.exists():
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    model=str(_MODEL_PATH),
                    config="",
                    input_size=(input_width, input_height),
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold,
                    top_k=1,
                )
                self._use_yunet = True
                logger.info(
                    "YuNet face detector loaded (input %dx%d)",
                    input_width,
                    input_height,
                )
            except Exception as exc:
                logger.warning("Failed to load YuNet model: %s", exc)
        else:
            logger.warning("YuNet model not found at %s", _MODEL_PATH)

        # Fallback to Haar cascade
        if not self._use_yunet:
            try:
                self._haar_cascade = cv2.CascadeClassifier(_HAAR_PATH)
                if self._haar_cascade.empty():
                    raise RuntimeError("Haar cascade file is empty")
                logger.info("Using Haar cascade fallback face detector")
            except Exception as exc:
                logger.error("Failed to load Haar cascade: %s", exc)
                self._haar_cascade = None

    def get_head_position(self, img: NDArray[np.uint8]) -> HeadTrackerResult:
        """Detect face and return eye-center position normalized to [-1, 1].

        Args:
            img: BGR image from the camera.

        Returns:
            Tuple of (eye_center, roll) where eye_center is an ndarray of
            shape (2,) with values in [-1, 1], and roll is in radians.
            Returns (None, None) if no face is detected.
        """
        h, w = img.shape[:2]

        if self._use_yunet:
            return self._detect_yunet(img, w, h)
        elif self._haar_cascade is not None:
            return self._detect_haar(img, w, h)

        return None, None

    def _detect_yunet(
        self, img: NDArray[np.uint8], w: int, h: int
    ) -> HeadTrackerResult:
        """Detect using YuNet (returns eye landmarks + roll)."""
        # Resize if needed for detector input
        if w != self._input_w or h != self._input_h:
            resized = cv2.resize(img, (self._input_w, self._input_h))
            scale_x = w / self._input_w
            scale_y = h / self._input_h
        else:
            resized = img
            scale_x = 1.0
            scale_y = 1.0

        self._detector.setInputSize((self._input_w, self._input_h))
        _, faces = self._detector.detect(resized)

        if faces is None or len(faces) == 0:
            return None, None

        # faces shape: (N, 15) = [x, y, w, h,
        #   right_eye_x, right_eye_y, left_eye_x, left_eye_y,
        #   nose_x, nose_y, right_mouth_x, right_mouth_y,
        #   left_mouth_x, left_mouth_y, score]
        face = faces[0]

        # Eye landmarks (in resized coords)
        right_eye_x = face[4] * scale_x
        right_eye_y = face[5] * scale_y
        left_eye_x = face[6] * scale_x
        left_eye_y = face[7] * scale_y

        # Eye center in pixel coords
        eye_cx = (left_eye_x + right_eye_x) / 2.0
        eye_cy = (left_eye_y + right_eye_y) / 2.0

        # Normalize to [-1, 1]: center of image = (0, 0)
        nx = (eye_cx / w) * 2.0 - 1.0
        ny = (eye_cy / h) * 2.0 - 1.0

        # Compute roll from eye positions
        dx = right_eye_x - left_eye_x
        dy = right_eye_y - left_eye_y
        roll = math.atan2(dy, dx) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0

        eye_center = np.array([nx, ny], dtype=np.float32)
        return eye_center, roll

    def _detect_haar(
        self, img: NDArray[np.uint8], w: int, h: int
    ) -> HeadTrackerResult:
        """Detect using Haar cascade (bounding box only, no landmarks)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize for speed
        scale = 0.5
        small = cv2.resize(gray, (int(w * scale), int(h * scale)))

        faces = self._haar_cascade.detectMultiScale(
            small,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            return None, None

        # Pick the largest face
        areas = faces[:, 2] * faces[:, 3]
        best_idx = np.argmax(areas)
        fx, fy, fw, fh = faces[best_idx]

        # Scale back to original coords
        fx = int(fx / scale)
        fy = int(fy / scale)
        fw = int(fw / scale)
        fh = int(fh / scale)

        # Use face center as approximate eye center
        cx = fx + fw / 2.0
        cy = fy + fh * 0.4  # Eyes are roughly at 40% from top of face bbox

        nx = (cx / w) * 2.0 - 1.0
        ny = (cy / h) * 2.0 - 1.0

        eye_center = np.array([nx, ny], dtype=np.float32)
        return eye_center, 0.0  # No roll estimate from Haar

    def close(self) -> None:
        """Release resources."""
        self._detector = None
        self._haar_cascade = None
        logger.info("OpenCVFaceTracker closed")
