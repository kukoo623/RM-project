"""Hand tracker using OpenCV skin-color segmentation.

No neural network or external model required.  Works at 30+ fps on a
Raspberry Pi by running at reduced resolution.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.vision.hand_tracking import HandTrackerResult

logger = logging.getLogger(__name__)


class OpenCVHandTracker:
    """Skin-color segmentation hand tracker.

    Converts the frame to HSV, applies a skin-colour mask, then finds the
    largest contour (excluding the face region when provided).

    Implements the ``HandTracker`` protocol.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.02,
        processing_scale: float = 0.5,
    ) -> None:
        """Initialize.

        Args:
            min_area_ratio: Minimum contour area as a fraction of the
                (full-resolution) frame area to count as a hand.
            processing_scale: Downscale factor for faster processing.
        """
        self._min_area_ratio = min_area_ratio
        self._scale = processing_scale

        # Morphological kernels
        self._kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def get_hand_position(
        self,
        img: NDArray[np.uint8],
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> HandTrackerResult:
        """Detect the largest hand-like skin region.

        Args:
            img: BGR image (full resolution).
            face_bbox: Optional ``(x, y, w, h)`` of the detected face in
                full-resolution coordinates.  The face region is masked out
                so the face itself is not mistaken for a hand.

        Returns:
            ``(palm_center_normalized, confidence)`` or ``(None, None)``.
        """
        h, w = img.shape[:2]
        frame_area = h * w

        # Work at reduced resolution for speed
        sw, sh = int(w * self._scale), int(h * self._scale)
        small = cv2.resize(img, (sw, sh))

        # Convert to HSV and apply skin mask
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Broad skin range covering various skin tones
        lower_skin = np.array([0, 30, 50], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Morphological cleanup
        mask = cv2.erode(mask, self._kernel_erode, iterations=1)
        mask = cv2.dilate(mask, self._kernel_dilate, iterations=2)

        # Mask out the face region if provided
        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            # Scale face bbox down and add margin
            margin = int(max(fw, fh) * 0.3 * self._scale)
            fx_s = max(0, int(fx * self._scale) - margin)
            fy_s = max(0, int(fy * self._scale) - margin)
            fw_s = int(fw * self._scale) + 2 * margin
            fh_s = int(fh * self._scale) + 2 * margin
            cv2.rectangle(
                mask,
                (fx_s, fy_s),
                (fx_s + fw_s, fy_s + fh_s),
                0,
                thickness=-1,
            )

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        # Pick the largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Check minimum area (in full-resolution terms)
        full_area = area / (self._scale * self._scale)
        if full_area / frame_area < self._min_area_ratio:
            return None, None

        # Compute centroid
        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            return None, None

        cx_small = moments["m10"] / moments["m00"]
        cy_small = moments["m01"] / moments["m00"]

        # Map back to full-resolution pixel coords
        cx = cx_small / self._scale
        cy = cy_small / self._scale

        # Normalize to [-1, 1]
        nx = (cx / w) * 2.0 - 1.0
        ny = (cy / h) * 2.0 - 1.0

        center = np.array([nx, ny], dtype=np.float32)
        confidence = min(1.0, full_area / frame_area / 0.1)  # Normalize

        return center, confidence

    def close(self) -> None:
        """No resources to release."""
        pass
