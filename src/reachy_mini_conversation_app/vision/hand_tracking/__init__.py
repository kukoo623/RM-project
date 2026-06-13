"""Hand tracking module — protocol and result types."""

from __future__ import annotations

from typing import Protocol, SupportsFloat, TypeAlias

import numpy as np
from numpy.typing import NDArray


# (palm_center_normalized, confidence)
# palm_center_normalized: ndarray of shape (2,) with values in [-1, 1]
# confidence: float in [0, 1] or None
HandTrackerResult: TypeAlias = tuple[NDArray[np.float32] | None, SupportsFloat | None]


class HandTracker(Protocol):
    """Protocol for hand tracking backends."""

    def get_hand_position(self, img: NDArray[np.uint8]) -> HandTrackerResult:
        """Return the detected hand position for a frame.

        Args:
            img: BGR image from the camera.

        Returns:
            A tuple ``(palm_center, confidence)`` where ``palm_center`` is a
            2-element ndarray with values in [-1, 1], or ``(None, None)``
            when no hand is detected.
        """
        ...
