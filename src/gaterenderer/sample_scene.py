"""Example gate geometry and camera calibration used by the demo scripts.

These are plain NumPy arrays (no CUDA at import time). Pass them straight into
:class:`gaterenderer.GateRenderer`.
"""
from __future__ import annotations

import numpy as np

# Four square gate frames in world space (NWU: x-forward, y-left, z-up).
SAMPLE_GATE_CONFIG = {
    "poses": [(1.5, -1.5, 0.5), (3.0, 0.0, 0.5), (1.5, 1.5, 0.5), 
              (-1.5, 1.5, 0.5), (-3.0, 0.0, 0.5), (-1.5, -1.5, 0.5)],
    "roll": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "pitch": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "yaw": [0.0, 90.0, 0.0, 0.0, 90.0, 0.0],
    "size": 1.0,
}

# Resolution (H, W) at which the intrinsics below were calibrated.
CALIB_RES = (244, 324)

# Pinhole intrinsics and OpenCV distortion coefficients for CALIB_RES.
SAMPLE_K = np.array([
    [181.79496728,   0.,         162.93307943],
    [  0.,         182.16557038, 162.99985265],
    [  0.,           0.,           1.        ],
])
SAMPLE_D = np.array([[-0.06720237, 0.00828066, 0.00372965, -0.00039607, -0.00919344]])


def scale_intrinsics(K: np.ndarray, calib_res, render_res) -> np.ndarray:
    """Scale a ``3x3`` intrinsics matrix from ``calib_res`` to ``render_res``.

    Uses the height ratio (matches the demo scripts; assumes aspect is preserved).
    """
    K = np.asarray(K, dtype=np.float64).copy()
    return K * (render_res[0] / calib_res[0])
