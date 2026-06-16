"""GateRenderer — massively-parallel CUDA ray-cast renderer for quad gate masks.

Public API:
    GateRenderer        - render binary segmentation masks of quad gate frames.
    matrix_from_quat    - quaternion (wxyz) -> rotation matrix.
    quat_from_euler_xyz - roll/pitch/yaw -> quaternion (wxyz).
    sample_scene        - example gate geometry + camera calibration.

The kernel is JIT-compiled lazily the first time a ``GateRenderer`` is constructed.
"""
from . import sample_scene
from .renderer import GateRenderer
from .transforms import matrix_from_quat, quat_from_euler_xyz

__all__ = [
    "GateRenderer",
    "matrix_from_quat",
    "quat_from_euler_xyz",
    "sample_scene",
]
__version__ = "0.1.0"
