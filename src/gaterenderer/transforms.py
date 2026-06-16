"""Rotation helpers (quaternions use the ``wxyz`` convention)."""
from __future__ import annotations

import torch


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions ``(..., 4)`` in ``wxyz`` order to ``(..., 3, 3)`` matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert roll/pitch/yaw (radians) to quaternions ``(..., 4)`` in ``wxyz`` order."""
    cy = torch.cos(yaw * 0.5); sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qw, qx, qy, qz], dim=-1)
