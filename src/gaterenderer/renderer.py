"""GateRenderer: batched ray-cast of quad gate frames into binary masks."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from ._kernel import load_raycast_module
from .transforms import matrix_from_quat, quat_from_euler_xyz

class GateRenderer:
    """Render binary segmentation masks of sqaure gates *frames*.

    For every camera and pixel, the precomputed camera-space ray is rotated to
    world space and intersected against the square. A pixel is set when its ray
    hits the border region of any square (the 75% central hole is cut out).

    The packed quad geometry is held in a per-instance global-memory tensor and
    passed to the kernel on each call, so multiple ``GateRenderer`` instances are
    fully independent and safe to interleave.

    Args:
        gate_config: dict with keys:
            "poses": list of (float, float, float)
            "roll": list of float
            "pitch": list of float
            "yaw": list of float
            "size": float
        n_cams: number of cameras rendered per :meth:`render` call.
        K: ``3x3`` pinhole intrinsics matching ``resolution``.
        D: OpenCV distortion coefficients (passed to ``cv2.undistortPoints``).
        resolution: ``(H, W)`` of the output masks.
        device: CUDA device (the kernel is CUDA-only).
        verbose: print the kernel compilation log on first construction.
    """

    def __init__(self, gate_config, n_cams, K, D, resolution, device="cuda", verbose=False):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

        self.n_cams = int(n_cams)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.H, self.W = self.resolution

        self._module = load_raycast_module(verbose=verbose)

        # Per-quad geometry, packed into a per-instance global-memory tensor.
        self._packed_quads = self._pack_quads(gate_config)

        # Precomputed, undistorted, normalized camera-space rays [H, W, 4].
        self.rays_cam = self._build_rays(K, D)

        # Reused output buffer (overwritten in place each render).
        self._output_buf = torch.zeros(
            (self.n_cams, self.H, self.W), dtype=torch.uint8, device=self.device
        )

    def _build_rays(self, K, D) -> torch.Tensor:
        K = np.asarray(K, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)

        y, x = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        pts = np.stack([x, y], axis=-1).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(pts, K, D).reshape(self.H, self.W, 2)

        # OpenCV camera space is +X right, +Y down, +Z forward. Flip to the
        # OpenGL-style viewing frame (+Y up, looking down -Z). Padded to 4
        # channels so the kernel can issue one coalesced 128-bit (float4) load.
        rays = np.zeros((self.H, self.W, 4), dtype=np.float32)
        rays[:, :, 0] = undistorted[:, :, 0]    # X stays right
        rays[:, :, 1] = -undistorted[:, :, 1]   # flip Y to up
        rays[:, :, 2] = -1.0                    # forward
        rays[:, :, :3] /= np.linalg.norm(rays[:, :, :3], axis=-1, keepdims=True)

        return torch.from_numpy(rays).to(self.device).contiguous()

    def _pack_quads(self, gate_config: dict) -> torch.Tensor:
        poses   = torch.tensor(gate_config["poses"])
        roll    = torch.tensor(gate_config.get("roll", [0.0] * len(poses)))
        pitch   = torch.tensor(gate_config.get("pitch", [0.0] * len(poses)))
        yaw     = torch.tensor(gate_config.get("yaw", [0.0] * len(poses)))
        size    = torch.tensor(gate_config.get("size", [1.0] * len(poses)))
        n_quads = poses.shape[0]

        rot_mats = matrix_from_quat(quat_from_euler_xyz(roll * torch.pi / 180, pitch * torch.pi / 180, yaw * torch.pi / 180))
        normal = rot_mats[:, :3, 0]
        u = -rot_mats[:, :3, 1]
        v = rot_mats[:, :3, 2]
        quads = torch.zeros((n_quads, 16), dtype=torch.float32, device=self.device)
        quads[:, 0:3] = poses   # center
        quads[:, 3:6] = normal  # normal
        quads[:, 6:9] = u       # e_1
        quads[:, 9:12] = v      # e_2
        quads[:, 12] = (normal * poses).sum(-1) # nd = dot(normal, center)
        quads[:, 13] = 2.0 / size
        # pad to 16 floats (128 bits) for coalesced loads in the kernel
        quads[:, 14] = 0.0
        quads[:, 15] = 0.0  

        return quads.contiguous()

    @torch.no_grad()
    def render(self, c2w_matrices) -> torch.Tensor:
        """Render masks for ``n_cams`` camera-to-world matrices.

        Args:
            c2w_matrices: ``[n_cams, 4, 4]`` (or ``[4, 4]`` when ``n_cams == 1``).

        Returns:
            ``uint8`` tensor ``[n_cams, H, W]``. This is an internal buffer reused
            across calls; clone it if you need to keep the result.
        """
        c2w = torch.as_tensor(c2w_matrices, dtype=torch.float32, device=self.device)
        if c2w.ndim == 2:
            c2w = c2w.unsqueeze(0)
        if c2w.shape[0] != self.n_cams or c2w.shape[1:] != (4, 4):
            raise ValueError(
                f"c2w_matrices must have shape [{self.n_cams}, 4, 4], got {tuple(c2w.shape)}"
            )

        self._module.raycast(
            self.rays_cam, c2w.contiguous(), self._packed_quads, self._output_buf, self.W, self.H
        )
        torch.cuda.synchronize()
        return self._output_buf
