"""Throughput benchmark: many cameras rendered in parallel.

Run with the package installed (``pip install -e .[examples]``):
    python examples/benchmark.py
"""
import time

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from gaterenderer import GateRenderer, matrix_from_quat, quat_from_euler_xyz
from gaterenderer.sample_scene import CALIB_RES, SAMPLE_D, SAMPLE_K, SAMPLE_GATE_CONFIG, scale_intrinsics


def main():
    torch.random.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1024
    RENDER_RES = (61, 81)  # 1/4 of calibration resolution to fit more cameras in memory

    renderer = GateRenderer(
        gate_config=SAMPLE_GATE_CONFIG,
        K=scale_intrinsics(SAMPLE_K, CALIB_RES, RENDER_RES),
        D=SAMPLE_D,
        resolution=RENDER_RES,
        device=device,
        verbose=True,
    )

    cam_pos = torch.randn(BATCH_SIZE, 3, device=device) * 2.0
    cam_pos[:, 2] = 0.5
    roll = torch.rand(BATCH_SIZE, device=device) * 30.0
    pitch = torch.rand(BATCH_SIZE, device=device) * 30.0
    yaw = 180+torch.atan2(cam_pos[:, 1], cam_pos[:, 0]) * 180 / np.pi
    view = torch.eye(4, dtype=torch.float32, device=device).repeat(BATCH_SIZE, 1, 1)

    rot = matrix_from_quat(quat_from_euler_xyz(roll * np.pi / 180, pitch * np.pi / 180, yaw * np.pi / 180))
    view[:, :3, :3] = rot
    view[:, :3, 3] = cam_pos

    print("\n" + "=" * 50)
    print(f"BENCHMARK: {BATCH_SIZE} Cameras")
    print("=" * 50)

    # Warm up and save an example output
    seg = renderer.render(view)[:64]
    save_image(
        make_grid(
            torch.swapaxes(seg.unsqueeze(-1).cpu().float().unsqueeze(1), 1, -1).squeeze(-1),
            nrow=round(seg.shape[0] ** 0.5), pad_value=0.5,
        ),
        "./benchmark_output.png",
    )

    # Benchmark
    start = time.time()
    num_iterations = 10000
    for _ in tqdm(range(num_iterations)):
        _ = renderer.render(view)
    elapsed = time.time() - start

    fps = (BATCH_SIZE * num_iterations) / elapsed
    print(f"Time for {BATCH_SIZE * num_iterations} renders: {elapsed:.3f}s")
    print(f"FPS (total cameras): {fps:.1f}")
    print(f"Iterations per second: {fps / BATCH_SIZE:.4f}")


if __name__ == "__main__":
    main()
