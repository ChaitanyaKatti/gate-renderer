"""JIT loader for the raycast CUDA extension.

The ``.cu`` / ``.cpp`` sources ship as package data under ``_kernels/`` and are
compiled on first use with :func:`torch.utils.cpp_extension.load_inline`. The
result is cached process-wide so the (one-time) compile happens at most once.
"""
from __future__ import annotations

import functools
from pathlib import Path

from torch.utils.cpp_extension import load_inline

_KERNEL_DIR = Path(__file__).resolve().parent / "_kernels"


@functools.lru_cache(maxsize=1)
def load_raycast_module(verbose: bool = False):
    """Compile (once) and return the raycast extension module.

    Exposes one function:
        raycast(rays_cam, c2w, quads_p, output_buf, W, H)
    where ``quads_p`` is the per-renderer packed geometry [n_quads, 16].
    """
    cuda_source = (_KERNEL_DIR / "raycast.cu").read_text()
    cpp_source = (_KERNEL_DIR / "raycast.cpp").read_text()

    return load_inline(
        name="gaterenderer_raycast",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["raycast"],
        extra_cuda_cflags=["-use_fast_math", "-O3"],
        verbose=verbose,
    )
