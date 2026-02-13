"""TripoSR-based mesh generation module."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image

ImageInput = Union[str, Path, Image.Image]


def generate_mesh(image: ImageInput, mc_resolution: int = 256):
    """Generate a mesh from a single image.

    This placeholder intentionally fails in CPU-only environments.
    Tests should replace this function with a mock.
    """
    raise RuntimeError(
        "GPU 기반 TripoSR 생성기는 아직 연결되지 않았습니다. "
        "Phase 2에서 구현하거나 테스트에서 mock을 사용하세요."
    )
