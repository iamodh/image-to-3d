"""TripoSR-based mesh generation module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image
import trimesh

ImageInput = Union[str, Path, Image.Image]

_MODEL: Any = None


def _load_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch를 불러오지 못했습니다. GPU 환경에서 `pip install torch torchvision` 후 다시 시도하세요."
        ) from exc
    return torch


def _load_tsr():
    try:
        from tsr.system import TSR
    except Exception as exc:
        raise RuntimeError(
            "TripoSR를 불러오지 못했습니다. TripoSR 저장소/의존성을 설치한 뒤 다시 시도하세요."
        ) from exc
    return TSR


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    torch = _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 찾을 수 없습니다. Colab/클라우드 GPU 환경에서 실행하거나 generator를 mock으로 대체하세요."
        )

    tsr_cls = _load_tsr()
    model = tsr_cls.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to("cuda:0")
    _MODEL = model
    return _MODEL


def _load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.open(image).convert("RGB")


def _to_trimesh(raw_mesh: Any) -> trimesh.Trimesh:
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    vertices = _to_numpy(raw_mesh.vertices)
    faces = _to_numpy(raw_mesh.faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def generate_mesh(image: ImageInput, mc_resolution: int = 256) -> trimesh.Trimesh:
    """Generate a mesh from a single image using TripoSR."""
    if mc_resolution <= 0:
        raise ValueError("mc_resolution은 1 이상의 정수여야 합니다.")

    model = _load_model()
    torch = _load_torch()
    pil_image = _load_image(image)

    with torch.no_grad():
        scene_codes = model([pil_image], device="cuda:0")
        try:
            meshes = model.extract_mesh(
                scene_codes,
                resolution=mc_resolution,
                has_vertex_color=False,
            )
        except TypeError:
            # Backward compatibility for older TripoSR signatures.
            meshes = model.extract_mesh(scene_codes, resolution=mc_resolution)

    return _to_trimesh(meshes[0])
