from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src import generator


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _NoGrad()


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.array(arr, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _RawMesh:
    def __init__(self):
        self.vertices = _FakeTensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.faces = _FakeTensor([[0, 1, 2]])


class _FakeModel:
    def __call__(self, images, device):
        assert len(images) == 1
        assert device == "cuda:0"
        return "scene"

    def extract_mesh(self, scene_codes, resolution):
        assert scene_codes == "scene"
        assert resolution == 256
        return [_RawMesh()]


def test_generate_mesh_converts_triposr_output(monkeypatch):
    monkeypatch.setattr(generator, "_load_model", lambda: _FakeModel())
    monkeypatch.setattr(generator, "_load_torch", lambda: _FakeTorch())

    image = Image.new("RGB", (8, 8), (255, 255, 255))
    mesh = generator.generate_mesh(image)

    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)


def test_generate_mesh_validates_resolution():
    with pytest.raises(ValueError):
        generator.generate_mesh(Image.new("RGB", (1, 1)), mc_resolution=0)


def test_load_model_raises_when_torch_missing(monkeypatch):
    monkeypatch.setattr(generator, "_MODEL", None)

    def fail_load_torch():
        raise RuntimeError("PyTorch를 불러오지 못했습니다.")

    monkeypatch.setattr(generator, "_load_torch", fail_load_torch)

    with pytest.raises(RuntimeError):
        generator._load_model()
