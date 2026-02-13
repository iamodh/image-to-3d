import math

import pytest
import trimesh

from src.processor import process_mesh


def test_process_mesh_scales_longest_axis_and_aligns_floor():
    mesh = trimesh.creation.box(extents=[10.0, 20.0, 40.0])
    mesh.apply_translation([5.0, -3.0, 7.5])

    processed = process_mesh(mesh, target_height=100.0)

    assert processed.extents.max() == pytest.approx(100.0, rel=1e-3)
    assert processed.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert processed.centroid[0] == pytest.approx(0.0, abs=1e-6)
    assert processed.centroid[1] == pytest.approx(0.0, abs=1e-6)
