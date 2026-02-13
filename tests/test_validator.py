import trimesh

from src.validator import validate_mesh


def test_validate_mesh_for_watertight_box():
    mesh = trimesh.creation.box(extents=[10.0, 20.0, 30.0])
    report = validate_mesh(mesh)

    assert report["watertight"] is True
    assert report["vertices"] > 0
    assert report["faces"] > 0
    assert report["issues"] == []


def test_validate_mesh_detects_small_dimension_issue():
    mesh = trimesh.creation.box(extents=[0.5, 2.0, 2.0])
    report = validate_mesh(mesh)

    assert any("1mm 미만" in issue for issue in report["issues"])
