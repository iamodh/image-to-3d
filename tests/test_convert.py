from pathlib import Path

import trimesh

import convert


def test_run_pipeline_with_mocks(monkeypatch, tmp_path):
    steps = []
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

    def fake_remove_background(path):
        steps.append(("bg", str(path)))
        return "processed-image"

    def fake_generate_mesh(image, mc_resolution):
        steps.append(("gen", image, mc_resolution))
        return mesh

    def fake_process_mesh(raw_mesh, target_height):
        steps.append(("proc", raw_mesh is mesh, target_height))
        return raw_mesh

    def fake_validate_mesh(processed_mesh):
        steps.append(("val", processed_mesh is mesh))
        return {
            "watertight": True,
            "vertices": 8,
            "faces": 12,
            "dimensions": "1.0 x 1.0 x 1.0 mm",
            "volume": 1.0,
            "issues": [],
        }

    monkeypatch.setattr(convert.bg_remover, "remove_background", fake_remove_background)
    monkeypatch.setattr(convert.generator, "generate_mesh", fake_generate_mesh)
    monkeypatch.setattr(convert.processor, "process_mesh", fake_process_mesh)
    monkeypatch.setattr(convert.validator, "validate_mesh", fake_validate_mesh)

    output_path = tmp_path / "result.stl"
    code = convert.run([
        "-i",
        "input.png",
        "-o",
        str(output_path),
        "--height",
        "123",
        "--mc-resolution",
        "320",
    ])

    assert code == 0
    assert output_path.exists()
    assert steps == [
        ("bg", "input.png"),
        ("gen", "processed-image", 320),
        ("proc", True, 123.0),
        ("val", True),
    ]


def test_run_without_bg_remove(monkeypatch, tmp_path):
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

    monkeypatch.setattr(convert.generator, "generate_mesh", lambda image, mc_resolution: mesh)
    monkeypatch.setattr(convert.processor, "process_mesh", lambda raw_mesh, target_height: raw_mesh)
    monkeypatch.setattr(
        convert.validator,
        "validate_mesh",
        lambda processed_mesh: {
            "watertight": True,
            "vertices": 8,
            "faces": 12,
            "dimensions": "1.0 x 1.0 x 1.0 mm",
            "volume": 1.0,
            "issues": [],
        },
    )

    called = {"bg": False}

    def fake_remove_background(path):
        called["bg"] = True
        return path

    monkeypatch.setattr(convert.bg_remover, "remove_background", fake_remove_background)

    output_path = tmp_path / "no_bg.obj"
    code = convert.run([
        "-i",
        "input.png",
        "-o",
        str(output_path),
        "--format",
        "obj",
        "--no-bg-remove",
    ])

    assert code == 0
    assert called["bg"] is False
    assert output_path.exists()
