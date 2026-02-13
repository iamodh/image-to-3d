"""Validation utilities for 3D-printable meshes."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh


def validate_mesh(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Validate mesh quality for 3D printing and return a report."""
    issues: list[str] = []

    is_watertight = bool(mesh.is_watertight)
    if not is_watertight:
        issues.append("메시가 watertight하지 않음 (구멍이 있을 수 있음)")

    dims = mesh.extents
    for i, axis in enumerate(["X", "Y", "Z"]):
        if float(dims[i]) < 1.0:
            issues.append(f"{axis}축 크기가 1mm 미만 ({dims[i]:.2f}mm)")

    if mesh.face_normals is not None and len(mesh.face_normals) > 0:
        inverted_ratio = float(np.sum(mesh.face_normals[:, 2] < 0) / len(mesh.face_normals))
        if inverted_ratio > 0.6:
            issues.append("면 법선이 대부분 뒤집혀 있을 수 있음")

    volume = float(mesh.volume) if is_watertight else -1.0

    return {
        "watertight": is_watertight,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "dimensions": f"{dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm",
        "volume": round(volume, 2),
        "issues": issues,
    }
