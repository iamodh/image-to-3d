"""Mesh post-processing utilities for 3D printing."""

from __future__ import annotations

import trimesh


def process_mesh(mesh: trimesh.Trimesh, target_height: float = 100.0) -> trimesh.Trimesh:
    """Repair, normalize, and align a mesh for printing."""
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)

    mesh.merge_vertices()

    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()

    current_height = float(mesh.extents.max())
    if current_height > 0:
        mesh.apply_scale(target_height / current_height)

    mesh.apply_translation([0.0, 0.0, -float(mesh.bounds[0][2])])

    centroid = mesh.centroid.copy()
    centroid[2] = 0.0
    mesh.apply_translation(-centroid)
    return mesh
