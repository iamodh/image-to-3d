"""Mesh post-processing utilities for 3D printing."""

from __future__ import annotations

import trimesh


def _keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:
        return mesh
    if not parts:
        return mesh
    return max(parts, key=lambda m: m.area)


def _voxel_solidify(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # Voxel solidify is a practical fallback for non-watertight meshes.
    pitch = max(float(mesh.extents.max()) / 220.0, 0.2)
    voxel = mesh.voxelized(pitch).fill()
    solid = voxel.marching_cubes
    return solid


def process_mesh(
    mesh: trimesh.Trimesh,
    target_height: float = 100.0,
    smooth_iterations: int = 8,
    keep_largest: bool = True,
    enforce_watertight: bool = True,
) -> trimesh.Trimesh:
    """Repair, normalize, and align a mesh for printing."""
    if keep_largest:
        mesh = _keep_largest_component(mesh)

    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass

    mesh.merge_vertices()

    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()

    if smooth_iterations > 0:
        try:
            trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iterations)
        except Exception:
            pass

    current_height = float(mesh.extents.max())
    if current_height > 0:
        mesh.apply_scale(target_height / current_height)

    mesh.apply_translation([0.0, 0.0, -float(mesh.bounds[0][2])])

    centroid = mesh.centroid.copy()
    centroid[2] = 0.0
    mesh.apply_translation(-centroid)

    if enforce_watertight and not mesh.is_watertight:
        try:
            mesh = _voxel_solidify(mesh)
            mesh.apply_translation([0.0, 0.0, -float(mesh.bounds[0][2])])
            centroid = mesh.centroid.copy()
            centroid[2] = 0.0
            mesh.apply_translation(-centroid)
        except Exception:
            pass

    if enforce_watertight and not mesh.is_watertight:
        try:
            mesh = mesh.convex_hull
            mesh.apply_translation([0.0, 0.0, -float(mesh.bounds[0][2])])
            centroid = mesh.centroid.copy()
            centroid[2] = 0.0
            mesh.apply_translation(-centroid)
        except Exception:
            pass

    return mesh
