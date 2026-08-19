"""Build the Taisofuku static-equilibrium visual review Blend and previews."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloth2bones.static_equilibrium import EquilibriumParameters, EquilibriumResult, optimize_static_equilibrium  # noqa: E402
from tools.export_ryuon_md_motion import _rest_matrices, _set_motion_pose, _set_rest_pose  # noqa: E402

DOWNLOADS_ROOT = Path("D:/") / "Users" / "Origin" / "Downloads"
SOURCE_DEFAULT = DOWNLOADS_ROOT / "RyuonTaisofuku_physbone_test.blend"
OUTPUT_DEFAULT = DOWNLOADS_ROOT / "cloth_poc_out" / "taisofuku_static_review"
EXPECTED_SOURCE_SHA256 = "00eace7d79ed201dd0c5684c3301f9af6c5653bff05c215e4d8ed5c6433801df"
EXPECTED_BODY_VERTICES = 12138
EXPECTED_SHIRT_VERTICES = 121471
EXPECTED_SHIRT_FACES = 115103
EXPECTED_MAIN_VERTICES = 87005
EXPECTED_MAIN_FACES = 86865
MAIN_OBJECT = "Taisofuku_Shirt"
BODY_OBJECT = "ComeBody"
ARMATURE_OBJECT = "ComeBody_Armature"
REQUIRED_GROUPS = ("Spine", "Chest", "Shoulder.L", "Shoulder.R", "Upper_arm.L", "Upper_arm.R")
PARAMETERS = {
    "stiff": EquilibriumParameters(edge=8000.0, smooth=12.0, tether=120.0, gravity=0.30, collision=50000.0),
    "balanced": EquilibriumParameters(edge=5000.0, smooth=6.0, tether=60.0, gravity=0.45, collision=50000.0),
    "soft": EquilibriumParameters(edge=3000.0, smooth=3.0, tether=30.0, gravity=0.60, collision=50000.0),
}
COLLECTION_NAMES = ("RYUON_A", "A_LBS_BASELINE", "A_OPT_STIFF", "A_OPT_BALANCED", "A_OPT_SOFT")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_object(name: str, object_type: str) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != object_type:
        raise RuntimeError(f"Required {object_type} object {name!r} was not found")
    return obj


def _component_data(mesh: bpy.types.Mesh) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return unique raw edges, main vertex indices, and main face indices."""

    face_count = len(mesh.polygons)
    parent = list(range(face_count))
    size = [1] * face_count

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    edge_owner: dict[tuple[int, int], int] = {}
    face_vertices: list[tuple[int, ...]] = []
    for face_index, polygon in enumerate(mesh.polygons):
        vertices = tuple(int(index) for index in polygon.vertices)
        face_vertices.append(vertices)
        for loop_index, left in enumerate(vertices):
            right = vertices[(loop_index + 1) % len(vertices)]
            edge = (left, right) if left < right else (right, left)
            previous = edge_owner.get(edge)
            if previous is None:
                edge_owner[edge] = face_index
            else:
                union(face_index, previous)

    components: Counter[int] = Counter(find(index) for index in range(face_count))
    main_root, _ = components.most_common(1)[0]
    main_faces = [index for index, vertices in enumerate(face_vertices) if vertices and find(index) == main_root]
    main_vertices = sorted({vertex for face_index in main_faces for vertex in face_vertices[face_index]})
    main_vertex_set = set(main_vertices)
    main_edges = sorted(
        edge
        for edge in edge_owner
        if edge[0] in main_vertex_set and edge[1] in main_vertex_set
    )
    return np.asarray(main_edges, dtype=np.int64), np.asarray(main_vertices, dtype=np.int64), main_faces


def _world_mesh_data(obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph, with_normals: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        positions = np.asarray(
            [[float(value) for value in evaluated.matrix_world @ vertex.co] for vertex in mesh.vertices],
            dtype=np.float64,
        )
        if not with_normals:
            return positions, None
        normal_matrix = evaluated.matrix_world.to_3x3().inverted().transposed()
        normals = []
        for vertex in mesh.vertices:
            normal = normal_matrix @ vertex.normal
            if normal.length <= 1.0e-12:
                raise RuntimeError(f"Object {obj.name} contains a zero-length evaluated vertex normal")
            normals.append([float(value) for value in normal.normalized()])
        return positions, np.asarray(normals, dtype=np.float64)
    finally:
        evaluated.to_mesh_clear()


def _group_weights(shirt: bpy.types.Object) -> dict[str, np.ndarray]:
    group_indices = {}
    for name in REQUIRED_GROUPS:
        group = shirt.vertex_groups.get(name)
        if group is None:
            raise RuntimeError(f"Taisofuku_Shirt is missing required vertex group {name!r}")
        group_indices[group.index] = name
    weights = {name: np.zeros(len(shirt.data.vertices), dtype=np.float64) for name in REQUIRED_GROUPS}
    for vertex in shirt.data.vertices:
        for item in vertex.groups:
            name = group_indices.get(item.group)
            if name is not None:
                weights[name][vertex.index] = float(item.weight)
    return weights


def _anchor_weights(shirt: bpy.types.Object, main_indices: np.ndarray, main_rest: np.ndarray) -> np.ndarray:
    weights = _group_weights(shirt)
    torso = np.maximum(weights["Spine"], weights["Chest"])[main_indices]
    shoulder = np.maximum(weights["Shoulder.L"], weights["Shoulder.R"])[main_indices]
    arm = np.maximum(weights["Upper_arm.L"], weights["Upper_arm.R"])[main_indices]
    pin_confidence = np.clip(np.maximum(torso - 0.5 * arm, 0.0) + 0.35 * shoulder, 0.0, 1.0)
    anchor = 0.15 + 0.85 * pin_confidence
    minimum = main_rest.min(axis=0)
    maximum = main_rest.max(axis=0)
    shirt_width = float(maximum[0] - minimum[0])
    center_x = float((maximum[0] + minimum[0]) * 0.5)
    z90 = float(np.percentile(main_rest[:, 2], 90.0))
    collar = (
        (np.abs(main_rest[:, 0] - center_x) <= shirt_width * 0.18)
        & (main_rest[:, 2] >= z90)
        & (torso >= arm)
    )
    anchor[collar] = 1.0
    return np.clip(anchor, 0.15, 1.0)


def _collision_associations(body: bpy.types.Object, main_base_a: np.ndarray, depsgraph: bpy.types.Depsgraph) -> tuple[np.ndarray, np.ndarray]:
    body_positions, body_normals = _world_mesh_data(body, depsgraph, with_normals=True)
    assert body_normals is not None
    centroid = body_positions.mean(axis=0)
    outward = np.sum(body_normals * (body_positions - centroid), axis=1) < 0.0
    body_normals[outward] *= -1.0
    tree = KDTree(len(body_positions))
    for index, point in enumerate(body_positions):
        tree.insert(Vector(point), index)
    tree.balance()
    collision_points = np.empty_like(main_base_a)
    collision_normals = np.empty_like(main_base_a)
    for index, point in enumerate(main_base_a):
        _, body_index, _ = tree.find(Vector(point))
        collision_points[index] = body_positions[body_index]
        collision_normals[index] = body_normals[body_index]
    return collision_points, collision_normals


def _propagate_to_full_mesh(
    full_rest: np.ndarray,
    full_base_a: np.ndarray,
    main_indices: np.ndarray,
    main_rest: np.ndarray,
    main_delta: np.ndarray,
) -> np.ndarray:
    final_positions = full_base_a.copy()
    final_positions[main_indices] = full_base_a[main_indices] + main_delta
    main_tree = KDTree(len(main_rest))
    for index, point in enumerate(main_rest):
        main_tree.insert(Vector(point), index)
    main_tree.balance()
    main_mask = np.zeros(len(full_rest), dtype=bool)
    main_mask[main_indices] = True
    for index in np.flatnonzero(~main_mask):
        nearest = main_tree.find_n(Vector(full_rest[index]), 4)
        if not nearest:
            raise RuntimeError("Main component KDTree returned no nearest vertices")
        distances = np.asarray([item[2] for item in nearest], dtype=np.float64)
        nearest_indices = np.asarray([item[1] for item in nearest], dtype=np.int64)
        weights = 1.0 / np.square(distances + 1.0e-6)
        weights /= weights.sum()
        final_positions[index] = full_base_a[index] + np.sum(weights[:, None] * main_delta[nearest_indices], axis=0)
    return final_positions


def _edge_metrics(rest: np.ndarray, positions: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    rest_lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    current_lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    strain = (current_lengths - rest_lengths) / np.maximum(rest_lengths, 1.0e-12)
    absolute = np.abs(strain)
    return {
        "rms": float(np.sqrt(np.mean(strain * strain))),
        "p95_absolute": float(np.percentile(absolute, 95.0)),
        "max_absolute": float(np.max(absolute)),
    }


def _candidate_report(
    name: str,
    parameters: EquilibriumParameters,
    result: EquilibriumResult,
    main_rest: np.ndarray,
    main_base_a: np.ndarray,
    main_edges: np.ndarray,
    collision_points: np.ndarray,
    collision_normals: np.ndarray,
    output_vertices: int,
    output_faces: int,
) -> dict[str, object]:
    correction_magnitude = np.linalg.norm(result.displacement, axis=1)
    signed_clearance = np.sum((result.positions - collision_points) * collision_normals, axis=1) - parameters.clearance
    nan_count = int(np.isnan(result.positions).sum() + np.isnan(result.displacement).sum())
    inf_count = int(np.isinf(result.positions).sum() + np.isinf(result.displacement).sum())
    edge_strain = _edge_metrics(main_rest, result.positions, main_edges)
    valid_numeric = (
        nan_count == 0
        and inf_count == 0
        and output_vertices == EXPECTED_SHIRT_VERTICES
        and output_faces == EXPECTED_SHIRT_FACES
        and float(np.max(correction_magnitude)) <= 0.10
        and edge_strain["max_absolute"] <= 0.20
    )
    weighted_energies = {
        "edge": parameters.edge * result.energies["edge"],
        "smooth": parameters.smooth * result.energies["smooth"],
        "tether": parameters.tether * result.energies["tether"],
        "gravity": parameters.gravity * result.energies["gravity"],
        "collision": parameters.collision * result.energies["collision"],
    }
    return {
        "name": name,
        "parameter_set": {
            "edge": parameters.edge,
            "smooth": parameters.smooth,
            "tether": parameters.tether,
            "gravity": parameters.gravity,
            "collision": parameters.collision,
            "clearance": parameters.clearance,
        },
        "iterations": parameters.iterations,
        "learning_rate": parameters.learning_rate,
        "final_total_energy": result.total_energy,
        "energy": result.energies,
        "weighted_energy": weighted_energies,
        "correction_displacement": {
            "mean": float(np.mean(correction_magnitude)),
            "p50": float(np.percentile(correction_magnitude, 50.0)),
            "p95": float(np.percentile(correction_magnitude, 95.0)),
            "max": float(np.max(correction_magnitude)),
        },
        "edge_strain": edge_strain,
        "collision_signed_clearance": {
            "min": float(np.min(signed_clearance)),
            "p01": float(np.percentile(signed_clearance, 1.0)),
            "mean": float(np.mean(signed_clearance)),
            "count_lt_0": int(np.count_nonzero(signed_clearance < 0.0)),
            "count_lt_minus_0_001": int(np.count_nonzero(signed_clearance < -0.001)),
        },
        "nan_count": nan_count,
        "inf_count": inf_count,
        "output_vertices": output_vertices,
        "output_faces": output_faces,
        "topology_preserved": output_vertices == EXPECTED_SHIRT_VERTICES and output_faces == EXPECTED_SHIRT_FACES,
        "valid_numeric": valid_numeric,
        "numeric_gate": {
            "max_correction_displacement_m": 0.10,
            "max_absolute_edge_strain": 0.20,
        },
    }


def _new_collection(name: str) -> bpy.types.Collection:
    if bpy.data.collections.get(name) is not None:
        raise RuntimeError(f"Source already contains output collection {name!r}")
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def _static_duplicate(
    source: bpy.types.Object,
    world_positions: np.ndarray,
    collection: bpy.types.Collection,
    name: str,
) -> bpy.types.Object:
    if len(world_positions) != len(source.data.vertices):
        raise RuntimeError(f"Static duplicate vertex mismatch for {name}: {len(world_positions)} != {len(source.data.vertices)}")
    result = source.copy()
    result.data = source.data.copy()
    result.name = name
    result.animation_data_clear()
    for modifier in list(result.modifiers):
        result.modifiers.remove(modifier)
    collection.objects.link(result)
    result.hide_viewport = False
    result.hide_render = False
    result.hide_set(False)
    inverse = result.matrix_world.inverted()
    for index, point in enumerate(world_positions):
        result.data.vertices[index].co = inverse @ Vector(point)
    result.data.update()
    return result


def _world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not corners:
        raise RuntimeError("Cannot render a scene without mesh bounds")
    minimum = Vector((min(point.x for point in corners), min(point.y for point in corners), min(point.z for point in corners)))
    maximum = Vector((max(point.x for point in corners), max(point.y for point in corners), max(point.z for point in corners)))
    return minimum, maximum


def _look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def _set_collection_visibility(collections: dict[str, bpy.types.Collection], active: str) -> None:
    for name, collection in collections.items():
        collection.hide_viewport = name != "RYUON_A" and name != active
        collection.hide_render = name != "RYUON_A" and name != active


def _render_previews(collections: dict[str, bpy.types.Collection], body: bpy.types.Object, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.world.color = (0.025, 0.025, 0.025)
    camera_data = bpy.data.cameras.new("STATIC_REVIEW_CAMERA_DATA")
    camera = bpy.data.objects.new("STATIC_REVIEW_CAMERA", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    scene.camera = camera
    rendered: list[str] = []
    view_directions = {
        "front": (Vector((0.0, -1.0, 0.0)), None),
        "back": (Vector((0.0, 1.0, 0.0)), None),
        "left_underarm": (Vector((-1.0, -0.85, 0.08)).normalized(), -1.0),
        "right_underarm": (Vector((1.0, -0.85, 0.08)).normalized(), 1.0),
    }
    filename_prefix = {
        "A_LBS_BASELINE": "baseline",
        "A_OPT_STIFF": "stiff",
        "A_OPT_BALANCED": "balanced",
        "A_OPT_SOFT": "soft",
    }
    for candidate_name, collection in collections.items():
        if candidate_name == "RYUON_A":
            continue
        _set_collection_visibility(collections, candidate_name)
        candidate_objects = [obj for obj in collection.objects if obj.type == "MESH"]
        minimum, maximum = _world_bounds([body, *candidate_objects])
        center = (minimum + maximum) * 0.5
        extent = maximum - minimum
        for view_name, (direction, side) in view_directions.items():
            target = center
            if side is not None:
                target = center + Vector((side * extent.x * 0.22, -extent.y * 0.08, extent.z * 0.10))
            distance = max(extent.length * 1.8, 1.0)
            camera.location = target + direction * distance
            _look_at(camera, target)
            camera.data.type = "ORTHO"
            camera.data.ortho_scale = max(extent.z * (0.68 if side is not None else 1.18), extent.x * 1.18, extent.y * 1.18, 0.1)
            filename = f"{filename_prefix[candidate_name]}_{view_name}.png"
            filepath = output_dir / filename
            scene.render.filepath = str(filepath)
            bpy.ops.render.render(write_still=True)
            if not filepath.is_file() or filepath.stat().st_size <= 0:
                raise RuntimeError(f"Workbench render did not produce a non-empty PNG: {filepath}")
            rendered.append(str(filepath))
    return rendered


def _validate_source(source: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    stat = source.stat()
    digest = _sha256(source)
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"Source SHA-256 mismatch: expected {EXPECTED_SOURCE_SHA256}, got {digest}")
    return {"path": str(source), "sha256": digest, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def main() -> None:
    args = _args()
    source = args.source.resolve()
    output_root = args.output_root.resolve()
    source_before = _validate_source(source)
    output_root.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(source))
    armature = _require_object(ARMATURE_OBJECT, "ARMATURE")
    body = _require_object(BODY_OBJECT, "MESH")
    shirt = _require_object(MAIN_OBJECT, "MESH")
    if len(body.data.vertices) != EXPECTED_BODY_VERTICES:
        raise RuntimeError(f"ComeBody vertex count mismatch: expected {EXPECTED_BODY_VERTICES}, got {len(body.data.vertices)}")
    if len(shirt.data.vertices) != EXPECTED_SHIRT_VERTICES or len(shirt.data.polygons) != EXPECTED_SHIRT_FACES:
        raise RuntimeError(
            f"Taisofuku_Shirt topology mismatch: expected {EXPECTED_SHIRT_VERTICES} vertices/{EXPECTED_SHIRT_FACES} faces, "
            f"got {len(shirt.data.vertices)} vertices/{len(shirt.data.polygons)} faces"
        )
    main_edges, main_indices, main_faces = _component_data(shirt.data)
    if len(main_indices) != EXPECTED_MAIN_VERTICES or len(main_faces) != EXPECTED_MAIN_FACES:
        raise RuntimeError(
            f"Main component mismatch: expected {EXPECTED_MAIN_VERTICES} vertices/{EXPECTED_MAIN_FACES} faces, "
            f"got {len(main_indices)} vertices/{len(main_faces)} faces"
        )
    if not main_edges.size:
        raise RuntimeError("Main component has no edges")

    rest_matrices = _rest_matrices(armature)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    _set_rest_pose(armature, rest_matrices)
    full_rest, _ = _world_mesh_data(shirt, depsgraph)
    _set_motion_pose(armature, rest_matrices, {"Upper_arm.L", "Upper_arm.R"}, 1.0)
    full_base_a, _ = _world_mesh_data(shirt, depsgraph)
    main_rest = full_rest[main_indices]
    main_base_a = full_base_a[main_indices]
    if len(full_rest) != EXPECTED_SHIRT_VERTICES or len(full_base_a) != EXPECTED_SHIRT_VERTICES:
        raise RuntimeError("Evaluated Taisofuku_Shirt vertex count changed under the Armature modifier")
    anchor = _anchor_weights(shirt, main_indices, main_rest)
    collision_points, collision_normals = _collision_associations(body, main_base_a, depsgraph)

    candidate_results: dict[str, EquilibriumResult] = {}
    candidate_reports: dict[str, dict[str, object]] = {}
    candidate_positions: dict[str, np.ndarray] = {}
    for name, parameters in PARAMETERS.items():
        result = optimize_static_equilibrium(
            main_rest,
            main_base_a,
            main_edges,
            collision_points,
            collision_normals,
            anchor,
            parameters,
        )
        candidate_results[name] = result
        candidate_positions[name] = _propagate_to_full_mesh(full_rest, full_base_a, main_indices, main_rest, result.displacement)
        candidate_reports[name] = _candidate_report(
            name,
            parameters,
            result,
            main_rest,
            main_base_a,
            main_edges,
            collision_points,
            collision_normals,
            EXPECTED_SHIRT_VERTICES,
            EXPECTED_SHIRT_FACES,
        )

    source_after_solver = source.stat()
    source_hash_after_solver = _sha256(source)
    source_unchanged = (
        source_hash_after_solver == source_before["sha256"]
        and source_after_solver.st_size == source_before["size"]
        and source_after_solver.st_mtime_ns == source_before["mtime_ns"]
    )
    if not source_unchanged:
        raise RuntimeError("Source Blend changed during static equilibrium optimization")
    valid_candidates = [name for name, report in candidate_reports.items() if report["valid_numeric"]]
    report_path = output_root / "static_equilibrium_report.json"
    if not valid_candidates:
        report_path.write_text(
            json.dumps(
                {
                    "source": {**source_before, "sha256_after": source_hash_after_solver, "size_after": source_after_solver.st_size, "mtime_ns_after": source_after_solver.st_mtime_ns, "unchanged": source_unchanged},
                    "original_total_vertex_count": EXPECTED_SHIRT_VERTICES,
                    "original_total_face_count": EXPECTED_SHIRT_FACES,
                    "main_component": {"vertices": len(main_indices), "faces": len(main_faces)},
                    "candidates": candidate_reports,
                    "previews": [],
                    "review_blend": None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(f"All static-equilibrium candidates failed the numeric safety gate; see {report_path}")

    for obj in bpy.data.objects:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.hide_set(True)
    collections = {name: _new_collection(name) for name in COLLECTION_NAMES}
    body_a = _static_duplicate(body, _world_mesh_data(body, depsgraph)[0], collections["RYUON_A"], "ComeBody_A_Static")
    _static_duplicate(shirt, full_base_a, collections["A_LBS_BASELINE"], "Taisofuku_Shirt_A_LBS_BASELINE")
    for name in PARAMETERS:
        collection_name = f"A_OPT_{name.upper()}"
        _static_duplicate(shirt, candidate_positions[name], collections[collection_name], f"Taisofuku_Shirt_A_OPT_{name.upper()}")
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_set(1)
    scene["cloth2bones_static_equilibrium"] = True
    scene["dem_bones_executed"] = False
    scene["weight_generation_executed"] = False
    scene["source_blend_sha256"] = source_before["sha256"]
    _set_collection_visibility(collections, "A_OPT_BALANCED")
    preview_paths = _render_previews(collections, body_a, output_root / "previews")
    _set_collection_visibility(collections, "A_OPT_BALANCED")
    review_blend = output_root / "Taisofuku_StaticTeacher_Review.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(review_blend))
    if not review_blend.is_file() or review_blend.stat().st_size <= 0:
        raise RuntimeError(f"Review Blend was not created: {review_blend}")
    source_after = source.stat()
    source_hash_after = _sha256(source)
    source_unchanged = (
        source_hash_after == source_before["sha256"]
        and source_after.st_size == source_before["size"]
        and source_after.st_mtime_ns == source_before["mtime_ns"]
    )
    report = {
        "source": {
            **source_before,
            "sha256_after": source_hash_after,
            "size_after": source_after.st_size,
            "mtime_ns_after": source_after.st_mtime_ns,
            "unchanged": source_unchanged,
        },
        "original_total_vertex_count": EXPECTED_SHIRT_VERTICES,
        "original_total_face_count": EXPECTED_SHIRT_FACES,
        "main_component": {"vertices": len(main_indices), "faces": len(main_faces), "unique_edges": len(main_edges)},
        "candidates": candidate_reports,
        "review_blend": str(review_blend),
        "previews": preview_paths,
        "collections": list(COLLECTION_NAMES),
        "default_visible_collection": "A_OPT_BALANCED",
        "dem_bones_executed": False,
        "bone_generation_executed": False,
        "weight_generation_executed": False,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not source_unchanged:
        raise RuntimeError("Source Blend changed during static review generation")
    if len(preview_paths) != 16:
        raise RuntimeError(f"Expected 16 previews, got {len(preview_paths)}")
    print(json.dumps({"report": str(report_path), "review_blend": str(review_blend), "preview_count": len(preview_paths), "source_unchanged": source_unchanged, "valid_candidates": valid_candidates}, indent=2))


if __name__ == "__main__":
    main()
