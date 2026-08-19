"""Build the Taisofuku static-equilibrium visual review Blend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloth2bones.pbd_settle import PBDSettleParameters, PBDSettleResult, settle_pbd  # noqa: E402
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
CLEARANCE = 0.0015
COLLISION_QUERY_MAX_DISTANCE = 0.08
MAIN_OBJECT = "Taisofuku_Shirt"
BODY_OBJECT = "ComeBody"
ARMATURE_OBJECT = "ComeBody_Armature"
REQUIRED_GROUPS = ("Spine", "Chest", "Shoulder.L", "Shoulder.R", "Upper_arm.L", "Upper_arm.R")
PARAMETERS = {
    "stiff": PBDSettleParameters(90, 1.0 / 60.0, 6, 3, 8, -9.81, 0.92, 0.90, 0.30, 0.16),
    "balanced": PBDSettleParameters(90, 1.0 / 60.0, 6, 3, 8, -9.81, 0.92, 0.90, 0.12, 0.10),
    "soft": PBDSettleParameters(90, 1.0 / 60.0, 6, 3, 8, -9.81, 0.92, 0.90, 0.04, 0.06),
}
BODY_COLLECTION = "00_RYUON_A_BODY"
DEFAULT_VISIBLE_COLLECTION = "03_A_REVIEW_BALANCED"
COLLECTION_NAMES = (BODY_COLLECTION, "01_A_BASELINE_NORMAL", "02_A_REVIEW_STIFF", DEFAULT_VISIBLE_COLLECTION, "04_A_REVIEW_SOFT")
DISPLAY_COLLECTIONS = {
    "baseline": "01_A_BASELINE_NORMAL",
    "stiff": "02_A_REVIEW_STIFF",
    "balanced": DEFAULT_VISIBLE_COLLECTION,
    "soft": "04_A_REVIEW_SOFT",
}
DISPLAY_OBJECTS = {
    "baseline": "Taisofuku_Shirt_A_Baseline_Normal",
    "stiff": "Taisofuku_Shirt_A_Review_Stiff",
    "balanced": "Taisofuku_Shirt_A_Review_Balanced",
    "soft": "Taisofuku_Shirt_A_Review_Soft",
}


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


def _pose_directions(armature: bpy.types.Object) -> dict[str, np.ndarray]:
    directions = {}
    for bone_name in ("Upper_arm.L", "Upper_arm.R"):
        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None:
            raise RuntimeError(f"ComeBody_Armature is missing pose bone {bone_name!r}")
        direction = pose_bone.tail - pose_bone.head
        if direction.length <= 1.0e-12:
            raise RuntimeError(f"Pose bone {bone_name!r} has zero-length direction")
        directions[bone_name] = np.asarray([float(value) for value in direction.normalized()], dtype=np.float64)
    return directions


def _pose_direction_angles(directions: dict[str, np.ndarray]) -> dict[str, float]:
    down = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    angles = {}
    for bone_name, direction in directions.items():
        dot = max(-1.0, min(1.0, float(np.dot(direction, down))))
        angles[bone_name] = math.degrees(math.acos(dot))
    return angles


def _verify_builder_pose(
    full_body_t: np.ndarray,
    full_body_a: np.ndarray,
    pose_directions: dict[str, np.ndarray],
) -> dict[str, object]:
    angles = _pose_direction_angles(pose_directions)
    body_pose_delta = np.linalg.norm(full_body_a - full_body_t, axis=1)
    nan_count = int(np.isnan(body_pose_delta).sum())
    inf_count = int(np.isinf(body_pose_delta).sum())
    pose_verified = all(11.5 <= angles[name] <= 12.5 for name in ("Upper_arm.L", "Upper_arm.R"))
    body_changed = (
        nan_count == 0
        and inf_count == 0
        and float(np.max(body_pose_delta)) > 0.05
    )
    if not pose_verified:
        raise RuntimeError(f"Upper arm pose angle is outside the requested A-pose range: {angles}")
    if not body_changed:
        raise RuntimeError("Body evaluated geometry did not change to the requested A pose")
    return {
        "upper_arm_left_angle_to_down_deg": angles["Upper_arm.L"],
        "upper_arm_right_angle_to_down_deg": angles["Upper_arm.R"],
        "body_t_to_a_rms_m": float(np.sqrt(np.mean(body_pose_delta * body_pose_delta))),
        "body_t_to_a_max_m": float(np.max(body_pose_delta)),
        "builder_pose_verified": True,
    }


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


def _bend_pairs(mesh: bpy.types.Mesh, main_faces: list[int], global_to_local: np.ndarray) -> np.ndarray:
    edge_opposites: dict[tuple[int, int], list[int]] = {}
    for face_index in main_faces:
        vertices = tuple(int(index) for index in mesh.polygons[face_index].vertices)
        if len(vertices) != 3:
            continue
        for left, right, opposite in ((vertices[0], vertices[1], vertices[2]), (vertices[1], vertices[2], vertices[0]), (vertices[2], vertices[0], vertices[1])):
            edge = (left, right) if left < right else (right, left)
            edge_opposites.setdefault(edge, []).append(opposite)
    pairs: set[tuple[int, int]] = set()
    for opposites in edge_opposites.values():
        if len(opposites) != 2 or opposites[0] == opposites[1]:
            continue
        left = int(global_to_local[opposites[0]])
        right = int(global_to_local[opposites[1]])
        if left < 0 or right < 0 or left == right:
            continue
        pairs.add((left, right) if left < right else (right, left))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape((-1, 2)) if pairs else np.empty((0, 2), dtype=np.int64)


def _world_mesh_topology(obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        vertices = np.asarray(
            [[float(value) for value in evaluated.matrix_world @ vertex.co] for vertex in mesh.vertices],
            dtype=np.float64,
        )
        polygons = [tuple(int(index) for index in polygon.vertices) for polygon in mesh.polygons]
        return vertices, polygons
    finally:
        evaluated.to_mesh_clear()


def _build_body_bvh(body: bpy.types.Object, depsgraph: bpy.types.Depsgraph) -> BVHTree:
    vertices, polygons = _world_mesh_topology(body, depsgraph)
    if not len(polygons):
        raise RuntimeError("ComeBody evaluated mesh has no polygons for collision BVH")
    return BVHTree.FromPolygons([Vector(vertex) for vertex in vertices], polygons, all_triangles=False)


def _body_collision_projector(body_bvh: BVHTree):
    def project(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        projected = np.asarray(positions, dtype=np.float64).copy()
        corrected = np.zeros(len(projected), dtype=bool)
        for _ in range(3):
            pass_corrected = np.zeros(len(projected), dtype=bool)
            for index, position in enumerate(projected):
                location, normal, _, _ = body_bvh.find_nearest(Vector(position), COLLISION_QUERY_MAX_DISTANCE)
                if location is None or normal is None:
                    continue
                normal_vector = Vector(normal)
                if normal_vector.length <= 1.0e-12:
                    raise RuntimeError("Body collision BVH returned a zero-length normal")
                normal_vector.normalize()
                signed = float((Vector(position) - location).dot(normal_vector))
                if signed < CLEARANCE:
                    projected[index] = np.asarray(location + normal_vector * CLEARANCE, dtype=np.float64)
                    pass_corrected[index] = True
            corrected |= pass_corrected
            if not np.any(pass_corrected):
                break
        return projected, corrected

    return project


def _collision_metrics(body_bvh: BVHTree, positions: np.ndarray) -> dict[str, float | int]:
    signed_values = []
    no_hit_count = 0
    for position in positions:
        location, normal, _, _ = body_bvh.find_nearest(Vector(position), COLLISION_QUERY_MAX_DISTANCE)
        if location is None or normal is None:
            no_hit_count += 1
            continue
        normal_vector = Vector(normal)
        if normal_vector.length <= 1.0e-12:
            raise RuntimeError("Body collision BVH returned a zero-length normal")
        normal_vector.normalize()
        signed_values.append(float((Vector(position) - location).dot(normal_vector)))
    values = np.asarray(signed_values, dtype=np.float64)
    if len(values):
        signed_min = float(np.min(values))
        signed_p01 = float(np.percentile(values, 1.0))
        signed_mean = float(np.mean(values))
    else:
        signed_min = None
        signed_p01 = None
        signed_mean = None
    return {
        "queried_count": len(positions),
        "no_hit_count": no_hit_count,
        "signed_min_m": signed_min,
        "signed_p01_m": signed_p01,
        "signed_mean_m": signed_mean,
        "count_signed_lt_0": int(np.count_nonzero(values < 0.0)),
        "count_signed_lt_minus_0_0005": int(np.count_nonzero(values < -0.0005)),
        "count_signed_lt_minus_0_001": int(np.count_nonzero(values < -0.001)),
        "count_signed_lt_minus_0_002": int(np.count_nonzero(values < -0.002)),
    }


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


def _edge_metrics(rest: np.ndarray, positions: np.ndarray, edges: np.ndarray) -> dict[str, float | int]:
    rest_lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    current_lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    included = rest_lengths >= 0.0005
    if not np.any(included):
        return {
            "included_count": 0,
            "rms": 0.0,
            "p95_absolute": 0.0,
            "max_absolute": 0.0,
            "length_error_rms_m": 0.0,
            "length_error_p95_m": 0.0,
            "length_error_max_m": 0.0,
        }
    rest_lengths = rest_lengths[included]
    current_lengths = current_lengths[included]
    length_error = current_lengths - rest_lengths
    strain = length_error / rest_lengths
    absolute = np.abs(strain)
    absolute_length_error = np.abs(length_error)
    return {
        "included_count": int(np.count_nonzero(included)),
        "rms": float(np.sqrt(np.mean(strain * strain))),
        "p95_absolute": float(np.percentile(absolute, 95.0)),
        "max_absolute": float(np.max(absolute)),
        "length_error_rms_m": float(np.sqrt(np.mean(length_error * length_error))),
        "length_error_p95_m": float(np.percentile(absolute_length_error, 95.0)),
        "length_error_max_m": float(np.max(absolute_length_error)),
    }


def _candidate_report(
    name: str,
    parameters: PBDSettleParameters,
    result: PBDSettleResult,
    main_rest: np.ndarray,
    main_base_a: np.ndarray,
    main_edges: np.ndarray,
    bend_pairs: np.ndarray,
    hard_pin_mask: np.ndarray,
    collision_metrics: dict[str, object],
    collision_eligibility_metrics: dict[str, object],
    final_full_positions: np.ndarray,
    output_vertices: int,
    output_faces: int,
) -> dict[str, object]:
    correction_magnitude = np.linalg.norm(result.displacement, axis=1)
    nan_count = int(np.isnan(final_full_positions).sum() + np.isnan(result.displacement).sum() + np.isnan(result.velocity).sum())
    inf_count = int(np.isinf(final_full_positions).sum() + np.isinf(result.displacement).sum() + np.isinf(result.velocity).sum())
    edge_strain = _edge_metrics(main_rest, result.positions, main_edges)
    topology_preserved = output_vertices == EXPECTED_SHIRT_VERTICES and output_faces == EXPECTED_SHIRT_FACES
    hard_pin_error = float(np.max(np.linalg.norm(result.positions[hard_pin_mask] - main_base_a[hard_pin_mask], axis=1))) if np.any(hard_pin_mask) else 0.0
    review_export_eligible = (
        nan_count == 0
        and inf_count == 0
        and topology_preserved
        and hard_pin_error <= 1.0e-8
        and collision_eligibility_metrics["count_signed_lt_minus_0_001"] == 0
    )
    valid_numeric = (
        review_export_eligible
        and float(np.max(correction_magnitude)) <= 0.10
        and edge_strain["max_absolute"] <= 0.20
    )
    numeric_warnings = []
    if float(np.max(correction_magnitude)) > 0.10:
        numeric_warnings.append("max_displacement_exceeds_0.10_m")
    if edge_strain["max_absolute"] > 0.20:
        numeric_warnings.append("max_absolute_edge_strain_exceeds_0.20")
    return {
        "name": name,
        "display_collection": DISPLAY_COLLECTIONS[name],
        "display_object": DISPLAY_OBJECTS[name],
        "parameter_set": {
            "frames": parameters.frames,
            "dt": parameters.dt,
            "solver_iterations": parameters.solver_iterations,
            "collision_interval": parameters.collision_interval,
            "final_projection_iterations": parameters.final_projection_iterations,
            "gravity": parameters.gravity,
            "damping": parameters.damping,
            "stretch_stiffness": parameters.stretch_stiffness,
            "bend_stiffness": parameters.bend_stiffness,
            "attachment_stiffness": parameters.attachment_stiffness,
            "collision_clearance": CLEARANCE,
            "collision_query_max_distance": COLLISION_QUERY_MAX_DISTANCE,
        },
        "frames": result.frames,
        "collision_projection_count": result.collision_projection_count,
        "correction_displacement": {
            "mean": float(np.mean(correction_magnitude)),
            "p50": float(np.percentile(correction_magnitude, 50.0)),
            "p95": float(np.percentile(correction_magnitude, 95.0)),
            "max": float(np.max(correction_magnitude)),
        },
        "edge_strain": edge_strain,
        "collision": collision_metrics,
        "collision_eligibility": collision_eligibility_metrics,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "output_vertices": output_vertices,
        "output_faces": output_faces,
        "topology_preserved": topology_preserved,
        "hard_pin_count": int(np.count_nonzero(hard_pin_mask)),
        "hard_pin_max_error_m": hard_pin_error,
        "stretch_edge_count": len(main_edges),
        "bend_pair_count": len(bend_pairs),
        "valid_numeric": valid_numeric,
        "review_export_eligible": review_export_eligible,
        "numeric_warnings": numeric_warnings,
        "numeric_gate": {
            "max_displacement_m": 0.10,
            "max_absolute_edge_strain": 0.20,
        },
    }


def _new_collection(name: str) -> bpy.types.Collection:
    if bpy.data.collections.get(name) is not None:
        raise RuntimeError(f"Source already contains output collection {name!r}")
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def _mesh_without_shape_keys(source_mesh: bpy.types.Mesh, name: str) -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(
        [vertex.co.copy() for vertex in source_mesh.vertices],
        [tuple(edge.vertices) for edge in source_mesh.edges],
        [tuple(polygon.vertices) for polygon in source_mesh.polygons],
    )
    for material in source_mesh.materials:
        mesh.materials.append(material)
    for destination, source_polygon in zip(mesh.polygons, source_mesh.polygons, strict=True):
        destination.material_index = source_polygon.material_index
        destination.use_smooth = source_polygon.use_smooth
    mesh.update()
    return mesh


def _static_duplicate(
    source: bpy.types.Object,
    world_positions: np.ndarray,
    collection: bpy.types.Collection,
    name: str,
) -> bpy.types.Object:
    if len(world_positions) != len(source.data.vertices):
        raise RuntimeError(f"Static duplicate vertex mismatch for {name}: {len(world_positions)} != {len(source.data.vertices)}")
    result = source.copy()
    result.data = _mesh_without_shape_keys(source.data, name)
    result.name = name
    result.parent = None
    result.matrix_parent_inverse = Matrix.Identity(4)
    result.matrix_world = Matrix.Identity(4)
    result.animation_data_clear()
    for modifier in list(result.modifiers):
        result.modifiers.remove(modifier)
    collection.objects.link(result)
    result.hide_viewport = False
    result.hide_render = False
    result.hide_set(False)
    for index, point in enumerate(world_positions):
        result.data.vertices[index].co = Vector(point)
    result.data.update()
    return result


def _set_layer_collection_visibility(layer_collection: bpy.types.LayerCollection) -> None:
    for child in layer_collection.children:
        child.exclude = False
        child.hide_viewport = False
        _set_layer_collection_visibility(child)


def _set_review_visibility() -> None:
    for name in COLLECTION_NAMES:
        collection = bpy.data.collections[name]
        collection.hide_viewport = False
        collection.hide_render = False
    _set_layer_collection_visibility(bpy.context.view_layer.layer_collection)
    visible_objects = {
        "Ryuon_A_Body_Static",
        DISPLAY_OBJECTS["balanced"],
    }
    for name in ("Ryuon_A_Body_Static", *DISPLAY_OBJECTS.values()):
        obj = bpy.data.objects[name]
        obj.hide_viewport = False
        obj.hide_render = False
        obj.hide_set(name not in visible_objects)


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
    bpy.context.view_layer.update()
    full_body_t, _ = _world_mesh_data(body, depsgraph)
    full_rest, _ = _world_mesh_data(shirt, depsgraph)
    _set_motion_pose(armature, rest_matrices, {"Upper_arm.L", "Upper_arm.R"}, 1.0)
    bpy.context.view_layer.update()
    full_body_a, _ = _world_mesh_data(body, depsgraph)
    full_base_a, _ = _world_mesh_data(shirt, depsgraph)
    if len(full_body_t) != EXPECTED_BODY_VERTICES or len(full_body_a) != EXPECTED_BODY_VERTICES:
        raise RuntimeError("Body evaluated vertex count changed under the Armature modifier")
    pose_directions = _pose_directions(armature)
    pose_verification = _verify_builder_pose(full_body_t, full_body_a, pose_directions)
    main_rest = full_rest[main_indices]
    main_base_a = full_base_a[main_indices]
    if len(full_rest) != EXPECTED_SHIRT_VERTICES or len(full_base_a) != EXPECTED_SHIRT_VERTICES:
        raise RuntimeError("Evaluated Taisofuku_Shirt vertex count changed under the Armature modifier")
    global_to_local = np.full(len(full_rest), -1, dtype=np.int64)
    global_to_local[main_indices] = np.arange(len(main_indices), dtype=np.int64)
    main_edges_local = global_to_local[main_edges]
    if np.any(main_edges_local < 0):
        raise RuntimeError("Main component edge remapping produced an invalid local index")
    bend_pairs = _bend_pairs(shirt.data, main_faces, global_to_local)
    anchor = _anchor_weights(shirt, main_indices, main_rest)
    attachment = np.clip((anchor - 0.15) / 0.85, 0.0, 1.0)
    hard_pin_mask = anchor >= 0.999
    body_bvh = _build_body_bvh(body, depsgraph)
    collision_projector = _body_collision_projector(body_bvh)
    baseline_collision = _collision_metrics(body_bvh, main_base_a)

    candidate_reports: dict[str, dict[str, object]] = {}
    candidate_positions: dict[str, np.ndarray] = {}
    for name, parameters in PARAMETERS.items():
        result = settle_pbd(
            main_rest,
            main_base_a,
            main_edges_local,
            bend_pairs,
            attachment,
            hard_pin_mask,
            parameters,
            collision_projector,
        )
        candidate_positions[name] = _propagate_to_full_mesh(full_rest, full_base_a, main_indices, main_rest, result.displacement)
        candidate_reports[name] = _candidate_report(
            name,
            parameters,
            result,
            main_rest,
            main_base_a,
            main_edges_local,
            bend_pairs,
            hard_pin_mask,
            _collision_metrics(body_bvh, result.positions),
            _collision_metrics(body_bvh, result.positions[~hard_pin_mask]),
            candidate_positions[name],
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
        raise RuntimeError("Source Blend changed during PBD settling")
    review_export_candidates = [name for name, report in candidate_reports.items() if report["review_export_eligible"]]
    report_path = output_root / "static_equilibrium_report.json"
    if not review_export_candidates:
        report_path.write_text(
            json.dumps(
                {
                    "source": {**source_before, "sha256_after": source_hash_after_solver, "size_after": source_after_solver.st_size, "mtime_ns_after": source_after_solver.st_mtime_ns, "unchanged": source_unchanged},
                    "original_total_vertex_count": EXPECTED_SHIRT_VERTICES,
                    "original_total_face_count": EXPECTED_SHIRT_FACES,
                    "main_component": {"vertices": len(main_indices), "faces": len(main_faces), "stretch_edge_count": len(main_edges_local), "bend_pair_count": len(bend_pairs), "hard_pin_count": int(np.count_nonzero(hard_pin_mask))},
                    "baseline_collision": baseline_collision,
                    "pose_verification": pose_verification,
                    "candidates": candidate_reports,
                    "collections": list(COLLECTION_NAMES),
                    "default_visible_collection": DEFAULT_VISIBLE_COLLECTION,
                    "review_exported": False,
                    "review_export_reason": "no_candidate_met_review_export_eligibility",
                    "review_blend": None,
                    "previews": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(f"All static-equilibrium candidates failed review export eligibility; see {report_path}")

    for obj in bpy.data.objects:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.hide_set(True)
    collections = {name: _new_collection(name) for name in COLLECTION_NAMES}
    _static_duplicate(body, full_body_a, collections[BODY_COLLECTION], "Ryuon_A_Body_Static")
    _static_duplicate(shirt, full_base_a, collections[DISPLAY_COLLECTIONS["baseline"]], DISPLAY_OBJECTS["baseline"])
    for name in PARAMETERS:
        _static_duplicate(shirt, candidate_positions[name], collections[DISPLAY_COLLECTIONS[name]], DISPLAY_OBJECTS[name])
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_set(1)
    scene["cloth2bones_static_equilibrium"] = True
    scene["review_only_visual_checkpoint"] = True
    scene["dem_bones_executed"] = False
    scene["bone_generation_executed"] = False
    scene["weight_generation_executed"] = False
    scene["default_visible_collection"] = DEFAULT_VISIBLE_COLLECTION
    scene["source_blend_sha256"] = source_before["sha256"]
    _set_review_visibility()
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
        "main_component": {"vertices": len(main_indices), "faces": len(main_faces), "stretch_edge_count": len(main_edges_local), "bend_pair_count": len(bend_pairs), "hard_pin_count": int(np.count_nonzero(hard_pin_mask))},
        "baseline_collision": baseline_collision,
        "pose_verification": pose_verification,
        "candidates": candidate_reports,
        "review_exported": True,
        "review_export_reason": "generated_for_visual_review_even_if_numeric_gate_failed",
        "review_blend": str(review_blend),
        "previews": [],
        "collections": list(COLLECTION_NAMES),
        "default_visible_collection": DEFAULT_VISIBLE_COLLECTION,
        "dem_bones_executed": False,
        "bone_generation_executed": False,
        "weight_generation_executed": False,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not source_unchanged:
        raise RuntimeError("Source Blend changed during static review generation")
    print(json.dumps({"report": str(report_path), "review_blend": str(review_blend), "preview_count": 0, "source_unchanged": source_unchanged, "review_export_candidates": review_export_candidates}, indent=2))


if __name__ == "__main__":
    main()
