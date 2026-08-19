"""Verify a static-equilibrium review Blend in a separate Blender process."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.export_ryuon_md_motion import _rest_matrices, _set_motion_pose, _set_rest_pose  # noqa: E402

SOURCE_SHA256 = "00eace7d79ed201dd0c5684c3301f9af6c5653bff05c215e4d8ed5c6433801df"
EXPECTED_VERTICES = 121471
EXPECTED_FACES = 115103
EXPECTED_MAIN_VERTICES = 87005
EXPECTED_MAIN_FACES = 86865
COLLISION_CLEARANCE = 0.0015
COLLISION_QUERY_MAX_DISTANCE = 0.08
BODY_COLLECTION = "00_RYUON_A_BODY"
DEFAULT_VISIBLE_COLLECTION = "03_A_REVIEW_BALANCED"
COLLECTIONS = (BODY_COLLECTION, "01_A_BASELINE_NORMAL", "02_A_REVIEW_STIFF", DEFAULT_VISIBLE_COLLECTION, "04_A_REVIEW_SOFT")
CANDIDATE_OBJECTS = {
    "01_A_BASELINE_NORMAL": "Taisofuku_Shirt_A_Baseline_Normal",
    "02_A_REVIEW_STIFF": "Taisofuku_Shirt_A_Review_Stiff",
    DEFAULT_VISIBLE_COLLECTION: "Taisofuku_Shirt_A_Review_Balanced",
    "04_A_REVIEW_SOFT": "Taisofuku_Shirt_A_Review_Soft",
}
CANDIDATE_RECORDS = {
    "02_A_REVIEW_STIFF": "stiff",
    DEFAULT_VISIBLE_COLLECTION: "balanced",
    "04_A_REVIEW_SOFT": "soft",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_mesh(obj: bpy.types.Object) -> bool:
    return all(math.isfinite(float(value)) for vertex in obj.data.vertices for value in vertex.co)


def _world_mesh_data(obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph) -> np.ndarray:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return np.asarray(
            [[float(value) for value in evaluated.matrix_world @ vertex.co] for vertex in mesh.vertices],
            dtype=np.float64,
        )
    finally:
        evaluated.to_mesh_clear()


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


def _component_data(mesh: bpy.types.Mesh) -> tuple[np.ndarray, np.ndarray, list[int]]:
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
    roots = [find(index) for index in range(face_count)]
    counts: dict[int, int] = {}
    for root in roots:
        counts[root] = counts.get(root, 0) + 1
    main_root = max(counts, key=counts.get)
    main_faces = [index for index, vertices in enumerate(face_vertices) if vertices and roots[index] == main_root]
    main_vertices = sorted({vertex for face_index in main_faces for vertex in face_vertices[face_index]})
    main_vertex_set = set(main_vertices)
    main_edges = sorted(edge for edge in edge_owner if edge[0] in main_vertex_set and edge[1] in main_vertex_set)
    return np.asarray(main_edges, dtype=np.int64), np.asarray(main_vertices, dtype=np.int64), main_faces


def _build_bvh(vertices: np.ndarray, polygons: list[tuple[int, ...]]) -> BVHTree:
    return BVHTree.FromPolygons([Vector(vertex) for vertex in vertices], polygons, all_triangles=False)


def _bend_pair_count(mesh: bpy.types.Mesh, main_faces: list[int], global_to_local: np.ndarray) -> int:
    edge_opposites: dict[tuple[int, int], list[int]] = {}
    for face_index in main_faces:
        vertices = tuple(int(index) for index in mesh.polygons[face_index].vertices)
        if len(vertices) != 3:
            continue
        for left, right, opposite in ((vertices[0], vertices[1], vertices[2]), (vertices[1], vertices[2], vertices[0]), (vertices[2], vertices[0], vertices[1])):
            edge = (left, right) if left < right else (right, left)
            edge_opposites.setdefault(edge, []).append(opposite)
    pairs = set()
    for opposites in edge_opposites.values():
        if len(opposites) != 2 or opposites[0] == opposites[1]:
            continue
        left = int(global_to_local[opposites[0]])
        right = int(global_to_local[opposites[1]])
        if left < 0 or right < 0 or left == right:
            continue
        pairs.add((left, right) if left < right else (right, left))
    return len(pairs)


def _collision_metrics(body_bvh: BVHTree, positions: np.ndarray) -> dict[str, object]:
    signed_values = []
    no_hit_count = 0
    for position in positions:
        location, normal, _, _ = body_bvh.find_nearest(Vector(position), COLLISION_QUERY_MAX_DISTANCE)
        if location is None or normal is None:
            no_hit_count += 1
            continue
        normal_vector = Vector(normal)
        if normal_vector.length <= 1.0e-12:
            raise RuntimeError("Review Body collision BVH returned a zero-length normal")
        normal_vector.normalize()
        signed_values.append(float((Vector(position) - location).dot(normal_vector)))
    values = np.asarray(signed_values, dtype=np.float64)
    return {
        "queried_count": len(positions),
        "no_hit_count": no_hit_count,
        "signed_min_m": float(np.min(values)) if len(values) else None,
        "signed_p01_m": float(np.percentile(values, 1.0)) if len(values) else None,
        "signed_mean_m": float(np.mean(values)) if len(values) else None,
        "count_signed_lt_0": int(np.count_nonzero(values < 0.0)),
        "count_signed_lt_minus_0_0005": int(np.count_nonzero(values < -0.0005)),
        "count_signed_lt_minus_0_001": int(np.count_nonzero(values < -0.001)),
        "count_signed_lt_minus_0_002": int(np.count_nonzero(values < -0.002)),
    }


def _assert_metrics_match(actual: dict[str, object], expected: dict[str, object], label: str) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise RuntimeError(f"{label} collision metric is missing {key!r}")
        actual_value = actual[key]
        if isinstance(expected_value, float) or isinstance(actual_value, float):
            if expected_value is None or actual_value is None or not math.isclose(float(actual_value), float(expected_value), rel_tol=1.0e-9, abs_tol=1.0e-9):
                raise RuntimeError(f"{label} collision metric {key!r} differs")
        elif actual_value != expected_value:
            raise RuntimeError(f"{label} collision metric {key!r} differs")


def _hard_pin_mask(shirt: bpy.types.Object, main_indices: np.ndarray, main_rest: np.ndarray) -> np.ndarray:
    required_groups = ("Spine", "Chest", "Shoulder.L", "Shoulder.R", "Upper_arm.L", "Upper_arm.R")
    group_indices = {}
    for name in required_groups:
        group = shirt.vertex_groups.get(name)
        if group is None:
            raise RuntimeError(f"Taisofuku_Shirt is missing required vertex group {name!r}")
        group_indices[group.index] = name
    weights = {name: np.zeros(len(shirt.data.vertices), dtype=np.float64) for name in required_groups}
    for vertex in shirt.data.vertices:
        for item in vertex.groups:
            name = group_indices.get(item.group)
            if name is not None:
                weights[name][vertex.index] = float(item.weight)
    torso = np.maximum(weights["Spine"], weights["Chest"])[main_indices]
    shoulder = np.maximum(weights["Shoulder.L"], weights["Shoulder.R"])[main_indices]
    arm = np.maximum(weights["Upper_arm.L"], weights["Upper_arm.R"])[main_indices]
    anchor = np.clip(np.maximum(torso - 0.5 * arm, 0.0) + 0.35 * shoulder, 0.0, 1.0)
    anchor = 0.15 + 0.85 * anchor
    minimum = main_rest.min(axis=0)
    maximum = main_rest.max(axis=0)
    shirt_width = float(maximum[0] - minimum[0])
    center_x = float((maximum[0] + minimum[0]) * 0.5)
    z90 = float(np.percentile(main_rest[:, 2], 90.0))
    collar = (np.abs(main_rest[:, 0] - center_x) <= shirt_width * 0.18) & (main_rest[:, 2] >= z90) & (torso >= arm)
    anchor[collar] = 1.0
    return anchor >= 0.999


def _layer_collections(layer_collection: bpy.types.LayerCollection) -> dict[str, bpy.types.LayerCollection]:
    result = {}
    for child in layer_collection.children:
        result[child.collection.name] = child
        result.update(_layer_collections(child))
    return result


def _pose_direction_angles(armature: bpy.types.Object) -> dict[str, float]:
    down = Vector((0.0, 0.0, -1.0))
    angles = {}
    for bone_name in ("Upper_arm.L", "Upper_arm.R"):
        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None:
            raise RuntimeError(f"ComeBody_Armature is missing pose bone {bone_name!r}")
        direction = pose_bone.tail - pose_bone.head
        if direction.length <= 1.0e-12:
            raise RuntimeError(f"Pose bone {bone_name!r} has zero-length direction")
        dot = max(-1.0, min(1.0, float(direction.normalized().dot(down))))
        angles[bone_name] = math.degrees(math.acos(dot))
    return angles


def _require_a_pose_angles(angles: dict[str, float]) -> None:
    if not all(11.5 <= angles[name] <= 12.5 for name in ("Upper_arm.L", "Upper_arm.R")):
        raise RuntimeError(f"Expected A-pose arm angles in [11.5, 12.5] degrees, got {angles}")


def main() -> None:
    args = _args()
    blend = args.blend.resolve()
    report_path = args.report.resolve()
    source = args.source.resolve()
    if not blend.is_file() or not report_path.is_file() or not source.is_file():
        raise FileNotFoundError(f"Missing verification input: blend={blend}, report={report_path}, source={source}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("review_exported") is not True:
        raise RuntimeError("Report does not mark the Review Blend as exported")
    if report.get("review_export_reason") != "generated_for_visual_review_even_if_numeric_gate_failed":
        raise RuntimeError("Report review_export_reason is inconsistent")
    if report.get("previews") != []:
        raise RuntimeError("Review-only report must contain an empty previews array")
    if report.get("collections") != list(COLLECTIONS):
        raise RuntimeError("Report collection list is inconsistent")
    if report.get("default_visible_collection") != DEFAULT_VISIBLE_COLLECTION:
        raise RuntimeError("Report default visible collection is inconsistent")
    if report.get("review_blend") and Path(report["review_blend"]).resolve() != blend:
        raise RuntimeError("Report review_blend path does not match the verified Blend")

    source_stat_before = source.stat()
    source_hash_before = _sha256(source)
    if source_hash_before != SOURCE_SHA256:
        raise RuntimeError(f"Source SHA-256 mismatch: expected {SOURCE_SHA256}, got {source_hash_before}")
    bpy.ops.wm.open_mainfile(filepath=str(source))
    armature = bpy.data.objects.get("ComeBody_Armature")
    source_body = bpy.data.objects.get("ComeBody")
    source_shirt = bpy.data.objects.get("Taisofuku_Shirt")
    if not all((armature, source_body, source_shirt)) or armature.type != "ARMATURE" or source_body.type != "MESH" or source_shirt.type != "MESH":
        raise RuntimeError("Source Blend is missing ComeBody_Armature, ComeBody, or Taisofuku_Shirt")
    rest_matrices = _rest_matrices(armature)
    source_depsgraph = bpy.context.evaluated_depsgraph_get()
    _set_rest_pose(armature, rest_matrices)
    bpy.context.view_layer.update()
    expected_body_t = _world_mesh_data(source_body, source_depsgraph)
    expected_shirt_t = _world_mesh_data(source_shirt, source_depsgraph)
    _set_motion_pose(armature, rest_matrices, {"Upper_arm.L", "Upper_arm.R"}, 1.0)
    bpy.context.view_layer.update()
    expected_body_a = _world_mesh_data(source_body, source_depsgraph)
    expected_shirt_a = _world_mesh_data(source_shirt, source_depsgraph)
    source_pose_angles = _pose_direction_angles(armature)
    _require_a_pose_angles(source_pose_angles)
    source_edges, main_indices, main_faces = _component_data(source_shirt.data)

    if len(expected_body_t) != len(expected_body_a) or len(expected_body_t) != len(source_body.data.vertices):
        raise RuntimeError("Source evaluated body topology is inconsistent")
    if len(expected_shirt_t) != len(expected_shirt_a) or len(expected_shirt_t) != len(source_shirt.data.vertices):
        raise RuntimeError("Source evaluated shirt topology is inconsistent")
    if len(main_indices) != EXPECTED_MAIN_VERTICES or len(main_faces) != EXPECTED_MAIN_FACES:
        raise RuntimeError("Source main component topology is inconsistent")
    global_to_local = np.full(len(source_shirt.data.vertices), -1, dtype=np.int64)
    global_to_local[main_indices] = np.arange(len(main_indices), dtype=np.int64)
    source_bend_pair_count = _bend_pair_count(source_shirt.data, main_faces, global_to_local)
    source_hard_pin_mask = _hard_pin_mask(source_shirt, main_indices, expected_shirt_t[main_indices])
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    missing_collections = [name for name in COLLECTIONS if bpy.data.collections.get(name) is None]
    if missing_collections:
        raise RuntimeError(f"Review Blend is missing collections: {missing_collections}")
    report_pose = report.get("pose_verification", {})
    if report_pose.get("builder_pose_verified") is not True:
        raise RuntimeError("Report does not contain a verified builder pose")

    original_shirt = bpy.data.objects.get("Taisofuku_Shirt")
    if original_shirt is None or original_shirt.type != "MESH":
        raise RuntimeError("Review Blend does not retain the original Taisofuku_Shirt object")
    original_vertices = len(original_shirt.data.vertices)
    original_faces = len(original_shirt.data.polygons)
    if original_vertices != EXPECTED_VERTICES or original_faces != EXPECTED_FACES:
        raise RuntimeError("Original Taisofuku_Shirt topology is inconsistent")

    body_collection = bpy.data.collections[BODY_COLLECTION]
    body_meshes = [obj for obj in body_collection.objects if obj.type == "MESH"]
    if len(body_meshes) != 1 or body_meshes[0].name != "Ryuon_A_Body_Static":
        raise RuntimeError("Body collection does not contain Ryuon_A_Body_Static as its only mesh")
    baseline_collection = bpy.data.collections["01_A_BASELINE_NORMAL"]
    baseline_meshes = [obj for obj in baseline_collection.objects if obj.type == "MESH"]
    if len(baseline_meshes) != 1 or baseline_meshes[0].name != "Taisofuku_Shirt_A_Baseline_Normal":
        raise RuntimeError("Baseline collection does not contain Taisofuku_Shirt_A_Baseline_Normal as its only mesh")
    review_depsgraph = bpy.context.evaluated_depsgraph_get()
    review_body, review_body_polygons = _world_mesh_topology(body_meshes[0], review_depsgraph)
    review_body_bvh = _build_bvh(review_body, review_body_polygons)
    review_baseline = _world_mesh_data(baseline_meshes[0], review_depsgraph)
    if review_body.shape != expected_body_a.shape or review_baseline.shape != expected_shirt_a.shape:
        raise RuntimeError("Review Body or baseline evaluated topology does not match source A pose")
    body_error_to_a = np.linalg.norm(review_body - expected_body_a, axis=1)
    body_error_to_t = np.linalg.norm(review_body - expected_body_t, axis=1)
    baseline_error_to_a = np.linalg.norm(review_baseline - expected_shirt_a, axis=1)
    body_review_to_a_rms = float(np.sqrt(np.mean(body_error_to_a * body_error_to_a)))
    body_review_to_a_max = float(np.max(body_error_to_a))
    body_review_to_t_rms = float(np.sqrt(np.mean(body_error_to_t * body_error_to_t)))
    baseline_review_to_a_rms = float(np.sqrt(np.mean(baseline_error_to_a * baseline_error_to_a)))
    baseline_review_to_a_max = float(np.max(baseline_error_to_a))
    if body_review_to_a_rms > 5.0e-5 or body_review_to_a_max > 2.0e-4:
        raise RuntimeError("Review Body does not match the expected source A-pose body")
    if body_review_to_t_rms <= 0.001 or body_review_to_a_rms >= body_review_to_t_rms * 0.05:
        raise RuntimeError("Review Body is not demonstrably closer to A pose than T pose")
    if baseline_review_to_a_rms > 5.0e-5 or baseline_review_to_a_max > 2.0e-4:
        raise RuntimeError("Baseline shirt does not match the expected source A-pose shirt")
    pose_match = {
        "upper_arm_left_angle_to_down_deg": source_pose_angles["Upper_arm.L"],
        "upper_arm_right_angle_to_down_deg": source_pose_angles["Upper_arm.R"],
        "body_review_to_a_rms_m": body_review_to_a_rms,
        "body_review_to_a_max_m": body_review_to_a_max,
        "body_review_to_t_rms_m": body_review_to_t_rms,
        "baseline_shirt_review_to_a_rms_m": baseline_review_to_a_rms,
        "baseline_shirt_review_to_a_max_m": baseline_review_to_a_max,
        "verified": True,
    }

    candidate_records = report.get("candidates", {})
    for collection_name, expected_object_name in CANDIDATE_OBJECTS.items():
        collection = bpy.data.collections[collection_name]
        mesh_objects = [obj for obj in collection.objects if obj.type == "MESH"]
        if len(mesh_objects) != 1:
            raise RuntimeError(f"Collection {collection_name} must contain one mesh, got {len(mesh_objects)}")
        candidate = mesh_objects[0]
        if candidate.name != expected_object_name:
            raise RuntimeError(f"Collection {collection_name} contains {candidate.name!r}, expected {expected_object_name!r}")
        if len(candidate.data.vertices) != original_vertices or len(candidate.data.polygons) != original_faces:
            raise RuntimeError(f"{candidate.name} topology does not match original Taisofuku_Shirt")
        if candidate.modifiers:
            raise RuntimeError(f"{candidate.name} still has modifiers")
        if not _finite_mesh(candidate):
            raise RuntimeError(f"{candidate.name} contains NaN or Inf vertex coordinates")
        candidate_positions = _world_mesh_data(candidate, review_depsgraph)
        if candidate_positions.shape != expected_shirt_a.shape:
            raise RuntimeError(f"{candidate.name} world topology does not match original Taisofuku_Shirt")
        candidate_collision = _collision_metrics(review_body_bvh, candidate_positions[main_indices])
        candidate_collision_eligibility = _collision_metrics(review_body_bvh, candidate_positions[main_indices][~source_hard_pin_mask])
        report_name = CANDIDATE_RECORDS.get(collection_name)
        if report_name is None:
            _assert_metrics_match(candidate_collision, report.get("baseline_collision", {}), "baseline")
            continue
        record = candidate_records.get(report_name)
        if not record or not record.get("topology_preserved"):
            raise RuntimeError(f"Report does not validate candidate {report_name}")
        if record.get("display_collection") != collection_name or record.get("display_object") != expected_object_name:
            raise RuntimeError(f"Report display mapping is inconsistent for {report_name}")
        if record.get("output_vertices") != original_vertices or record.get("output_faces") != original_faces:
            raise RuntimeError(f"Report topology counts are inconsistent for {report_name}")
        if record.get("stretch_edge_count") != len(source_edges) or record.get("bend_pair_count") != source_bend_pair_count or record.get("hard_pin_count") != int(np.count_nonzero(source_hard_pin_mask)):
            raise RuntimeError(f"Report constraint counts are inconsistent for {report_name}")
        _assert_metrics_match(candidate_collision, record.get("collision", {}), report_name)
        _assert_metrics_match(candidate_collision_eligibility, record.get("collision_eligibility", {}), f"{report_name} eligibility")
        candidate_hard_pin_error = float(np.max(np.linalg.norm(candidate_positions[main_indices][source_hard_pin_mask] - expected_shirt_a[main_indices][source_hard_pin_mask], axis=1))) if np.any(source_hard_pin_mask) else 0.0
        if candidate_hard_pin_error > 1.0e-8 or record.get("hard_pin_max_error_m") is None or not math.isclose(float(record["hard_pin_max_error_m"]), candidate_hard_pin_error, rel_tol=1.0e-9, abs_tol=1.0e-9):
            raise RuntimeError(f"{report_name} hard pin error does not pass")
        if record.get("nan_count") != 0 or record.get("inf_count") != 0:
            raise RuntimeError(f"{report_name} contains invalid geometry or collision penetration")
        if record.get("review_export_eligible") and candidate_collision_eligibility["count_signed_lt_minus_0_001"] != 0:
            raise RuntimeError(f"Eligible {report_name} contains collision penetration")

    for name in COLLECTIONS:
        collection = bpy.data.collections[name]
        if collection.hide_viewport or collection.hide_render:
            raise RuntimeError(f"Collection {name} is globally hidden")
    layer_collections = _layer_collections(bpy.context.view_layer.layer_collection)
    for name in COLLECTIONS:
        layer_collection = layer_collections.get(name)
        if layer_collection is None or layer_collection.exclude or layer_collection.hide_viewport:
            raise RuntimeError(f"LayerCollection visibility mismatch for {name}")
    expected_object_hidden = {
        body_meshes[0].name: False,
        CANDIDATE_OBJECTS["01_A_BASELINE_NORMAL"]: True,
        CANDIDATE_OBJECTS["02_A_REVIEW_STIFF"]: True,
        CANDIDATE_OBJECTS[DEFAULT_VISIBLE_COLLECTION]: False,
        CANDIDATE_OBJECTS["04_A_REVIEW_SOFT"]: True,
    }
    for object_name, hidden in expected_object_hidden.items():
        if bpy.data.objects[object_name].hide_get() != hidden:
            raise RuntimeError(f"Object visibility mismatch for {object_name}")
    scene = bpy.context.scene
    expected_scene_properties = {
        "cloth2bones_static_equilibrium": True,
        "review_only_visual_checkpoint": True,
        "dem_bones_executed": False,
        "bone_generation_executed": False,
        "weight_generation_executed": False,
        "default_visible_collection": DEFAULT_VISIBLE_COLLECTION,
    }
    for key, expected in expected_scene_properties.items():
        if scene.get(key) != expected:
            raise RuntimeError(f"Scene property {key!r} is inconsistent")

    source_stat = source.stat()
    source_hash = _sha256(source)
    source_record = report.get("source", {})
    source_unchanged = (
        source_hash == SOURCE_SHA256
        and source_hash == source_hash_before
        and source_record.get("sha256") == SOURCE_SHA256
        and source_record.get("sha256_after") == SOURCE_SHA256
        and source_stat.st_size == source_stat_before.st_size
        and source_record.get("size") == source_stat_before.st_size
        and source_record.get("size_after") == source_stat_before.st_size
        and source_stat.st_mtime_ns == source_stat_before.st_mtime_ns
        and source_record.get("mtime_ns") == source_stat_before.st_mtime_ns
        and source_record.get("mtime_ns_after") == source_stat_before.st_mtime_ns
        and source_record.get("unchanged") is True
    )
    if not source_unchanged:
        raise RuntimeError("Source Blend hash, size, or mtime is not unchanged")
    if report.get("original_total_vertex_count") != original_vertices or report.get("original_total_face_count") != original_faces:
        raise RuntimeError("Report original topology counts are inconsistent")
    main_report = report.get("main_component", {})
    if main_report.get("vertices") != EXPECTED_MAIN_VERTICES or main_report.get("faces") != EXPECTED_MAIN_FACES or main_report.get("stretch_edge_count") != len(source_edges):
        raise RuntimeError("Report main component counts are inconsistent")
    if main_report.get("bend_pair_count") != source_bend_pair_count or main_report.get("hard_pin_count") != int(np.count_nonzero(source_hard_pin_mask)):
        raise RuntimeError("Report hard pin count is inconsistent")
    result = {
        "valid": True,
        "blend": str(blend),
        "report": str(report_path),
        "collections": list(COLLECTIONS),
        "pose_match": pose_match,
        "candidate_topology": {name: {"object": object_name, "vertices": original_vertices, "faces": original_faces} for name, object_name in CANDIDATE_OBJECTS.items()},
        "scene_properties": expected_scene_properties,
        "nan_inf_free": True,
        "report_consistent": True,
        "source_unchanged": True,
        "previews": [],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
