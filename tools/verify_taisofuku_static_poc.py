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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.export_ryuon_md_motion import _rest_matrices, _set_motion_pose, _set_rest_pose  # noqa: E402

SOURCE_SHA256 = "00eace7d79ed201dd0c5684c3301f9af6c5653bff05c215e4d8ed5c6433801df"
EXPECTED_VERTICES = 121471
EXPECTED_FACES = 115103
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

    if len(expected_body_t) != len(expected_body_a) or len(expected_body_t) != len(source_body.data.vertices):
        raise RuntimeError("Source evaluated body topology is inconsistent")
    if len(expected_shirt_t) != len(expected_shirt_a) or len(expected_shirt_t) != len(source_shirt.data.vertices):
        raise RuntimeError("Source evaluated shirt topology is inconsistent")
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
    review_body = _world_mesh_data(body_meshes[0], review_depsgraph)
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
        report_name = CANDIDATE_RECORDS.get(collection_name)
        if report_name is None:
            continue
        record = candidate_records.get(report_name)
        if not record or not record.get("topology_preserved") or not record.get("review_export_eligible"):
            raise RuntimeError(f"Report does not validate candidate {report_name}")
        if record.get("display_collection") != collection_name or record.get("display_object") != expected_object_name:
            raise RuntimeError(f"Report display mapping is inconsistent for {report_name}")
        if record.get("output_vertices") != original_vertices or record.get("output_faces") != original_faces:
            raise RuntimeError(f"Report topology counts are inconsistent for {report_name}")

    expected_visibility = {BODY_COLLECTION: False, "01_A_BASELINE_NORMAL": True, "02_A_REVIEW_STIFF": True, DEFAULT_VISIBLE_COLLECTION: False, "04_A_REVIEW_SOFT": True}
    for name, hidden in expected_visibility.items():
        if bpy.data.collections[name].hide_viewport != hidden or bpy.data.collections[name].hide_render != hidden:
            raise RuntimeError(f"Collection visibility mismatch for {name}")
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
