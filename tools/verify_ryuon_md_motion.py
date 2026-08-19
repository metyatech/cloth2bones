"""Verify exported Ryuon motion FBXs by importing each into a clean Blender scene."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

REQUIRED_BONES = ("Hips", "Chest", "Upper_arm.L", "Upper_arm.R")
FRAME_START = 0
FRAME_END = 119
ANGLE_TOLERANCE_DEGREES = 0.5
DOWN_ANGLE_TARGET = 12.0


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--fbx", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def _angle_degrees(left: Vector, right: Vector) -> float:
    dot = max(-1.0, min(1.0, float(left.normalized().dot(right.normalized()))))
    return math.degrees(math.acos(dot))


def _matrix_delta(left, right) -> float:
    return max(abs(float(left[row][column] - right[row][column])) for row in range(4) for column in range(4))


def _source_avatar(armature: bpy.types.Object) -> bpy.types.Object:
    required_groups = set(REQUIRED_BONES)
    candidates = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or "Taisofuku" in obj.name:
            continue
        if not any(modifier.type == "ARMATURE" and modifier.object == armature for modifier in obj.modifiers):
            continue
        if required_groups.issubset({group.name for group in obj.vertex_groups}):
            candidates.append(obj)
    if len(candidates) != 1:
        raise RuntimeError(f"Source avatar selection is not unique: {[obj.name for obj in candidates]}")
    return candidates[0]


def _bone_direction(armature: bpy.types.Object, name: str) -> Vector:
    pose_bone = armature.pose.bones.get(name)
    if pose_bone is None:
        raise RuntimeError(f"Imported armature is missing {name}")
    direction = pose_bone.tail - pose_bone.head
    if direction.length <= 1.0e-8:
        raise RuntimeError(f"Imported bone {name} has zero length")
    return direction.normalized()


def _finite_mesh(obj: bpy.types.Object) -> bool:
    return all(all(math.isfinite(float(value)) for value in vertex.co) for vertex in obj.data.vertices)


def _evaluated_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        points = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()
    if not points:
        raise RuntimeError(f"Mesh {obj.name} has no evaluated vertices")
    minimum = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    maximum = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    return minimum, maximum


def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _animation_data_range(armature: bpy.types.Object) -> tuple[float, float] | None:
    animation_data = armature.animation_data
    if animation_data is None:
        return None
    ranges = []
    if animation_data.action is not None:
        ranges.append(tuple(float(value) for value in animation_data.action.frame_range))
    for track in animation_data.nla_tracks:
        for strip in track.strips:
            ranges.append((float(strip.frame_start), float(strip.frame_end)))
    if not ranges:
        return None
    return min(value[0] for value in ranges), max(value[1] for value in ranges)


def _capture_source(source: Path) -> dict[str, object]:
    bpy.ops.wm.open_mainfile(filepath=str(source))
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE" and obj.name == "ComeBody_Armature"]
    if len(armatures) != 1:
        raise RuntimeError("Source must contain exactly one ComeBody_Armature")
    armature = armatures[0]
    missing = [name for name in REQUIRED_BONES if armature.data.bones.get(name) is None]
    if missing:
        raise RuntimeError(f"Source armature is missing required bones: {missing}")
    avatar = _source_avatar(armature)
    bpy.context.scene.frame_set(FRAME_START)
    bpy.context.view_layer.update()
    minimum, maximum = _evaluated_bounds(avatar)
    return {
        "avatar_vertex_count": len(avatar.data.vertices),
        "avatar_bounds_min": [float(value) for value in minimum],
        "avatar_bounds_max": [float(value) for value in maximum],
        "avatar_dimensions": [float(value) for value in maximum - minimum],
        "rest_directions": {name: [float(value) for value in _bone_direction(armature, name)] for name in ("Upper_arm.L", "Upper_arm.R")},
    }


def _import_one(path: Path, source_info: dict[str, object]) -> dict[str, object]:
    _reset_scene()
    if not path.is_file():
        raise FileNotFoundError(path)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True, automatic_bone_orientation=False, ignore_leaf_bones=False, force_connect_children=False)
    bpy.context.view_layer.update()
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    matching_meshes = [obj for obj in meshes if len(obj.data.vertices) == source_info["avatar_vertex_count"]]
    checks: dict[str, bool] = {}
    errors: list[str] = []
    checks["one_armature"] = len(armatures) == 1
    if not checks["one_armature"]:
        errors.append(f"expected one armature, found {len(armatures)}")
    if not armatures:
        return {"fbx": str(path), "checks": checks, "errors": errors, "pass": False}
    armature = armatures[0]
    checks["required_bones"] = all(armature.data.bones.get(name) is not None for name in REQUIRED_BONES)
    if not checks["required_bones"]:
        errors.append("required bone missing")
    checks["avatar_mesh"] = len(matching_meshes) == 1
    if not checks["avatar_mesh"]:
        errors.append(f"expected one avatar mesh with {source_info['avatar_vertex_count']} vertices, found {len(matching_meshes)}")
    if not matching_meshes:
        return {"fbx": str(path), "checks": checks, "errors": errors, "pass": False}
    avatar = matching_meshes[0]
    checks["finite_vertices"] = _finite_mesh(avatar)
    if not checks["finite_vertices"]:
        errors.append("mesh contains NaN or Inf")
    imported_scene_frame_range = [bpy.context.scene.frame_start, bpy.context.scene.frame_end]
    animation_range = _animation_data_range(armature)
    exact_range = animation_range is not None and abs(animation_range[0] - FRAME_START) <= 1.0e-6 and abs(animation_range[1] - FRAME_END) <= 1.0e-6
    fbx_one_based_equivalent = animation_range is not None and abs(animation_range[0] - (FRAME_START + 1)) <= 1.0e-6 and abs(animation_range[1] - (FRAME_END + 1)) <= 1.0e-6
    checks["frame_range"] = exact_range or fbx_one_based_equivalent
    if not checks["frame_range"]:
        errors.append(f"expected animation data range {FRAME_START}..{FRAME_END}, got {animation_range}")
    bpy.context.scene.frame_start = FRAME_START
    bpy.context.scene.frame_end = FRAME_END

    frame_data: dict[str, object] = {}
    matrices: dict[int, dict[str, object]] = {}
    down = Vector((0.0, 0.0, -1.0))
    for frame in (FRAME_START, 60, FRAME_END):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        directions = {name: _bone_direction(armature, name) for name in ("Upper_arm.L", "Upper_arm.R")}
        matrices[frame] = {name: armature.pose.bones[name].matrix.copy() for name in ("Hips", "Chest")}
        frame_data[str(frame)] = {name: [float(value) for value in direction] for name, direction in directions.items()}

    active_name = "Upper_arm.L" if "LeftDown" in path.stem else "Upper_arm.R" if "RightDown" in path.stem else None
    active_names = ["Upper_arm.L", "Upper_arm.R"] if active_name is None else [active_name]
    active_angles = {name: _angle_degrees(Vector(frame_data["60"][name]), down) for name in active_names}
    checks["frame0_tpose_direction"] = all(
        _angle_degrees(Vector(frame_data["0"][name]), Vector(source_info["rest_directions"][name])) <= ANGLE_TOLERANCE_DEGREES
        for name in ("Upper_arm.L", "Upper_arm.R")
    )
    checks["frame119_returns"] = all(
        _angle_degrees(Vector(frame_data["119"][name]), Vector(frame_data["0"][name])) <= ANGLE_TOLERANCE_DEGREES
        for name in ("Upper_arm.L", "Upper_arm.R")
    )
    checks["active_frame60_down_angle"] = all(abs(angle - DOWN_ANGLE_TARGET) <= 1.0 for angle in active_angles.values())
    inactive_names = [name for name in ("Upper_arm.L", "Upper_arm.R") if name not in active_names]
    inactive_drift = {name: _angle_degrees(Vector(frame_data["0"][name]), Vector(frame_data["60"][name])) for name in inactive_names}
    checks["inactive_arm_stable"] = all(value <= ANGLE_TOLERANCE_DEGREES for value in inactive_drift.values())
    hips_drift = _matrix_delta(matrices[0]["Hips"], matrices[60]["Hips"])
    chest_drift = _matrix_delta(matrices[0]["Chest"], matrices[60]["Chest"])
    checks["hips_stable"] = hips_drift <= 1.0e-4
    checks["chest_stable"] = chest_drift <= 1.0e-4
    minimum, maximum = _evaluated_bounds(avatar)
    imported_dimensions = maximum - minimum
    source_dimensions = Vector(source_info["avatar_dimensions"])
    bbox_ratios = [float(imported_dimensions[index] / source_dimensions[index]) for index in range(3)]
    checks["bbox_scale_match"] = all(0.999 <= ratio <= 1.001 for ratio in bbox_ratios)
    if not checks["frame0_tpose_direction"]:
        errors.append("frame 0 arm direction differs from source T-pose")
    if not checks["frame119_returns"]:
        errors.append("frame 119 does not return within 0.5 degrees")
    if not checks["active_frame60_down_angle"]:
        errors.append(f"active arm frame 60 angles are outside 12 +/- 1 degrees: {active_angles}")
    if not checks["inactive_arm_stable"]:
        errors.append(f"inactive arm drift exceeds 0.5 degrees: {inactive_drift}")
    if not checks["hips_stable"] or not checks["chest_stable"]:
        errors.append(f"Hips/Chest drift: hips={hips_drift}, chest={chest_drift}")
    if not checks["bbox_scale_match"]:
        errors.append(f"bbox dimension ratios are outside 0.999..1.001: {bbox_ratios}")
    return {
        "fbx": str(path),
        "armature": armature.name,
        "avatar_mesh": avatar.name,
        "vertex_count": len(avatar.data.vertices),
        "imported_scene_frame_range": imported_scene_frame_range,
        "animation_data_range": list(animation_range) if animation_range is not None else None,
        "animation_range_encoding": "exact_zero_based" if exact_range else "fbx_one_based_equivalent" if fbx_one_based_equivalent else "invalid",
        "frame_data": frame_data,
        "active_arm_frame60_down_angle_degrees": active_angles,
        "inactive_arm_frame0_to60_angle_degrees": inactive_drift,
        "hips_frame0_to60_matrix_max_abs_delta": hips_drift,
        "chest_frame0_to60_matrix_max_abs_delta": chest_drift,
        "bbox_dimensions": [float(value) for value in imported_dimensions],
        "bbox_dimension_ratios_source_to_reimport": bbox_ratios,
        "checks": checks,
        "errors": errors,
        "pass": all(checks.values()) and not errors,
    }


def main() -> None:
    args = _args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_info = _capture_source(source)
    results = [_import_one(path.resolve(), source_info) for path in args.fbx]
    report = {"source": str(source), "source_info": source_info, "results": results, "pass": all(result["pass"] for result in results)}
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "pass": report["pass"], "motions": len(results)}, indent=2))
    if not report["pass"]:
        raise RuntimeError("One or more Ryuon motion FBXs failed verification")


if __name__ == "__main__":
    main()
