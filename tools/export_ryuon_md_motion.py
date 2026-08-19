"""Export deterministic Ryuon arm motions for Marvelous Designer import."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

FRAME_START = 0
FRAME_END = 119
FPS = 30
REQUIRED_BONES = ("Hips", "Chest", "Upper_arm.L", "Upper_arm.R")
MOTION_NAMES = ("LeftDown", "RightDown", "BothDown")
ARM_NAMES = ("Upper_arm.L", "Upper_arm.R")
DOWN_ANGLE_DEGREES = 12.0


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_delta(left: Matrix, right: Matrix) -> float:
    return max(abs(float(left[row][column] - right[row][column])) for row in range(4) for column in range(4))


def _angle_degrees(left: Vector, right: Vector) -> float:
    left_normalized = left.normalized()
    right_normalized = right.normalized()
    dot = max(-1.0, min(1.0, float(left_normalized.dot(right_normalized))))
    return math.degrees(math.acos(dot))


def _smoothstep(value: float) -> float:
    clamped = max(0.0, min(1.0, value))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _motion_progress(frame: int) -> float:
    if frame < 20:
        return 0.0
    if frame < 60:
        return _smoothstep((frame - 20) / 40.0)
    if frame < 80:
        return 1.0
    return 1.0 - _smoothstep((frame - 80) / 40.0)


def _avatar_candidates(armature: bpy.types.Object) -> list[bpy.types.Object]:
    required_groups = set(REQUIRED_BONES)
    candidates = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or "Taisofuku" in obj.name:
            continue
        has_target = any(modifier.type == "ARMATURE" and modifier.object == armature for modifier in obj.modifiers)
        groups = {group.name for group in obj.vertex_groups}
        if has_target and required_groups.issubset(groups):
            candidates.append(obj)
    return candidates


def _candidate_summary(obj: bpy.types.Object) -> dict[str, object]:
    return {"name": obj.name, "vertices": len(obj.data.vertices), "vertex_groups": sorted(group.name for group in obj.vertex_groups)}


def _require_source(source: Path) -> tuple[bpy.types.Object, bpy.types.Object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    bpy.ops.wm.open_mainfile(filepath=str(source))
    armature = bpy.data.objects.get("ComeBody_Armature")
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("ComeBody_Armature must exist as an ARMATURE object")
    missing = [name for name in REQUIRED_BONES if armature.data.bones.get(name) is None]
    if missing:
        raise RuntimeError(f"ComeBody_Armature is missing required bones: {missing}")
    candidates = _avatar_candidates(armature)
    if len(candidates) != 1:
        details = [_candidate_summary(candidate) for candidate in candidates]
        raise RuntimeError(f"Avatar mesh selection is not unique; candidates={json.dumps(details, indent=2)}")
    return armature, candidates[0]


def _rest_matrices(armature: bpy.types.Object) -> dict[str, Matrix]:
    return {bone.name: bone.matrix_local.copy() for bone in armature.data.bones}


def _arm_direction(armature: bpy.types.Object, bone_name: str) -> Vector:
    bone = armature.data.bones[bone_name]
    direction = bone.tail_local - bone.head_local
    if direction.length <= 1.0e-8:
        raise RuntimeError(f"Bone {bone_name} has zero rest length")
    return direction.normalized()


def _target_direction(rest_direction: Vector) -> Vector:
    outside_sign = 1.0 if rest_direction.x >= 0.0 else -1.0
    angle = math.radians(DOWN_ANGLE_DEGREES)
    target = Vector((outside_sign * math.sin(angle), 0.0, -math.cos(angle)))
    return target.normalized()


def _pose_matrix_for_direction(rest_matrix: Matrix, rest_direction: Vector, target_direction: Vector) -> Matrix:
    rotation = rest_direction.rotation_difference(target_direction)
    head = rest_matrix.translation.copy()
    return Matrix.Translation(head) @ rotation.to_matrix().to_4x4() @ Matrix.Translation(-head) @ rest_matrix


def _set_rest_pose(armature: bpy.types.Object, rest_matrices: dict[str, Matrix]) -> None:
    armature.data.pose_position = "POSE"
    armature.animation_data_clear()
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.matrix = rest_matrices[pose_bone.name]
    bpy.context.view_layer.update()


def _set_motion_pose(armature: bpy.types.Object, rest_matrices: dict[str, Matrix], active: set[str], progress: float) -> None:
    for pose_bone in armature.pose.bones:
        pose_bone.matrix = rest_matrices[pose_bone.name]
    for bone_name in sorted(active):
        rest_direction = _arm_direction(armature, bone_name)
        target_direction = _target_direction(rest_direction)
        interpolated_direction = rest_direction.lerp(target_direction, progress).normalized()
        armature.pose.bones[bone_name].matrix = _pose_matrix_for_direction(rest_matrices[bone_name], rest_direction, interpolated_direction)
    bpy.context.view_layer.update()


def _keyframe_pose(armature: bpy.types.Object, frame: int) -> None:
    for pose_bone in armature.pose.bones:
        pose_bone.keyframe_insert(data_path="location", frame=frame, group=pose_bone.name)
        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=pose_bone.name)
        pose_bone.keyframe_insert(data_path="scale", frame=frame, group=pose_bone.name)


def _make_motion(armature: bpy.types.Object, rest_matrices: dict[str, Matrix], motion_name: str) -> dict[str, object]:
    active = {
        "LeftDown": {"Upper_arm.L"},
        "RightDown": {"Upper_arm.R"},
        "BothDown": {"Upper_arm.L", "Upper_arm.R"},
    }[motion_name]
    _set_rest_pose(armature, rest_matrices)
    action = bpy.data.actions.new(f"Ryuon_MD_{motion_name}")
    armature.animation_data_create()
    armature.animation_data.action = action
    scene = bpy.context.scene
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_END
    scene.render.fps = FPS
    for frame in range(FRAME_START, FRAME_END + 1):
        scene.frame_set(frame)
        _set_motion_pose(armature, rest_matrices, active, _motion_progress(frame))
        _keyframe_pose(armature, frame)
    scene.frame_set(FRAME_START)
    bpy.context.view_layer.update()
    return {"name": motion_name, "active_arms": sorted(active), "action": action.name}


def _direction_record(armature: bpy.types.Object, frame: int) -> dict[str, list[float]]:
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    result = {}
    for name in ARM_NAMES:
        pose_bone = armature.pose.bones[name]
        result[name] = [float(value) for value in (pose_bone.tail - pose_bone.head).normalized()]
    return result


def _export(armature: bpy.types.Object, avatar: bpy.types.Object, output: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    avatar.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.fbx(
        filepath=str(output),
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        add_leaf_bones=False,
        use_armature_deform_only=False,
        axis_forward="-Z",
        axis_up="Y",
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        bake_space_transform=False,
        mesh_smooth_type="OFF",
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"FBX export did not produce a non-empty file: {output}")


def main() -> None:
    args = _args()
    source = args.source.resolve()
    output_root = args.output_root.resolve()
    motion_root = output_root / "motion"
    motion_root.mkdir(parents=True, exist_ok=True)
    source_before = source.stat()
    source_hash_before = _sha256(source)
    armature, avatar = _require_source(source)
    rest_matrices = _rest_matrices(armature)
    rest_directions = {name: _arm_direction(armature, name) for name in ARM_NAMES}
    down = Vector((0.0, 0.0, -1.0))
    motions = []
    for motion_name in MOTION_NAMES:
        motion_info = _make_motion(armature, rest_matrices, motion_name)
        output = motion_root / f"Ryuon_MD_{motion_name}.fbx"
        _export(armature, avatar, output)
        active = set(motion_info["active_arms"])
        directions = {str(frame): _direction_record(armature, frame) for frame in (0, 60, 119)}
        active_angles = {name: _angle_degrees(Vector(directions["60"][name]), down) for name in active}
        inactive_drift = {
            name: _angle_degrees(Vector(directions["0"][name]), Vector(directions["60"][name]))
            for name in ARM_NAMES
            if name not in active
        }
        hips = armature.pose.bones["Hips"]
        chest = armature.pose.bones["Chest"]
        bpy.context.scene.frame_set(0)
        bpy.context.view_layer.update()
        hips0 = hips.matrix.copy()
        chest0 = chest.matrix.copy()
        bpy.context.scene.frame_set(60)
        bpy.context.view_layer.update()
        hips60 = hips.matrix.copy()
        chest60 = chest.matrix.copy()
        motions.append(
            {
                "name": motion_name,
                "active_arms": sorted(active),
                "source_blend": str(source),
                "source_sha256": source_hash_before,
                "armature": armature.name,
                "avatar_mesh": avatar.name,
                "vertex_count": len(avatar.data.vertices),
                "frame_start": FRAME_START,
                "frame_end": FRAME_END,
                "fps": FPS,
                "rest_directions": {name: [float(value) for value in vector] for name, vector in rest_directions.items()},
                "directions": directions,
                "active_arm_frame60_down_angle_degrees": active_angles,
                "inactive_arm_frame0_to60_angle_degrees": inactive_drift,
                "hips_frame0_to60_matrix_max_abs_delta": _matrix_delta(hips0, hips60),
                "chest_frame0_to60_matrix_max_abs_delta": _matrix_delta(chest0, chest60),
                "fbx_path": str(output),
                "fbx_size": output.stat().st_size,
            }
        )
    source_after = source.stat()
    source_hash_after = _sha256(source)
    report = {
        "source_blend": str(source),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_unchanged": source_hash_before == source_hash_after and source_before.st_size == source_after.st_size and source_before.st_mtime_ns == source_after.st_mtime_ns,
        "armature": armature.name,
        "avatar_mesh": avatar.name,
        "avatar_mesh_vertex_count": len(avatar.data.vertices),
        "frame_start": FRAME_START,
        "frame_end": FRAME_END,
        "fps": FPS,
        "motions": motions,
    }
    report_path = output_root / "ryuon_md_motion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["source_unchanged"]:
        raise RuntimeError("Source Blend changed during export")
    print(json.dumps({"report": str(report_path), "motions": len(motions), "avatar_mesh": avatar.name, "source_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
