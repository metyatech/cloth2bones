"""Generate one semantic-skeleton Blender Cloth physics sequence.

This Blender-side generator deliberately creates the teacher geometry with the
Cloth modifier. The downstream predictor is allowed to inspect the exported
cloth trajectory only when fitting teacher helper-bone targets; it must never
construct those targets procedurally from this motion specification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion

SEMANTIC_BONES = (
    "Hips",
    "Spine",
    "Chest",
    "Shoulder.L",
    "Upper_arm.L",
    "Lower_arm.L",
    "Shoulder.R",
    "Upper_arm.R",
    "Lower_arm.R",
)


def _args() -> argparse.Namespace:
    argv = os.sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-poses", required=True, help="NPZ containing the common cloth rest vertices")
    parser.add_argument("--triangles", required=True, help="NPZ containing the common cloth triangles")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--motion", required=True, choices=("A_left_down", "B_right_down", "C_both_down", "D_left_down_up", "E_right_down_up", "F_alternating", "G_diagonal", "H_unseen_combined", "I_speed_variant", "J_reverse_history"))
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--settle", type=int, default=20)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args(argv)


def _read_common_mesh(reference_path: Path, triangles_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(reference_path, allow_pickle=False) as source:
        vertices = np.asarray(source["rest_vertices"], dtype=np.float64)
    with np.load(triangles_path, allow_pickle=False) as source:
        triangles = np.asarray(source["triangles"], dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected rest_vertices with shape (V, 3), got {vertices.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Expected triangles with shape (F, 3), got {triangles.shape}")
    if int(triangles.max()) >= len(vertices):
        raise ValueError("Triangle index exceeds the common cloth vertex count")
    return vertices, triangles


def _add_armature() -> bpy.types.Object:
    bpy.ops.object.armature_add(enter_editmode=True, location=(0.0, 0.0, 0.0))
    armature = bpy.context.object
    armature.name = "SemanticBody"
    armature.data.name = "SemanticBodyRig"
    armature.data.display_type = "BBONE"
    armature.data.show_names = True
    for bone in list(armature.data.edit_bones):
        armature.data.edit_bones.remove(bone)

    definitions = {
        "Hips": ((0.0, 0.0, 0.72), (0.0, 0.0, 0.94), None),
        "Spine": ((0.0, 0.0, 0.94), (0.0, 0.0, 1.20), "Hips"),
        "Chest": ((0.0, 0.0, 1.20), (0.0, 0.0, 1.43), "Spine"),
        "Shoulder.L": ((-0.02, 0.0, 1.39), (-0.20, 0.0, 1.42), "Chest"),
        "Upper_arm.L": ((-0.20, 0.0, 1.42), (-0.43, 0.0, 1.42), "Shoulder.L"),
        "Lower_arm.L": ((-0.43, 0.0, 1.42), (-0.66, 0.0, 1.42), "Upper_arm.L"),
        "Shoulder.R": ((0.02, 0.0, 1.39), (0.20, 0.0, 1.42), "Chest"),
        "Upper_arm.R": ((0.20, 0.0, 1.42), (0.43, 0.0, 1.42), "Shoulder.R"),
        "Lower_arm.R": ((0.43, 0.0, 1.42), (0.66, 0.0, 1.42), "Upper_arm.R"),
    }
    for name in SEMANTIC_BONES:
        head, tail, parent_name = definitions[name]
        bone = armature.data.edit_bones.new(name)
        bone.head = head
        bone.tail = tail
        bone.use_connect = False
        if parent_name:
            bone.parent = armature.data.edit_bones[parent_name]
    bpy.ops.object.mode_set(mode="OBJECT")
    for bone in armature.pose.bones:
        bone.rotation_mode = "QUATERNION"
    return armature


def _collision_sphere(armature: bpy.types.Object, name: str, location: tuple[float, float, float], scale: tuple[float, float, float], bone_name: str) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
    obj.display_type = "WIRE"
    obj.hide_render = True
    obj["semantic_bone"] = bone_name
    obj.modifiers.new("ClothCollision", "COLLISION")
    bpy.context.view_layer.update()
    if obj.collision is not None:
        if hasattr(obj.collision, "thickness_outer"):
            obj.collision.thickness_outer = 0.018
        if hasattr(obj.collision, "thickness_inner"):
            obj.collision.thickness_inner = 0.008
        if hasattr(obj.collision, "use_culling"):
            obj.collision.use_culling = False
    return obj


def _add_body_colliders(armature: bpy.types.Object) -> list[bpy.types.Object]:
    return [
        _collision_sphere(armature, "Body.Torso", (0.0, 0.0, 1.10), (0.245, 0.18, 0.43), "Spine"),
        _collision_sphere(armature, "Body.Chest", (0.0, -0.005, 1.38), (0.255, 0.185, 0.25), "Chest"),
        _collision_sphere(armature, "Body.Shoulder.L", (-0.20, 0.0, 1.41), (0.14, 0.16, 0.15), "Shoulder.L"),
        _collision_sphere(armature, "Body.UpperArm.L", (-0.33, 0.0, 1.42), (0.20, 0.115, 0.115), "Upper_arm.L"),
        _collision_sphere(armature, "Body.LowerArm.L", (-0.55, 0.0, 1.42), (0.19, 0.105, 0.105), "Lower_arm.L"),
        _collision_sphere(armature, "Body.Shoulder.R", (0.20, 0.0, 1.41), (0.14, 0.16, 0.15), "Shoulder.R"),
        _collision_sphere(armature, "Body.UpperArm.R", (0.33, 0.0, 1.42), (0.20, 0.115, 0.115), "Upper_arm.R"),
        _collision_sphere(armature, "Body.LowerArm.R", (0.55, 0.0, 1.42), (0.19, 0.105, 0.105), "Lower_arm.R"),
    ]


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def _add_cloth(armature: bpy.types.Object, rest_vertices: np.ndarray, triangles: np.ndarray, frames: int) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("PhysicsTShirtMesh")
    mesh.from_pydata(rest_vertices.tolist(), [], triangles.tolist())
    mesh.update()
    cloth_obj = bpy.data.objects.new("PhysicsTShirt", mesh)
    bpy.context.scene.collection.objects.link(cloth_obj)
    cloth_obj.data.materials.append(_material("PhysicsTShirtMaterial", (0.12, 0.35, 0.78, 1.0)))

    group = cloth_obj.vertex_groups.new(name="Chest")
    group.add(list(range(len(rest_vertices))), 1.0, "REPLACE")
    pin_group = cloth_obj.vertex_groups.new(name="Pin")
    pin_indices = np.flatnonzero(rest_vertices[:, 2] >= np.quantile(rest_vertices[:, 2], 0.88)).astype(int).tolist()
    if len(pin_indices) < 8:
        pin_indices = np.argsort(rest_vertices[:, 2])[-8:].astype(int).tolist()
    pin_group.add(pin_indices, 1.0, "REPLACE")
    armature_modifier = cloth_obj.modifiers.new("ChestFollow", "ARMATURE")
    armature_modifier.object = armature
    armature_modifier.use_deform_preserve_volume = False

    cloth = cloth_obj.modifiers.new("BlenderClothTeacher", "CLOTH")
    settings = cloth.settings
    settings.quality = 4
    settings.mass = 0.32
    settings.tension_stiffness = 18.0
    settings.compression_stiffness = 18.0
    settings.shear_stiffness = 8.0
    settings.bending_stiffness = 0.35
    settings.tension_damping = 7.0
    settings.compression_damping = 7.0
    settings.shear_damping = 6.0
    settings.bending_damping = 0.8
    settings.vertex_group_mass = "Pin"
    collision = cloth.collision_settings
    collision.use_collision = True
    collision.use_self_collision = False
    collision.distance_min = 0.012
    collision.collision_quality = 3
    cache = cloth.point_cache
    cache.frame_start = 1
    cache.frame_end = frames
    cache.name = "PhysicsTeacherCache"
    return cloth_obj


def _smoothstep(value: float) -> float:
    x = max(0.0, min(1.0, value))
    return x * x * (3.0 - 2.0 * x)


def _triangle_wave(value: float) -> float:
    x = max(0.0, min(1.0, value))
    return 2.0 * x if x <= 0.5 else 2.0 * (1.0 - x)


def _motion_controls(motion: str, phase: float) -> dict[str, tuple[float, float]]:
    p = max(0.0, min(1.0, phase))
    smooth = _smoothstep(p)
    controls = {name: (0.0, 0.0) for name in ("Chest", "Upper_arm.L", "Lower_arm.L", "Upper_arm.R", "Lower_arm.R")}
    if motion == "A_left_down":
        controls["Upper_arm.L"] = (math.radians(-68.0) * smooth, 0.0)
        controls["Lower_arm.L"] = (math.radians(-18.0) * smooth, math.radians(8.0) * smooth)
    elif motion == "B_right_down":
        controls["Upper_arm.R"] = (math.radians(68.0) * smooth, 0.0)
        controls["Lower_arm.R"] = (math.radians(18.0) * smooth, math.radians(-8.0) * smooth)
    elif motion == "C_both_down":
        controls["Upper_arm.L"] = (math.radians(-64.0) * smooth, 0.0)
        controls["Upper_arm.R"] = (math.radians(64.0) * smooth, 0.0)
        controls["Lower_arm.L"] = (math.radians(-28.0) * smooth, 0.0)
        controls["Lower_arm.R"] = (math.radians(28.0) * smooth, 0.0)
    elif motion == "D_left_down_up":
        value = _triangle_wave(p)
        controls["Upper_arm.L"] = (math.radians(-72.0) * value, 0.0)
        controls["Lower_arm.L"] = (math.radians(-25.0) * value, 0.0)
    elif motion == "E_right_down_up":
        value = _triangle_wave(p)
        controls["Upper_arm.R"] = (math.radians(72.0) * value, 0.0)
        controls["Lower_arm.R"] = (math.radians(25.0) * value, 0.0)
    elif motion == "F_alternating":
        left = 0.5 + 0.5 * math.sin(2.0 * math.pi * p - math.pi / 2.0)
        right = 0.5 + 0.5 * math.sin(2.0 * math.pi * p + math.pi / 2.0)
        controls["Upper_arm.L"] = (math.radians(-56.0) * left, 0.0)
        controls["Upper_arm.R"] = (math.radians(56.0) * right, 0.0)
        controls["Lower_arm.L"] = (math.radians(-20.0) * left, 0.0)
        controls["Lower_arm.R"] = (math.radians(20.0) * right, 0.0)
    elif motion == "G_diagonal":
        controls["Chest"] = (math.radians(7.0) * math.sin(math.pi * p), math.radians(-5.0) * smooth)
        controls["Upper_arm.L"] = (math.radians(-48.0) * smooth, math.radians(-18.0) * smooth)
        controls["Upper_arm.R"] = (math.radians(36.0) * smooth, math.radians(22.0) * smooth)
        controls["Lower_arm.L"] = (math.radians(-35.0) * smooth, math.radians(5.0) * smooth)
        controls["Lower_arm.R"] = (math.radians(18.0) * smooth, math.radians(-6.0) * smooth)
    elif motion == "H_unseen_combined":
        left = _smoothstep(min(1.0, p * 1.18))
        right = _smoothstep(max(0.0, (p - 0.08) / 0.92))
        controls["Chest"] = (math.radians(10.0) * math.sin(1.5 * math.pi * p), math.radians(8.0) * math.sin(math.pi * p))
        controls["Upper_arm.L"] = (math.radians(-82.0) * left, math.radians(-12.0) * math.sin(math.pi * p))
        controls["Upper_arm.R"] = (math.radians(49.0) * right, math.radians(27.0) * math.sin(math.pi * p))
        controls["Lower_arm.L"] = (math.radians(-38.0) * left, math.radians(14.0) * left)
        controls["Lower_arm.R"] = (math.radians(31.0) * right, math.radians(-13.0) * right)
    elif motion == "I_speed_variant":
        value = 0.5 + 0.5 * math.sin(3.0 * math.pi * p - math.pi / 2.0)
        controls["Upper_arm.L"] = (math.radians(-66.0) * value, 0.0)
        controls["Upper_arm.R"] = (math.radians(66.0) * (1.0 - value), 0.0)
        controls["Lower_arm.L"] = (math.radians(-22.0) * value, 0.0)
        controls["Lower_arm.R"] = (math.radians(22.0) * (1.0 - value), 0.0)
    elif motion == "J_reverse_history":
        first = _triangle_wave(min(1.0, p * 1.55))
        late = _smoothstep(max(0.0, (p - 0.42) / 0.58))
        value = min(1.0, first * 0.9 + late * 0.25)
        controls["Upper_arm.L"] = (math.radians(-70.0) * value, 0.0)
        controls["Upper_arm.R"] = (math.radians(60.0) * value, 0.0)
        controls["Lower_arm.L"] = (math.radians(-26.0) * value, 0.0)
        controls["Lower_arm.R"] = (math.radians(22.0) * value, 0.0)
    return controls


def _set_pose(armature: bpy.types.Object, controls: dict[str, tuple[float, float]], frame: int) -> None:
    for bone in armature.pose.bones:
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    for name, (around_y, around_x) in controls.items():
        bone = armature.pose.bones[name]
        # Blender bones point along local +Y. Local X swings the arm in the
        # world X/Z plane; local Z supplies the secondary diagonal component.
        q_y = Quaternion((1.0, 0.0, 0.0), around_y)
        q_x = Quaternion((0.0, 0.0, 1.0), around_x)
        bone.rotation_quaternion = q_y @ q_x
    for bone in armature.pose.bones:
        bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=bone.name)


def _keyframe_colliders(armature: bpy.types.Object, colliders: list[bpy.types.Object], frame: int) -> None:
    for obj in colliders:
        bone_name = str(obj["semantic_bone"])
        pose_bone = armature.pose.bones[bone_name]
        rest_bone = armature.data.bones[bone_name]
        delta = pose_bone.matrix @ rest_bone.matrix_local.inverted()
        obj.rotation_mode = "QUATERNION"
        obj.matrix_world = delta
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        obj.keyframe_insert(data_path="scale", frame=frame)


def _evaluate_vertices(obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph) -> np.ndarray:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return np.asarray([(evaluated.matrix_world @ vertex.co)[:] for vertex in mesh.vertices], dtype=np.float64)
    finally:
        evaluated.to_mesh_clear()


def _evaluate_skeleton(armature: bpy.types.Object) -> np.ndarray:
    result = []
    for name in SEMANTIC_BONES:
        pose = armature.pose.bones[name]
        rest = armature.data.bones[name].matrix_local
        result.append(np.asarray(pose.matrix @ rest.inverted(), dtype=np.float64))
    return np.asarray(result, dtype=np.float64)


def _bake(scene: bpy.types.Scene, cloth_obj: bpy.types.Object) -> None:
    scene.frame_set(scene.frame_start)
    bpy.context.view_layer.update()
    bpy.ops.ptcache.bake_all(bake=True)


def _save_scene(scene: bpy.types.Scene, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))


def main() -> None:
    args = _args()
    if args.frames <= args.settle + 10:
        raise ValueError("frames must leave at least ten motion frames after settle")
    output_root = Path(args.out_root).resolve()
    sequence_root = output_root / "physics_sequences" / args.motion
    sequence_root.mkdir(parents=True, exist_ok=True)
    rest_vertices, triangles = _read_common_mesh(Path(args.reference_poses).resolve(), Path(args.triangles).resolve())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = int(round(args.fps))
    scene.frame_start = 1
    scene.frame_end = args.frames
    scene.frame_set(1)
    scene.gravity = (0.0, 0.0, -9.81)
    armature = _add_armature()
    colliders = _add_body_colliders(armature)
    cloth_obj = _add_cloth(armature, rest_vertices, triangles, args.frames)
    scene["cloth2bones_poc"] = "3.3_blender_physics"
    scene["sequence_id"] = args.motion
    scene["semantic_bones"] = list(SEMANTIC_BONES)

    for frame in range(1, args.frames + 1):
        phase = (frame - 1 - args.settle) / max(1, args.frames - args.settle - 1)
        controls = _motion_controls(args.motion, phase)
        _set_pose(armature, controls, frame)
        bpy.context.view_layer.update()
        _keyframe_colliders(armature, colliders, frame)
    if armature.animation_data and armature.animation_data.action:
        armature.animation_data.action.name = f"SemanticMotion_{args.motion}"

    bpy.context.view_layer.update()
    _bake(scene, cloth_obj)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cloth_traj = []
    skeleton_traj = []
    for frame in range(1, args.frames + 1):
        scene.frame_set(frame)
        depsgraph.update()
        cloth_traj.append(_evaluate_vertices(cloth_obj, depsgraph))
        skeleton_traj.append(_evaluate_skeleton(armature))
    cloth_values = np.asarray(cloth_traj, dtype=np.float64)
    skeleton_values = np.asarray(skeleton_traj, dtype=np.float64)
    cloth_velocity = np.concatenate([np.zeros_like(cloth_values[:1]), np.diff(cloth_values, axis=0)], axis=0)
    root_values = np.tile(np.eye(4, dtype=np.float64), (args.frames, 1, 1))
    np.savez_compressed(
        sequence_root / "teacher.npz",
        sequence_id=np.asarray(args.motion),
        motion_label=np.asarray(args.motion),
        fps=np.asarray(args.fps),
        frames=np.arange(1, args.frames + 1, dtype=np.int32),
        rest_vertices=rest_vertices,
        triangles=triangles,
        traj=cloth_values,
        traj_vel=cloth_velocity,
        skeleton_transforms=skeleton_values,
        skeleton_names=np.asarray(SEMANTIC_BONES),
        root_transforms=root_values,
    )
    metadata = {
        "sequence_id": args.motion,
        "motion_label": args.motion,
        "frames": args.frames,
        "settle_frames": args.settle,
        "fps": args.fps,
        "cloth_vertices": int(len(rest_vertices)),
        "cloth_triangles": int(len(triangles)),
        "semantic_bones": list(SEMANTIC_BONES),
        "teacher_source": "Blender 5.2 Cloth modifier cache",
        "holdout": args.motion == "H_unseen_combined",
        "hysteresis_candidate": args.motion in {"D_left_down_up", "J_reverse_history"},
    }
    (sequence_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _save_scene(scene, sequence_root / "scene.blend")
    print(json.dumps({"sequence_id": args.motion, "frames": args.frames, "vertices": len(rest_vertices), "triangles": len(triangles), "traj_min": float(cloth_values.min()), "traj_max": float(cloth_values.max())}, indent=2))


if __name__ == "__main__":
    main()
