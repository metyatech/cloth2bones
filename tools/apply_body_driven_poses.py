"""Apply predicted cloth bone transforms to a clean FBX and export another FBX."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import suppress
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloth2bones.body_motion import validate_rig_pose_contract  # noqa: E402


def _args() -> argparse.Namespace:
    argv = os.sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--expected", help="Optional NPZ containing cloth_local for pre-export verification")
    parser.add_argument("--report", help="Optional JSON report for pre-export verification")
    return parser.parse_args(argv)


def _weights(mesh_obj: bpy.types.Object, names: list[str]) -> np.ndarray:
    weights = np.zeros((len(mesh_obj.data.vertices), len(names)), dtype=np.float64)
    for bone_index, name in enumerate(names):
        group = mesh_obj.vertex_groups.get(name)
        if group is None:
            continue
        for vertex_index in range(len(weights)):
            with suppress(RuntimeError):
                weights[vertex_index, bone_index] = group.weight(vertex_index)
    sums = weights.sum(axis=1)
    if np.any(sums <= 1.0e-12):
        raise RuntimeError("The target FBX contains vertices without usable bone weights")
    return weights / sums[:, None]


def main() -> None:
    args = _args()
    with np.load(args.poses, allow_pickle=False) as data:
        transforms = np.asarray(data["bone_transforms"], dtype=np.float64)
        names = [str(name) for name in data["bone_names"].tolist()]
        reference_rest = np.asarray(data["rest_vertices"], dtype=np.float64) if "rest_vertices" in data else None
        reference_weights = np.asarray(data["weights"], dtype=np.float64) if "weights" in data else None
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(Path(args.fbx).resolve()), use_anim=True, automatic_bone_orientation=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and any(mod.type == "ARMATURE" for mod in obj.modifiers)]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError("Expected one armature and one skinned mesh")
    armature = armatures[0]
    if set(names) != {bone.name for bone in armature.data.bones}:
        raise RuntimeError("Predicted bone names do not match the clean FBX")
    actual_rest = np.asarray([(meshes[0].matrix_world @ vertex.co)[:] for vertex in meshes[0].data.vertices], dtype=np.float64)
    actual_weights = _weights(meshes[0], names)
    try:
        validate_rig_pose_contract(reference_rest, reference_weights, actual_rest, actual_weights)
    except ValueError as error:
        raise RuntimeError(f"{error}; export poses from the same clean rig") from error
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    action = bpy.data.actions.new("bodyDrivenCloth")
    armature.animation_data_create()
    armature.animation_data.action = action
    scene = bpy.context.scene
    scene.frame_start = args.start
    scene.frame_end = args.start + len(transforms) - 1
    name_to_index = {name: index for index, name in enumerate(names)}
    for frame_index, frame_transforms in enumerate(transforms):
        frame = args.start + frame_index
        scene.frame_set(frame)
        for bone in armature.data.bones:
            pose_bone = armature.pose.bones[bone.name]
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.matrix = Matrix(frame_transforms[name_to_index[bone.name]].tolist()) @ bone.matrix_local
            pose_bone.keyframe_insert(data_path="location", frame=frame, group=bone.name)
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=bone.name)
            pose_bone.keyframe_insert(data_path="scale", frame=frame, group=bone.name)
    if args.expected and args.report:
        with np.load(args.expected, allow_pickle=False) as expected_data:
            expected = np.asarray(expected_data["cloth_local"], dtype=np.float64)
        base_weights = actual_weights
        rest_world = np.asarray([(meshes[0].matrix_world @ vertex.co)[:] for vertex in meshes[0].data.vertices], dtype=np.float64)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        pre_export = []
        manual = []
        for index, frame in enumerate(range(args.start, args.start + len(transforms))):
            scene.frame_set(frame)
            depsgraph.update()
            evaluated = meshes[0].evaluated_get(depsgraph)
            mesh_data = evaluated.to_mesh()
            try:
                actual = np.asarray([(evaluated.matrix_world @ vertex.co)[:] for vertex in mesh_data.vertices], dtype=np.float64)
            finally:
                evaluated.to_mesh_clear()
            distance = np.linalg.norm(actual - expected[index], axis=1)
            pre_export.append({"frame": frame, "rms": float(np.sqrt(np.mean(distance * distance))), "max_point_error": float(distance.max())})
            manual_bones = []
            for name in names:
                pose_bone = armature.pose.bones[name]
                manual_bones.append(np.asarray(pose_bone.matrix @ pose_bone.bone.matrix_local.inverted(), dtype=np.float64))
            manual_transform = np.asarray(manual_bones, dtype=np.float64)
            transformed = np.einsum("bij,vj->bvi", manual_transform[:, :3, :3], rest_world) + manual_transform[:, None, :3, 3]
            manual_points = np.einsum("vb,bvi->vi", base_weights, transformed)
            manual_distance = np.linalg.norm(manual_points - expected[index], axis=1)
            modifier_distance = np.linalg.norm(actual - manual_points, axis=1)
            manual.append({"frame": frame, "rms": float(np.sqrt(np.mean(manual_distance * manual_distance))), "modifier_minus_manual_rms": float(np.sqrt(np.mean(modifier_distance * modifier_distance)))})
        report = {
            "expected": str(Path(args.expected).resolve()),
            "mean_rms": float(np.mean([value["rms"] for value in pre_export])),
            "max_rms": float(np.max([value["rms"] for value in pre_export])),
            "max_point_error": float(np.max([value["max_point_error"] for value in pre_export])),
            "samples": [pre_export[index] for index in (0, 60, 120, 180, 239)],
            "manual_skinning": {"mean_rms": float(np.mean([value["rms"] for value in manual])), "samples": [manual[index] for index in (0, 60, 120, 180, 239)]},
        }
        Path(args.report).resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    meshes[0].select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.fbx(
        filepath=str(output),
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        apply_unit_scale=True,
        global_scale=1.0,
        apply_scale_options="FBX_SCALE_NONE",
        bake_space_transform=False,
        axis_forward="-Z",
        axis_up="Y",
        add_leaf_bones=False,
        use_armature_deform_only=True,
        mesh_smooth_type="OFF",
    )
    print(f"Exported body-driven FBX to {output}")


if __name__ == "__main__":
    main()
