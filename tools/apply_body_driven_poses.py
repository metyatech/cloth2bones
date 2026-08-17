"""Apply predicted cloth bone transforms to a clean FBX and export another FBX."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def _args() -> argparse.Namespace:
    argv = os.sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=1)
    return parser.parse_args(argv)


def main() -> None:
    args = _args()
    with np.load(args.poses, allow_pickle=False) as data:
        transforms = np.asarray(data["bone_transforms"], dtype=np.float64)
        names = [str(name) for name in data["bone_names"].tolist()]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(Path(args.fbx).resolve()), use_anim=True, automatic_bone_orientation=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and any(mod.type == "ARMATURE" for mod in obj.modifiers)]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError("Expected one armature and one skinned mesh")
    armature = armatures[0]
    if set(names) != {bone.name for bone in armature.data.bones}:
        raise RuntimeError("Predicted bone names do not match the clean FBX")
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
