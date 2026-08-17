"""Export an NPZ cloth trajectory as a shape-key animated FBX."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import bpy
import numpy as np


def _args() -> argparse.Namespace:
    argv = os.sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--blend", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = _args()
    with np.load(args.npz, allow_pickle=False) as source:
        trajectory = np.asarray(source["traj"], dtype=np.float64)
        triangles = np.asarray(source["triangles"], dtype=np.int32)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new("PhysicsTeacherMesh")
    mesh.from_pydata(trajectory[0].tolist(), [], triangles.tolist())
    mesh.update()
    obj = bpy.data.objects.new("PhysicsTeacher", mesh)
    bpy.context.scene.collection.objects.link(obj)
    material = bpy.data.materials.new("PhysicsTeacherMaterial")
    material.diffuse_color = (0.80, 0.80, 0.88, 1.0)
    mesh.materials.append(material)
    obj.shape_key_add(name="Basis")
    blocks = []
    for frame_index, values in enumerate(trajectory):
        block = obj.shape_key_add(name=f"PhysicsFrame_{frame_index + 1:04d}")
        for vertex, point in zip(block.data, values, strict=True):
            vertex.co = point
        blocks.append(block)
    for frame_index in range(len(trajectory)):
        for block_index, block in enumerate(blocks):
            block.value = 1.0 if block_index == frame_index else 0.0
            block.keyframe_insert(data_path="value", frame=frame_index + 1)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(trajectory)
    scene.render.fps = 30
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(output),
        use_selection=True,
        object_types={"MESH"},
        apply_unit_scale=True,
        global_scale=1.0,
        apply_scale_options="FBX_SCALE_NONE",
        bake_space_transform=False,
        axis_forward="-Z",
        axis_up="Y",
        bake_anim=True,
        add_leaf_bones=False,
        mesh_smooth_type="OFF",
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.blend).resolve()))
    print(f"Exported physics teacher shape-key FBX to {output}")


if __name__ == "__main__":
    main()
