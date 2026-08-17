"""Export clean Blender rig weights, rest vertices, and per-frame bone transforms."""

from __future__ import annotations

import argparse
import os
from contextlib import suppress
from pathlib import Path

import bpy
import numpy as np


def _args() -> argparse.Namespace:
    argv = os.sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=240)
    return parser.parse_args(argv)


def _world_vertices(mesh_obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph) -> np.ndarray:
    evaluated = mesh_obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return np.asarray([(evaluated.matrix_world @ vertex.co)[:] for vertex in mesh.vertices], dtype=np.float64)
    finally:
        evaluated.to_mesh_clear()


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
        raise RuntimeError("The FBX contains vertices without usable bone weights")
    return weights / sums[:, None]


def main() -> None:
    args = _args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(Path(args.fbx).resolve()), use_anim=True, automatic_bone_orientation=False)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and any(mod.type == "ARMATURE" for mod in obj.modifiers)]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(meshes) != 1 or len(armatures) != 1:
        raise RuntimeError(f"Expected one skinned mesh and one armature, got {len(meshes)} and {len(armatures)}")
    mesh = meshes[0]
    armature = armatures[0]
    names = [bone.name for bone in armature.data.bones]
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene.frame_set(args.start)
    depsgraph.update()
    rest_vertices = _world_vertices(mesh, depsgraph)
    weights = _weights(mesh, names)
    transforms = []
    for frame in range(args.start, args.end + 1):
        scene.frame_set(frame)
        depsgraph.update()
        frame_transforms = []
        for bone_name in names:
            pose_bone = armature.pose.bones[bone_name]
            transform = pose_bone.matrix @ pose_bone.bone.matrix_local.inverted()
            frame_transforms.append(np.asarray(transform, dtype=np.float64))
        transforms.append(frame_transforms)
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        bone_transforms=np.asarray(transforms, dtype=np.float64),
        bone_names=np.asarray(names),
        rest_vertices=rest_vertices,
        weights=weights,
        frames=np.arange(args.start, args.end + 1, dtype=np.int32),
    )
    print(f"Exported {len(transforms)} frames, {len(names)} bones, {len(rest_vertices)} vertices to {output}")


if __name__ == "__main__":
    main()
