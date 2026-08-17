"""Compare evaluated FBX skinning against same-index Alembic vertices."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import bpy


def point_list(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return [list(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


def diff(first, second, scale=1.0):
    if len(first) != len(second):
        raise RuntimeError(f"Vertex count mismatch: {len(first)} vs {len(second)}")
    distances = [math.sqrt(sum((first_value[i] - scale * second_value[i]) ** 2 for i in range(3))) for first_value, second_value in zip(first, second, strict=True)]
    return {"vertices": len(distances), "mean": sum(distances) / len(distances), "rms": math.sqrt(sum(value * value for value in distances) / len(distances)), "max": max(distances)}


def main():
    argv = os.sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--abc", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--frames", default="1,2,120,240")
    parser.add_argument("--abc-scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(Path(args.fbx).resolve()), use_anim=True, automatic_bone_orientation=False, ignore_leaf_bones=False, force_connect_children=False)
    rig_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and any(mod.type == "ARMATURE" for mod in obj.modifiers)]
    if len(rig_meshes) != 1:
        raise RuntimeError(f"Expected one skinned mesh, found {len(rig_meshes)}")
    rig = rig_meshes[0]
    bpy.ops.wm.alembic_import(filepath=str(Path(args.abc).resolve()), as_background_job=False, set_frame_range=False)
    cache_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj != rig and not any(mod.type == "ARMATURE" for mod in obj.modifiers)]
    if len(cache_meshes) != 1:
        raise RuntimeError(f"Expected one cache mesh, found {len(cache_meshes)}")
    cache = cache_meshes[0]
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    report = {"fbx": str(Path(args.fbx).resolve()), "abc": str(Path(args.abc).resolve()), "abc_scale": args.abc_scale, "frames": []}
    for frame in [int(value) for value in args.frames.split(",") if value.strip()]:
        scene.frame_set(frame)
        depsgraph.update()
        report["frames"].append({"frame": frame, "diff": diff(point_list(rig, depsgraph), point_list(cache, depsgraph), args.abc_scale)})
    output = Path(args.json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
