"""Verify a body-local skinned FBX against the canonicalized cloth teacher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import bpy
import numpy as np


def _args() -> argparse.Namespace:
    argv = os.sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--teacher", required=True, help="NPZ containing canonicalized cloth_local vertices")
    parser.add_argument("--json", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=240)
    return parser.parse_args(argv)


def _vertices(mesh: bpy.types.Object, depsgraph: bpy.types.Depsgraph) -> np.ndarray:
    evaluated = mesh.evaluated_get(depsgraph)
    data = evaluated.to_mesh()
    try:
        return np.asarray([(evaluated.matrix_world @ vertex.co)[:] for vertex in data.vertices], dtype=np.float64)
    finally:
        evaluated.to_mesh_clear()


def _metrics(predicted: np.ndarray, teacher: np.ndarray) -> dict[str, float]:
    distance = np.linalg.norm(predicted - teacher, axis=1)
    return {
        "rms": float(np.sqrt(np.mean(distance * distance))),
        "max_point_error": float(distance.max()),
        "p95_point_error": float(np.percentile(distance, 95)),
    }


def main() -> None:
    args = _args()
    with np.load(args.teacher, allow_pickle=False) as source:
        teacher = np.asarray(source["cloth_local"], dtype=np.float64)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(Path(args.fbx).resolve()), use_anim=True, automatic_bone_orientation=False)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and any(mod.type == "ARMATURE" for mod in obj.modifiers)]
    if len(meshes) != 1:
        raise RuntimeError(f"Expected one skinned mesh, got {len(meshes)}")
    mesh = meshes[0]
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    if args.end - args.start + 1 != len(teacher):
        raise ValueError("Teacher frame count does not match requested FBX frame range")
    frames = []
    for index, frame in enumerate(range(args.start, args.end + 1)):
        scene.frame_set(frame)
        depsgraph.update()
        frames.append(_metrics(_vertices(mesh, depsgraph), teacher[index]))
    result = {
        "fbx": str(Path(args.fbx).resolve()),
        "teacher": str(Path(args.teacher).resolve()),
        "frames": len(frames),
        "vertices": len(teacher[0]),
        "mean_rms": float(np.mean([value["rms"] for value in frames])),
        "max_rms": float(np.max([value["rms"] for value in frames])),
        "max_point_error": float(np.max([value["max_point_error"] for value in frames])),
        "p95_point_error": float(np.percentile([value["max_point_error"] for value in frames], 95)),
        "samples": [{"frame": args.start + index, **frames[index]} for index in (0, 60, 120, 180, 239)],
    }
    output = Path(args.json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
