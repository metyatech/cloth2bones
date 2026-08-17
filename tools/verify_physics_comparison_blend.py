"""Verify that the PoC 3.3 comparison Blend contains animated panels."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def _bbox(obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph) -> tuple[list[float], list[float]]:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        points = np.asarray([(evaluated.matrix_world @ vertex.co)[:] for vertex in mesh.vertices], dtype=np.float64)
        return points.min(axis=0).tolist(), points.max(axis=0).tolist()
    finally:
        evaluated.to_mesh_clear()


def main() -> None:
    args = _args()
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()))
    teacher = bpy.data.objects.get("TeacherPhysics")
    predicted = [obj for obj in bpy.context.scene.objects if obj.name.startswith("PredictedResearch_") and obj.type == "MESH"]
    rest = [obj for obj in bpy.context.scene.objects if obj.name.startswith("RestBaseline_") and obj.type == "MESH"]
    if teacher is None or len(predicted) != 1 or len(rest) != 1:
        raise RuntimeError("Comparison blend must contain one teacher, predicted, and rest mesh")
    samples = []
    for frame in (1, 61, 120):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        key_blocks = teacher.data.shape_keys.key_blocks if teacher.data.shape_keys else []
        active_keys = sum(float(block.value) > 0.5 for block in key_blocks[1:])
        samples.append({"frame": frame, "teacher_bbox": _bbox(teacher, bpy.context.evaluated_depsgraph_get()), "predicted_bbox": _bbox(predicted[0], bpy.context.evaluated_depsgraph_get()), "rest_bbox": _bbox(rest[0], bpy.context.evaluated_depsgraph_get()), "active_teacher_shape_keys": active_keys})
    result = {"blend": str(Path(args.blend).resolve()), "samples": samples, "timeline_frames": [bpy.context.scene.frame_start, bpy.context.scene.frame_end]}
    Path(args.out).resolve().write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
