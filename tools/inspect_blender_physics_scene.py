"""Print physics-scene transform and cloth diagnostics from a Blender file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import bpy
import numpy as np


def _args() -> argparse.Namespace:
    argv = os.sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--frame2", type=int, default=120)
    return parser.parse_args(argv)


def _location(obj: bpy.types.Object | bpy.types.PoseBone) -> list[float]:
    matrix = obj.matrix_world if isinstance(obj, bpy.types.Object) else obj.matrix
    return [float(value) for value in matrix.translation]


def main() -> None:
    args = _args()
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.scene).resolve()))
    armature = bpy.data.objects.get("SemanticBody")
    cloth = bpy.data.objects.get("PhysicsTShirt")
    if armature is None or cloth is None:
        raise RuntimeError("Expected SemanticBody and PhysicsTShirt")
    for frame in (args.frame, args.frame2):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        print("FRAME", frame)
        for name in ("Upper_arm.L", "Lower_arm.L", "Upper_arm.R", "Lower_arm.R"):
            pose = armature.pose.bones[name]
            print("BONE", name, _location(pose), "matrix", [[round(float(value), 4) for value in row] for row in pose.matrix])
        for name in ("Body.UpperArm.L", "Body.LowerArm.L", "Body.UpperArm.R", "Body.LowerArm.R"):
            collider = bpy.data.objects[name]
            print("COLLIDER_MATRIX", name, [[round(float(value), 4) for value in row] for row in collider.matrix_world])
            evaluated_collider = collider.evaluated_get(bpy.context.evaluated_depsgraph_get())
            collider_mesh = evaluated_collider.to_mesh()
            try:
                collider_points = np.asarray([(evaluated_collider.matrix_world @ vertex.co)[:] for vertex in collider_mesh.vertices], dtype=np.float64)
                print("COLLIDER", name, "bbox", collider_points.min(axis=0).tolist(), collider_points.max(axis=0).tolist())
            finally:
                evaluated_collider.to_mesh_clear()
        evaluated = cloth.evaluated_get(bpy.context.evaluated_depsgraph_get())
        data = evaluated.to_mesh()
        try:
            vertices = np.asarray([(evaluated.matrix_world @ vertex.co)[:] for vertex in data.vertices], dtype=np.float64)
            print("CLOTH_BBOX", vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist())
        finally:
            evaluated.to_mesh_clear()


if __name__ == "__main__":
    main()
