"""Small Blender-side acceptance check for an exported clean rig."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def bbox(vertices):
    points = [Vector(vertex) for vertex in vertices]
    if not points:
        raise RuntimeError("The evaluated mesh has no vertices")
    return {"count": len(points), "min": [min(point[i] for point in points) for i in range(3)], "max": [max(point[i] for point in points) for i in range(3)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--frames", default="1,120,240")
    parser.add_argument("--expected-bones", type=int, required=True)
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    args = parser.parse_args(argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(Path(args.fbx).resolve()), automatic_bone_orientation=False)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(meshes) != 1 or len(armatures) != 1:
        raise RuntimeError(f"Expected one mesh and one armature, got {len(meshes)} and {len(armatures)}")
    mesh = meshes[0]
    armature = armatures[0]
    if len(armature.data.bones) != args.expected_bones:
        raise RuntimeError(f"Expected {args.expected_bones} bones, got {len(armature.data.bones)}")
    if not mesh.vertex_groups:
        raise RuntimeError("The mesh has no vertex groups")
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result = {"fbx": str(Path(args.fbx).resolve()), "mesh_vertices": len(mesh.data.vertices), "mesh_polygons": len(mesh.data.polygons), "bones": len(armature.data.bones), "mesh_matrix_world": [list(row) for row in mesh.matrix_world], "armature_matrix_world": [list(row) for row in armature.matrix_world], "frames": []}
    for frame_text in args.frames.split(","):
        frame = int(frame_text)
        scene.frame_set(frame)
        depsgraph.update()
        evaluated = mesh.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            points = [mesh.matrix_world @ vertex.co for vertex in evaluated_mesh.vertices]
            values = [component for point in points for component in point]
            if not values or not all(math.isfinite(float(value)) for value in values):
                raise RuntimeError(f"Non-finite evaluated vertices at frame {frame}")
            result["frames"].append({"frame": frame, "bbox": bbox(points)})
        finally:
            evaluated.to_mesh_clear()
    output = Path(args.json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
