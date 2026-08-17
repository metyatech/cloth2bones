"""Build a timeline-playable teacher/predicted/rest comparison Blend file."""

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
    parser.add_argument("--teacher-npz", required=True)
    parser.add_argument("--predicted-fbx", required=True)
    parser.add_argument("--rest-fbx", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def _teacher_mesh(trajectory: np.ndarray, triangles: np.ndarray) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("PhysicsTeacherTimelineMesh")
    mesh.from_pydata(trajectory[0].tolist(), [], triangles.tolist())
    mesh.update()
    obj = bpy.data.objects.new("TeacherPhysics", mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.materials.append(_material("TeacherPhysicsMaterial", (0.80, 0.80, 0.88, 1.0)))
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
    return obj


def _import_fbx(path: Path, name: str, offset: float, material: bpy.types.Material) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True, automatic_bone_orientation=False)
    objects = [obj for obj in bpy.context.scene.objects if obj not in before]
    for obj in objects:
        obj.name = f"{name}_{obj.name}"
        obj.location.x += offset
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(material)
    return objects


def main() -> None:
    args = _args()
    with np.load(args.teacher_npz, allow_pickle=False) as source:
        trajectory = np.asarray(source["traj"], dtype=np.float64)
        triangles = np.asarray(source["triangles"], dtype=np.int32)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(trajectory)
    scene.render.fps = 30
    _teacher_mesh(trajectory, triangles).location.x = -1.35
    _import_fbx(Path(args.predicted_fbx).resolve(), "PredictedResearch", -0.45, _material("PredictedResearchMaterial", (0.08, 0.35, 1.0, 1.0)))
    _import_fbx(Path(args.rest_fbx).resolve(), "RestBaseline", 0.45, _material("RestBaselineMaterial", (0.12, 0.75, 0.32, 1.0)))
    world = bpy.data.worlds.new("PhysicsComparisonWorld")
    world.color = (0.025, 0.025, 0.025)
    scene.world = world
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "paint.sl"
    scene.display.shading.color_type = "MATERIAL"
    camera_data = bpy.data.cameras.new("PhysicsComparisonCamera")
    camera = bpy.data.objects.new("PhysicsComparisonCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (2.3, -4.2, 2.1)
    camera.rotation_euler = ((1.05, 0.0, 0.0))
    scene.camera = camera
    camera_data.lens = 58.0
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(f"Saved timeline-playable physics comparison to {output}")


if __name__ == "__main__":
    main()
