"""Render a few posed frames of a skinned FBX for visual QA."""

from __future__ import annotations

import os
from pathlib import Path

import bpy
from mathutils import Vector


def _args_after_separator():
    argv = os.sys.argv
    return argv[argv.index("--") + 1 :] if "--" in argv else []


def _evaluated_points(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return [obj.matrix_world @ vertex.co for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


def main():
    args = _args_after_separator()
    fbx = Path(args[args.index("--fbx") + 1]).resolve()
    output_dir = Path(args[args.index("--out") + 1]).resolve()
    frames = [int(value) for value in args[args.index("--frames") + 1].split(",")]
    output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx), use_anim=True, automatic_bone_orientation=False)
    mesh = next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "paint.sl"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.world = bpy.data.worlds.new("PreviewWorld")
    scene.world.color = (0.035, 0.035, 0.035)
    camera_data = bpy.data.cameras.new("PreviewCamera")
    camera = bpy.data.objects.new("PreviewCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = 58.0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for frame in frames:
        scene.frame_set(frame)
        depsgraph.update()
        points = _evaluated_points(mesh, depsgraph)
        minimum = Vector([min(point[i] for point in points) for i in range(3)])
        maximum = Vector([max(point[i] for point in points) for i in range(3)])
        center = (minimum + maximum) * 0.5
        radius = max((maximum - minimum).length * 0.75, 0.5)
        camera.location = center + Vector((radius * 1.8, -radius * 2.4, radius * 1.1))
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(output_dir / f"cloth_frame_{frame:03d}.png")
        bpy.ops.render.render(write_still=True)
    print(f"Rendered preview frames to {output_dir}")


if __name__ == "__main__":
    main()
