"""Render teacher/predicted/error panels for a body-local cloth PoC."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import bpy
from mathutils import Vector


def _args() -> argparse.Namespace:
    argv = os.sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-fbx", required=True)
    parser.add_argument("--predicted-fbx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--frames", default="1,61,121,181,240")
    return parser.parse_args(argv)


def _import_group(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True, automatic_bone_orientation=False)
    return [obj for obj in bpy.context.scene.objects if obj not in before]


def _skinned_mesh(objects: list[bpy.types.Object]) -> bpy.types.Object:
    meshes = [obj for obj in objects if obj.type == "MESH" and any(mod.type == "ARMATURE" for mod in obj.modifiers)]
    if len(meshes) != 1:
        raise RuntimeError(f"Expected one skinned mesh, got {len(meshes)}")
    return meshes[0]


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def _set_mesh_material(mesh: bpy.types.Object, material: bpy.types.Material) -> None:
    mesh.data.materials.clear()
    mesh.data.materials.append(material)


def _points(mesh: bpy.types.Object, depsgraph: bpy.types.Depsgraph):
    evaluated = mesh.evaluated_get(depsgraph)
    data = evaluated.to_mesh()
    try:
        points = [(evaluated.matrix_world @ vertex.co)[:] for vertex in data.vertices]
        polygons = [[int(index) for index in polygon.vertices] for polygon in data.polygons]
        return points, polygons
    finally:
        evaluated.to_mesh_clear()


def _error_mesh(name: str, points, polygons, offset: float, errors):
    data = bpy.data.meshes.new(name + "Mesh")
    shifted = [(point[0] + offset, point[1], point[2]) for point in points]
    data.from_pydata(shifted, [], polygons)
    data.update()
    mesh = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(mesh)
    colors = (
        _material(name + "Low", (0.1, 0.35, 1.0, 1.0)),
        _material(name + "Medium", (0.1, 0.9, 0.45, 1.0)),
        _material(name + "High", (1.0, 0.75, 0.05, 1.0)),
        _material(name + "Peak", (1.0, 0.08, 0.04, 1.0)),
    )
    for material in colors:
        data.materials.append(material)
    maximum = max(errors)
    limits = (maximum * 0.2, maximum * 0.5, maximum * 0.8)
    for polygon in data.polygons:
        value = float(sum(errors[index] for index in polygon.vertices) / len(polygon.vertices))
        polygon.material_index = 0 if value <= limits[0] else 1 if value <= limits[1] else 2 if value <= limits[2] else 3
    return mesh


def _static_panel(name: str, points, polygons, offset: float, material: bpy.types.Material) -> bpy.types.Object:
    data = bpy.data.meshes.new(name + "Mesh")
    shifted = [(point[0] + offset, point[1], point[2]) for point in points]
    data.from_pydata(shifted, [], polygons)
    data.update()
    data.materials.append(material)
    mesh = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(mesh)
    return mesh


def _camera_for_points(scene: bpy.types.Scene, points: list[Vector]) -> None:
    minimum = Vector([min(point[index] for point in points) for index in range(3)])
    maximum = Vector([max(point[index] for point in points) for index in range(3)])
    center = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.72, 0.5)
    camera = scene.camera
    camera.location = center + Vector((radius * 1.65, -radius * 2.4, radius * 1.0))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    args = _args()
    frames = [int(value) for value in args.frames.split(",")]
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher_path = Path(args.teacher_fbx).resolve()
    predicted_path = Path(args.predicted_fbx).resolve()
    offset = 0.9
    for frame in frames:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        teacher_objects = _import_group(teacher_path)
        teacher_mesh = _skinned_mesh(teacher_objects)
        scene = bpy.context.scene
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        teacher_points, polygons = _points(teacher_mesh, depsgraph)
        for obj in teacher_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        predicted_objects = _import_group(predicted_path)
        predicted_mesh = _skinned_mesh(predicted_objects)
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        predicted_points, _ = _points(predicted_mesh, depsgraph)
        for obj in predicted_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        teacher_material = _material("TeacherCloth", (0.85, 0.85, 0.9, 1.0))
        predicted_material = _material("PredictedCloth", (0.08, 0.35, 1.0, 1.0))
        _static_panel("TeacherPanel", teacher_points, polygons, -offset, teacher_material)
        _static_panel("PredictedPanel", predicted_points, polygons, 0.0, predicted_material)
        errors = [(Vector(predicted_points[index]) - Vector(teacher_points[index])).length for index in range(len(teacher_points))]
        _error_mesh("ErrorPanel", teacher_points, polygons, offset * 2.0, errors)
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.resolution_x = 1200
        scene.render.resolution_y = 500
        scene.render.resolution_percentage = 100
        scene.display.shading.light = "STUDIO"
        scene.display.shading.studio_light = "paint.sl"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.show_shadows = True
        scene.display.shading.show_cavity = True
        scene.display.shading.cavity_type = "WORLD"
        scene.world = bpy.data.worlds.new("BodyLocalPreviewWorld")
        scene.world.color = (0.025, 0.025, 0.025)
        camera_data = bpy.data.cameras.new("BodyLocalPreviewCamera")
        camera = bpy.data.objects.new("BodyLocalPreviewCamera", camera_data)
        scene.collection.objects.link(camera)
        scene.camera = camera
        camera_data.lens = 58.0
        all_points = [Vector(point) + Vector((-offset, 0.0, 0.0)) for point in teacher_points]
        all_points.extend(Vector(point) for point in predicted_points)
        all_points.extend(Vector(point) + Vector((offset * 2.0, 0.0, 0.0)) for point in teacher_points)
        _camera_for_points(scene, all_points)
        scene.render.filepath = str(output / f"body_local_comparison_{frame:03d}.png")
        bpy.ops.render.render(write_still=True)
        if frame != frames[-1]:
            bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.blend).resolve()))
    print(f"Rendered {len(frames)} body-local comparison frames to {output}")


if __name__ == "__main__":
    main()
