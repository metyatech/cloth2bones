"""Render teacher, body-driven prediction, rest baseline, and error panels."""

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
    parser.add_argument("--rest-fbx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--frames", default="1,31,61,91,120")
    return parser.parse_args(argv)


def _import_mesh(path: Path) -> tuple[list[bpy.types.Object], bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True, automatic_bone_orientation=False)
    objects = [obj for obj in bpy.context.scene.objects if obj not in before]
    meshes = [obj for obj in objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError(f"Expected one skinned mesh in {path}, got {len(meshes)}")
    return objects, meshes[0]


def _points(mesh: bpy.types.Object, depsgraph: bpy.types.Depsgraph) -> tuple[list[Vector], list[list[int]]]:
    evaluated = mesh.evaluated_get(depsgraph)
    data = evaluated.to_mesh()
    try:
        points = [Vector((evaluated.matrix_world @ vertex.co)[:]) for vertex in data.vertices]
        polygons = [[int(index) for index in polygon.vertices] for polygon in data.polygons]
        return points, polygons
    finally:
        evaluated.to_mesh_clear()


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def _panel(name: str, points: list[Vector], polygons: list[list[int]], offset: float, material: bpy.types.Material) -> bpy.types.Object:
    data = bpy.data.meshes.new(name + "Mesh")
    data.from_pydata([(point.x + offset, point.y, point.z) for point in points], [], polygons)
    data.update()
    data.materials.append(material)
    mesh = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(mesh)
    return mesh


def _camera(scene: bpy.types.Scene, points: list[Vector]) -> None:
    minimum = Vector([min(point[index] for point in points) for index in range(3)])
    maximum = Vector([max(point[index] for point in points) for index in range(3)])
    center = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.78, 0.5)
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
    rest_path = Path(args.rest_fbx).resolve()
    offsets = (-1.35, -0.45, 0.45)
    for frame in frames:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        imported = []
        meshes = []
        for path in (teacher_path, predicted_path, rest_path):
            objects, mesh = _import_mesh(path)
            imported.append(objects)
            meshes.append(mesh)
        scene = bpy.context.scene
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        teacher_points, polygons = _points(meshes[0], depsgraph)
        predicted_points, _ = _points(meshes[1], depsgraph)
        rest_points, _ = _points(meshes[2], depsgraph)
        for objects in imported:
            for obj in objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        teacher_material = _material("Teacher", (0.83, 0.83, 0.90, 1.0))
        predicted_material = _material("Predicted", (0.08, 0.35, 1.0, 1.0))
        rest_material = _material("RestBaseline", (0.12, 0.75, 0.32, 1.0))
        _panel("Teacher", teacher_points, polygons, offsets[0], teacher_material)
        _panel("Predicted", predicted_points, polygons, offsets[1], predicted_material)
        _panel("RestBaseline", rest_points, polygons, offsets[2], rest_material)
        errors = [(predicted_points[index] - teacher_points[index]).length for index in range(len(teacher_points))]
        error_data = bpy.data.meshes.new("ErrorMesh")
        error_data.from_pydata([(point.x + 1.35, point.y, point.z) for point in teacher_points], [], polygons)
        error_data.update()
        error_materials = (
            _material("ErrorLow", (0.10, 0.25, 0.90, 1.0)),
            _material("ErrorMedium", (0.10, 0.85, 0.40, 1.0)),
            _material("ErrorHigh", (1.0, 0.70, 0.05, 1.0)),
            _material("ErrorPeak", (1.0, 0.05, 0.03, 1.0)),
        )
        for material in error_materials:
            error_data.materials.append(material)
        maximum = max(errors) if errors else 1.0
        limits = (maximum * 0.2, maximum * 0.5, maximum * 0.8)
        for polygon in error_data.polygons:
            value = sum(errors[index] for index in polygon.vertices) / len(polygon.vertices)
            polygon.material_index = 0 if value <= limits[0] else 1 if value <= limits[1] else 2 if value <= limits[2] else 3
        error_object = bpy.data.objects.new("PredictionError", error_data)
        bpy.context.scene.collection.objects.link(error_object)
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.resolution_x = 1600
        scene.render.resolution_y = 500
        scene.render.resolution_percentage = 100
        scene.display.shading.light = "STUDIO"
        scene.display.shading.studio_light = "paint.sl"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.show_shadows = True
        scene.display.shading.show_cavity = True
        scene.display.shading.cavity_type = "WORLD"
        scene.world = bpy.data.worlds.new("CrossSequenceWorld")
        scene.world.color = (0.025, 0.025, 0.025)
        camera_data = bpy.data.cameras.new("CrossSequenceCamera")
        camera = bpy.data.objects.new("CrossSequenceCamera", camera_data)
        scene.collection.objects.link(camera)
        scene.camera = camera
        camera_data.lens = 58.0
        camera_points = [point + Vector((offsets[0], 0.0, 0.0)) for point in teacher_points]
        camera_points.extend(point + Vector((offsets[1], 0.0, 0.0)) for point in predicted_points)
        camera_points.extend(point + Vector((offsets[2], 0.0, 0.0)) for point in rest_points)
        camera_points.extend(point + Vector((1.35, 0.0, 0.0)) for point in teacher_points)
        _camera(scene, camera_points)
        scene.render.filepath = str(output / f"cross_sequence_comparison_{frame:03d}.png")
        bpy.ops.render.render(write_still=True)
        if frame != frames[-1]:
            bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.blend).resolve()))
    print(f"Rendered {len(frames)} cross-sequence comparison frames to {output}")


if __name__ == "__main__":
    main()
