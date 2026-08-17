"""Build a clean Blender armature from Dem Bones weights and an Alembic cache."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import suppress
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def world_points(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return np.asarray([[float(x) for x in (evaluated.matrix_world @ v.co)] for v in mesh.vertices], dtype=np.float64)
    finally:
        evaluated.to_mesh_clear()


def get_weights(mesh_obj, bone_names):
    count = len(mesh_obj.data.vertices)
    weights = np.zeros((count, len(bone_names)), dtype=np.float64)
    missing = []
    for bone_index, name in enumerate(bone_names):
        group = mesh_obj.vertex_groups.get(name)
        if group is None:
            missing.append(name)
            continue
        for vertex_index in range(count):
            with suppress(RuntimeError):
                weights[vertex_index, bone_index] = group.weight(vertex_index)
    sums = weights.sum(axis=1)
    zero_count = int(np.count_nonzero(sums <= 1.0e-12))
    sums[sums <= 1.0e-12] = 1.0
    return weights / sums[:, None], missing, zero_count


def rigid_fit(target, rest, weights, initial, passes):
    transforms = [np.array(transform, dtype=np.float64, copy=True) for transform in initial]
    for _ in range(passes):
        for bone_index in range(weights.shape[1]):
            weight = weights[:, bone_index]
            active = weight > 1.0e-6
            if int(np.count_nonzero(active)) < 3:
                continue
            other = np.zeros_like(target)
            for other_index, transform in enumerate(transforms):
                if other_index != bone_index:
                    other += weights[:, other_index, None] * (rest @ transform[:3, :3].T + transform[:3, 3])
            target_for_bone = (target[active] - other[active]) / weight[active, None]
            source_for_bone = rest[active]
            alpha = weight[active] * weight[active]
            total = float(alpha.sum())
            source_center = np.sum(source_for_bone * alpha[:, None], axis=0) / total
            target_center = np.sum(target_for_bone * alpha[:, None], axis=0) / total
            covariance = ((source_for_bone - source_center) * alpha[:, None]).T @ (target_for_bone - target_center)
            u, _, vt = np.linalg.svd(covariance)
            rotation = vt.T @ u.T
            if np.linalg.det(rotation) < 0.0:
                vt[-1, :] *= -1.0
                rotation = vt.T @ u.T
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation
            transform[:3, 3] = target_center - rotation @ source_center
            transforms[bone_index] = transform
    predicted = np.zeros_like(target)
    for bone_index, transform in enumerate(transforms):
        predicted += weights[:, bone_index, None] * (rest @ transform[:3, :3].T + transform[:3, 3])
    distances = np.linalg.norm(target - predicted, axis=1)
    return transforms, {
        "rms": float(np.sqrt(np.mean(distances * distances))),
        "mean": float(np.mean(distances)),
        "max": float(np.max(distances)),
    }


def apply_world_skin_transform(pose_bone, transform):
    pose_bone.rotation_mode = "QUATERNION"
    pose_bone.matrix = Matrix(transform.tolist()) @ pose_bone.bone.matrix_local


def actual_metrics(mesh_obj, target, depsgraph):
    actual = world_points(mesh_obj, depsgraph)
    distances = np.linalg.norm(target - actual, axis=1)
    return {"rms": float(np.sqrt(np.mean(distances * distances))), "mean": float(np.mean(distances)), "max": float(np.max(distances))}


def main():
    argv = os.sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem-fbx", required=True)
    parser.add_argument("--abc", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=240)
    parser.add_argument("--passes", type=int, default=8)
    parser.add_argument("--export-global-scale", type=float, default=1.0)
    parser.add_argument("--export-content-scale", type=float, default=1.0)
    parser.add_argument("--export-apply-unit-scale", action="store_true")
    parser.add_argument("--export-unit-scale-length", type=float, default=1.0)
    args = parser.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(Path(args.dem_fbx).resolve()), use_anim=True, automatic_bone_orientation=False, ignore_leaf_bones=False, force_connect_children=False)
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    source_armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    source_meshes = [obj for obj in scene.objects if obj.type == "MESH" and any(mod.type == "ARMATURE" for mod in obj.modifiers)]
    if len(source_armatures) != 1 or len(source_meshes) != 1:
        raise RuntimeError(f"Expected one Dem Bones armature/mesh, found {len(source_armatures)}/{len(source_meshes)}")
    source_armature = source_armatures[0]
    source_mesh = source_meshes[0]
    source_bones = list(source_armature.data.bones)
    if any(bone.parent is not None for bone in source_bones):
        raise RuntimeError("Use a flat Dem Bones output; grouped output is intentionally rejected")
    bone_names = [bone.name for bone in source_bones]
    weights, missing_groups, zero_weight_vertices = get_weights(source_mesh, bone_names)

    bpy.ops.wm.alembic_import(filepath=str(Path(args.abc).resolve()), as_background_job=False, set_frame_range=False)
    cache_objects = [obj for obj in scene.objects if obj.type == "MESH" and obj != source_mesh and not any(mod.type == "ARMATURE" for mod in obj.modifiers)]
    if len(cache_objects) != 1:
        raise RuntimeError(f"Expected one Alembic mesh, found {len(cache_objects)}")
    cache = cache_objects[0]
    scene.frame_set(args.start)
    depsgraph.update()
    rest = world_points(cache, depsgraph)
    if rest.shape[0] != len(source_mesh.data.vertices):
        raise RuntimeError(f"Rest vertex mismatch: cache={rest.shape[0]} weights={len(source_mesh.data.vertices)}")
    cache_eval_obj = cache.evaluated_get(depsgraph)
    cache_eval = cache_eval_obj.to_mesh()
    try:
        polygons = [[int(index) for index in polygon.vertices] for polygon in cache_eval.polygons]
    finally:
        cache_eval_obj.to_mesh_clear()

    for obj in list(scene.objects):
        if obj != cache:
            bpy.data.objects.remove(obj, do_unlink=True)
    for old_action in list(bpy.data.actions):
        bpy.data.actions.remove(old_action)
    mesh_data = bpy.data.meshes.new("cloth_clean_mesh")
    mesh_data.from_pydata(rest.tolist(), [], polygons)
    mesh_data.update()
    mesh_obj = bpy.data.objects.new("cloth_clean", mesh_data)
    scene.collection.objects.link(mesh_obj)
    armature_data = bpy.data.armatures.new("cloth_clean_armature")
    armature = bpy.data.objects.new("cloth_clean_armature", armature_data)
    scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for bone_index, name in enumerate(bone_names):
        edit_bone = armature_data.edit_bones.new(name)
        weight = weights[:, bone_index]
        center = (rest * weight[:, None]).sum(axis=0) / max(float(weight.sum()), 1.0e-12)
        edit_bone.head = center.tolist()
        edit_bone.tail = (center + np.asarray([0.0, 0.0, 0.05])).tolist()
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.select_set(False)
    for name in bone_names:
        mesh_obj.vertex_groups.new(name=name)
    for bone_index, name in enumerate(bone_names):
        group = mesh_obj.vertex_groups[name]
        for vertex_index, value in enumerate(weights[:, bone_index]):
            if value > 1.0e-8:
                group.add([vertex_index], float(value), "REPLACE")
    modifier = mesh_obj.modifiers.new("cloth_clean_armature", "ARMATURE")
    modifier.object = armature
    mesh_obj.parent = armature
    mesh_obj.parent_type = "OBJECT"

    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    action = bpy.data.actions.new("demBones_Clean")
    armature.animation_data_create()
    armature.animation_data.action = action
    scene.frame_start = args.start
    scene.frame_end = args.end
    scene.render.fps = 24
    report = {"dem_fbx": str(Path(args.dem_fbx).resolve()), "abc": str(Path(args.abc).resolve()), "bones": len(bone_names), "vertices": len(rest), "missing_vertex_groups": missing_groups, "zero_weight_vertices": zero_weight_vertices, "action": action.name, "frames": []}
    previous = [np.eye(4, dtype=np.float64) for _ in bone_names]
    for frame in range(args.start, args.end + 1):
        scene.frame_set(frame)
        depsgraph.update()
        target = world_points(cache, depsgraph)
        if frame == args.start:
            transforms = [np.eye(4, dtype=np.float64) for _ in bone_names]
            fit = {"rms": 0.0, "mean": 0.0, "max": 0.0}
        else:
            transforms, fit = rigid_fit(target, rest, weights, previous, args.passes)
        previous = transforms
        for name, transform in zip(bone_names, transforms, strict=True):
            pose_bone = armature.pose.bones[name]
            apply_world_skin_transform(pose_bone, transform)
            pose_bone.keyframe_insert(data_path="location", frame=frame, group=name)
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=name)
            pose_bone.keyframe_insert(data_path="scale", frame=frame, group=name)
        depsgraph.update()
        report["frames"].append({"frame": frame, "fit": fit, "actual": actual_metrics(mesh_obj, target, depsgraph)})
    bpy.data.objects.remove(cache, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    mesh_obj.select_set(True)
    scene.view_layers[0].objects.active = armature
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.export_content_scale != 1.0:
        armature.scale = (args.export_content_scale,) * 3
    scene.unit_settings.scale_length = args.export_unit_scale_length
    bpy.ops.export_scene.fbx(filepath=str(output), use_selection=True, object_types={"ARMATURE", "MESH"}, apply_unit_scale=args.export_apply_unit_scale, global_scale=args.export_global_scale, apply_scale_options="FBX_SCALE_NONE", bake_space_transform=False, axis_forward="-Z", axis_up="Y", add_leaf_bones=False, use_armature_deform_only=True, mesh_smooth_type="OFF")
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report_path), "bones": len(bone_names), "frames": len(report["frames"])}))


if __name__ == "__main__":
    main()
