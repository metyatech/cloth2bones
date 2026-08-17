"""Generate a small, legal synthetic cross-sequence cloth benchmark.

This fixture reuses only the user's local rest mesh and clean-rig weights; it
does not ship any dataset or binary.  Four sequences share one cloth topology,
one rest shape, one body topology, and one 50-bone rig.  Their body drivers
have different temporal compositions, while the teacher cloth response is a
fixed body-region mapping with a small velocity-dependent term.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloth2bones.body_motion import apply_skinning, infer_axis_transform, rotation_matrix_to_rotvec, rotvec_to_rotation_matrix  # noqa: E402


def _kmeans(values: np.ndarray, count: int, iterations: int = 60) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    centers = [0]
    distances = np.full(len(data), np.inf, dtype=np.float64)
    for _ in range(1, count):
        distances = np.minimum(distances, np.sum((data - data[centers[-1]]) ** 2, axis=1))
        centers.append(int(np.argmax(distances)))
    centroids = data[np.asarray(centers)].copy()
    labels = np.zeros(len(data), dtype=np.int64)
    for _ in range(iterations):
        distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster in range(count):
            if np.any(labels == cluster):
                centroids[cluster] = data[labels == cluster].mean(axis=0)
    order = np.lexsort((centroids[:, 0], centroids[:, 1], centroids[:, 2]))
    remap = np.empty(count, dtype=np.int64)
    remap[order] = np.arange(count)
    return remap[labels]


def _rotation_about(point: np.ndarray, axis: np.ndarray, angle: float, translation: np.ndarray) -> np.ndarray:
    rotation = rotvec_to_rotation_matrix(np.asarray(axis, dtype=np.float64) * float(angle))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = point + translation - rotation @ point
    return transform


def _global_transform(u: float) -> np.ndarray:
    angle = 0.08 * np.sin(2.0 * np.pi * u)
    rotation = rotvec_to_rotation_matrix(np.asarray([0.0, angle, 0.0], dtype=np.float64))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray([0.10 * np.sin(2.0 * np.pi * u), 0.0, 0.04 * np.cos(2.0 * np.pi * u) - 0.04])
    return transform


def _drivers(sequence_id: str, u: float, previous_u: float) -> tuple[np.ndarray, np.ndarray]:
    if sequence_id == "seq_A_left_raise":
        pose = np.asarray([np.sin(2.0 * np.pi * u), 0.35 * np.sin(4.0 * np.pi * u), 0.25 * np.sin(np.pi * u)])
        previous = np.asarray([np.sin(2.0 * np.pi * previous_u), 0.35 * np.sin(4.0 * np.pi * previous_u), 0.25 * np.sin(np.pi * previous_u)])
    elif sequence_id == "seq_B_right_raise":
        pose = np.asarray([0.60 * np.sin(3.0 * np.pi * u), np.sin(2.0 * np.pi * u), 0.25 * np.sin(5.0 * np.pi * u)])
        previous = np.asarray([0.60 * np.sin(3.0 * np.pi * previous_u), np.sin(2.0 * np.pi * previous_u), 0.25 * np.sin(5.0 * np.pi * previous_u)])
    elif sequence_id == "seq_C_alternating_swing":
        pose = np.asarray([0.75 * np.sin(np.pi * u), -0.55 * np.sin(3.0 * np.pi * u), 0.20 * np.sin(7.0 * np.pi * u)])
        previous = np.asarray([0.75 * np.sin(np.pi * previous_u), -0.55 * np.sin(3.0 * np.pi * previous_u), 0.20 * np.sin(7.0 * np.pi * previous_u)])
    elif sequence_id == "seq_D_combined_holdout":
        pose = np.asarray([0.85 * np.sin(2.0 * np.pi * u) + 0.25 * np.sin(5.0 * np.pi * u), 0.65 * np.sin(3.0 * np.pi * u), 0.40 * np.sin(np.pi * u) ** 2])
        previous = np.asarray([0.85 * np.sin(2.0 * np.pi * previous_u) + 0.25 * np.sin(5.0 * np.pi * previous_u), 0.65 * np.sin(3.0 * np.pi * previous_u), 0.40 * np.sin(np.pi * previous_u) ** 2])
    else:
        raise ValueError(f"Unknown synthetic sequence {sequence_id}")
    return pose, pose - previous


def _make_sequence(
    sequence_id: str,
    motion_label: str,
    frames: int,
    body_rest: np.ndarray,
    body_triangles: np.ndarray,
    body_edges: np.ndarray,
    cloth_rest: np.ndarray,
    cloth_triangles: np.ndarray,
    cloth_edges: np.ndarray,
    weights: np.ndarray,
    bone_centers: np.ndarray,
    bone_regions: np.ndarray,
    region_labels: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    region_count = int(region_labels.max()) + 1
    region_centers = np.stack([body_rest[region_labels == index].mean(axis=0) for index in range(region_count)])
    body_local_frames = []
    local_bone_frames = []
    for frame in range(frames):
        u = frame / max(frames - 1, 1)
        previous_u = max(frame - 1, 0) / max(frames - 1, 1)
        pose, velocity = _drivers(sequence_id, u, previous_u)
        region_transforms = []
        for region, center in enumerate(region_centers):
            side = -1.0 if center[0] < float(np.median(region_centers[:, 0])) else 1.0
            height = (center[2] - body_rest[:, 2].min()) / max(float(np.ptp(body_rest[:, 2])), 1.0e-6)
            region_signal = side * pose[0] + (0.4 + height) * pose[1] + (region + 1) / region_count * pose[2]
            axis = np.asarray([0.25 * side, 0.9, 0.2 * (height - 0.5)], dtype=np.float64)
            translation = np.asarray(
                [0.012 * side * pose[1], 0.010 * (height - 0.5) * pose[0], 0.018 * region_signal],
                dtype=np.float64,
            )
            region_transforms.append(_rotation_about(center, axis / max(np.linalg.norm(axis), 1.0e-9), 0.16 * region_signal, translation))
        region_transforms_array = np.asarray(region_transforms)
        body_local = np.empty_like(body_rest)
        for region in range(region_count):
            mask = region_labels == region
            transform = region_transforms_array[region]
            body_local[mask] = body_rest[mask] @ transform[:3, :3].T + transform[:3, 3]
        body_local_frames.append(body_local)

        bone_transforms = np.empty((len(bone_centers), 4, 4), dtype=np.float64)
        for bone, center in enumerate(bone_centers):
            region = int(bone_regions[bone])
            other = (region + 1 + bone % max(region_count - 1, 1)) % region_count
            primary = region_transforms_array[region]
            secondary = region_transforms_array[other]
            primary_rotvec = rotation_matrix_to_rotvec(primary[:3, :3])
            secondary_rotvec = rotation_matrix_to_rotvec(secondary[:3, :3])
            primary_angle = 0.16 * (primary_rotvec[1] + 0.8 * velocity[0])
            secondary_angle = 0.16 * (secondary_rotvec[1] + 0.5 * velocity[1])
            # The sign is tied to body-side geometry, not the sequence label.
            side = -1.0 if center[0] < 0.0 else 1.0
            angle = side * (0.50 * primary_angle + 0.18 * secondary_angle)
            axis = np.asarray([0.15 * side, 0.95, 0.20], dtype=np.float64)
            translation = 0.35 * primary[:3, 3] + 0.12 * secondary[:3, 3]
            translation += np.asarray([0.004 * side * pose[1], 0.003 * velocity[2], 0.002 * pose[0]], dtype=np.float64)
            bone_transforms[bone] = _rotation_about(center, axis / np.linalg.norm(axis), angle, translation)
        local_bone_frames.append(bone_transforms)

    local_body = np.asarray(body_local_frames)
    local_bones = np.asarray(local_bone_frames)
    globals_ = np.asarray([_global_transform(frame / max(frames - 1, 1)) for frame in range(frames)])
    world_body = np.einsum("tij,tvj->tvi", globals_[:, :3, :3], local_body) + globals_[:, None, :3, 3]
    world_bones = np.einsum("tij,tbjk->tbik", globals_, local_bones)
    world_cloth = apply_skinning(cloth_rest, weights, world_bones)
    traj_vel = np.zeros_like(world_cloth, dtype=np.float32)
    traj_vel[1:] = world_cloth[1:] - world_cloth[:-1]
    collision_vel = np.zeros_like(world_body, dtype=np.float32)
    collision_vel[1:] = world_body[1:] - world_body[:-1]
    fields = {
        "initial": np.concatenate([cloth_rest, np.zeros_like(cloth_rest)], axis=1),
        "traj": world_cloth.astype(np.float32),
        "traj_vel": traj_vel,
        "triangles": cloth_triangles.astype(np.int32),
        "edges": cloth_edges.astype(np.int32),
        "collision_vertices": world_body.astype(np.float32),
        "collision_vel": collision_vel,
        "collision_triangles": body_triangles.astype(np.int32),
        "collision_edges": body_edges.astype(np.int32),
        "teacher_bone_transforms_world": world_bones,
        "teacher_bone_transforms_local_precanonical": local_bones,
        "sequence_id": np.asarray(sequence_id),
        "motion_label": np.asarray(motion_label),
        "fps": np.asarray(60.0),
    }
    metadata = {"sequence_id": sequence_id, "motion_label": motion_label, "frames": frames, "global_transform_injected": True}
    return fields, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a four-sequence synthetic cross-sequence cloth fixture")
    parser.add_argument("--reference-npz", type=Path, required=True, help="ClothTransformer-style NPZ providing common topology/body rest")
    parser.add_argument("--poses", type=Path, required=True, help="Clean-rig pose NPZ providing common rest mesh and weights")
    parser.add_argument("--out-root", type=Path, required=True, help="Output directory outside the repository")
    parser.add_argument("--frames", type=int, default=120, help="Frames per synthetic sequence")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    with np.load(args.reference_npz, allow_pickle=False) as source:
        cloth_triangles = np.asarray(source["triangles"], dtype=np.int64)
        cloth_edges = np.asarray(source["edges"], dtype=np.int64)
        body_triangles = np.asarray(source["collision_triangles"], dtype=np.int64)
        body_edges = np.asarray(source["collision_edges"], dtype=np.int64)
    with np.load(args.poses, allow_pickle=False) as pose_data:
        cloth_rest = np.asarray(pose_data["rest_vertices"], dtype=np.float64)
        weights = np.asarray(pose_data["weights"], dtype=np.float64)
        bone_names = [str(value) for value in pose_data["bone_names"].tolist()]
    with np.load(args.reference_npz, allow_pickle=False) as source:
        cloth_source_rest = np.asarray(source["initial"], dtype=np.float64)[:, :3] if "initial" in source else np.asarray(source["traj"][0], dtype=np.float64)
        source_body_rest = np.asarray(source["collision_vertices"][0], dtype=np.float64)
        axis = infer_axis_transform(cloth_source_rest, cloth_rest)
        offset = cloth_rest.mean(axis=0) - (cloth_source_rest @ axis.T).mean(axis=0)
        body_rest = source_body_rest @ axis.T + offset
        source_cloth_count = int(np.asarray(source["traj"]).shape[1])
        source_body_count = int(np.asarray(source["collision_vertices"]).shape[1])
    if len(cloth_rest) != source_cloth_count or len(body_rest) != source_body_count:
        raise ValueError("Reference NPZ and clean-rig pose mesh do not have matching vertex counts")
    region_labels = _kmeans(body_rest, 6)
    bone_mass = weights.sum(axis=0)
    bone_centers = (weights.T @ cloth_rest) / np.maximum(bone_mass[:, None], 1.0e-9)
    region_centers = np.stack([body_rest[region_labels == index].mean(axis=0) for index in range(6)])
    bone_regions = np.argmin(np.sum((bone_centers[:, None, :] - region_centers[None, :, :]) ** 2, axis=2), axis=1)
    sequences = (
        ("seq_A_left_raise", "left_raise"),
        ("seq_B_right_raise", "right_raise"),
        ("seq_C_alternating_swing", "alternating_swing"),
        ("seq_D_combined_holdout", "combined_holdout"),
    )
    manifest = {"schema_version": 1, "common_rig_bones": bone_names, "region_count": 6, "sequences": []}
    for sequence_id, motion_label in sequences:
        fields, metadata = _make_sequence(
            sequence_id,
            motion_label,
            args.frames,
            body_rest,
            body_triangles,
            body_edges,
            cloth_rest,
            cloth_triangles,
            cloth_edges,
            weights,
            bone_centers,
            bone_regions,
            region_labels,
        )
        path = args.out_root / f"{sequence_id}.npz"
        np.savez_compressed(path, **fields)
        metadata["path"] = str(path.resolve())
        manifest["sequences"].append(metadata)
    manifest["common_rig"] = {"rest_vertices": int(len(cloth_rest)), "bones": len(bone_names), "weights_shape": list(weights.shape)}
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
