"""PoC 3.1: predict local cloth-bone motion from canonicalized body motion.

The script deliberately uses only NumPy and the body collider stream.  It
removes the frame-wise rigid body transform before fitting any model, keeps
frame/time out of the feature matrix, and evaluates several splits and model
families.  It writes research artifacts outside the repository when invoked
with an explicit ``--out-root``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloth2bones.body_motion import (  # noqa: E402
    apply_skinning,
    infer_axis_transform,
    invert_rigid_transforms,
    kabsch_transform,
    rotation_matrix_to_rotvec,
    rotvec_to_rotation_matrix,
    transform_points,
)


@dataclass(frozen=True)
class RegionLayout:
    names: tuple[str, ...]
    labels: np.ndarray
    masks: tuple[np.ndarray, ...]
    centroids: np.ndarray
    connected_components: tuple[int, ...]


def _kmeans(values: np.ndarray, count: int, iterations: int = 40) -> np.ndarray:
    """Deterministic farthest-point initialized k-means."""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or count < 1 or count > len(data):
        raise ValueError(f"Invalid k-means input {data.shape} and count {count}")
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
            members = data[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
    return labels


def _zscore(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    scale = data.std(axis=0)
    scale[scale < 1.0e-9] = 1.0
    return (data - data.mean(axis=0)) / scale


def _connected_component_count(labels: np.ndarray, edges: np.ndarray, region: int) -> int:
    """Count adjacency components for one region without changing labels."""

    members = np.flatnonzero(labels == region)
    if not len(members):
        return 0
    member_set = set(int(value) for value in members)
    adjacency: dict[int, list[int]] = {int(value): [] for value in members}
    for start, end in np.asarray(edges, dtype=np.int64):
        first, second = int(start), int(end)
        if first in member_set and second in member_set:
            adjacency[first].append(second)
            adjacency[second].append(first)
    visited: set[int] = set()
    components = 0
    for start in members:
        vertex = int(start)
        if vertex in visited:
            continue
        components += 1
        stack = [vertex]
        visited.add(vertex)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    return components


def auto_regions(body_local: np.ndarray, edges: np.ndarray, count: int = 6) -> RegionLayout:
    """Cluster rest-space position and canonical residual motion into regions."""

    rest = np.asarray(body_local[0], dtype=np.float64)
    displacement = np.asarray(body_local, dtype=np.float64) - rest[None, ...]
    sampled = displacement[:: max(1, len(displacement) // 32)].transpose(1, 0, 2).reshape(len(rest), -1)
    centered = sampled - sampled.mean(axis=0)
    left_singular, singular, _ = np.linalg.svd(centered, full_matrices=False)
    component_count = min(4, left_singular.shape[1])
    motion_scores = left_singular[:, :component_count] * singular[:component_count]
    features = np.concatenate([_zscore(rest), _zscore(motion_scores)], axis=1)
    labels = _kmeans(features, count)
    masks = tuple(labels == region for region in range(count))
    if any(int(mask.sum()) < 6 for mask in masks):
        raise ValueError(f"Automatic region clustering produced a tiny region: {[int(mask.sum()) for mask in masks]}")
    centroids = np.stack([rest[mask].mean(axis=0) for mask in masks], axis=0)
    order = np.lexsort((centroids[:, 0], centroids[:, 1], centroids[:, 2]))
    remap = np.empty(count, dtype=np.int64)
    remap[order] = np.arange(count)
    labels = remap[labels]
    masks = tuple(labels == region for region in range(count))
    centroids = np.stack([rest[mask].mean(axis=0) for mask in masks], axis=0)
    names = tuple(f"auto_region_{index:02d}" for index in range(count))
    components = tuple(_connected_component_count(labels, edges, index) for index in range(count))
    return RegionLayout(names=names, labels=labels, masks=masks, centroids=centroids, connected_components=components)


def _fit_region_transforms(rest: np.ndarray, frames: np.ndarray, layout: RegionLayout) -> np.ndarray:
    transforms = np.empty((len(frames), len(layout.masks), 4, 4), dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        for region_index, mask in enumerate(layout.masks):
            transforms[frame_index, region_index] = kabsch_transform(rest[mask], frame[mask])
    return transforms


def _region_features(rest: np.ndarray, frames: np.ndarray, layout: RegionLayout) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return rotvec/translation/centroid/scale features and region transforms."""

    transforms = _fit_region_transforms(rest, frames, layout)
    feature_blocks: list[np.ndarray] = []
    names: list[str] = []
    for region_index, mask in enumerate(layout.masks):
        source = rest[mask]
        source_zero = source - source.mean(axis=0)
        source_inverse = np.linalg.pinv(source_zero.T @ source_zero + np.eye(3) * 1.0e-9)
        blocks = []
        for frame_index, frame in enumerate(frames):
            target = frame[mask]
            transform = transforms[frame_index, region_index]
            rotvec = rotation_matrix_to_rotvec(transform[:3, :3])
            translation = transform[:3, 3]
            centroid_delta = target.mean(axis=0) - source.mean(axis=0)
            target_zero = target - target.mean(axis=0)
            linear = target_zero.T @ source_zero @ source_inverse
            scale_shear = np.log(np.maximum(np.linalg.svd(linear, compute_uv=False), 1.0e-9))
            blocks.append(np.concatenate([rotvec, translation, centroid_delta, scale_shear]))
        feature_blocks.append(np.asarray(blocks, dtype=np.float64))
        names.extend(
            [
                f"{layout.names[region_index]}.rotation_vector.x",
                f"{layout.names[region_index]}.rotation_vector.y",
                f"{layout.names[region_index]}.rotation_vector.z",
                f"{layout.names[region_index]}.translation.x",
                f"{layout.names[region_index]}.translation.y",
                f"{layout.names[region_index]}.translation.z",
                f"{layout.names[region_index]}.centroid_delta.x",
                f"{layout.names[region_index]}.centroid_delta.y",
                f"{layout.names[region_index]}.centroid_delta.z",
                f"{layout.names[region_index]}.log_scale.x",
                f"{layout.names[region_index]}.log_scale.y",
                f"{layout.names[region_index]}.log_scale.z",
            ]
        )
    return np.concatenate(feature_blocks, axis=1), names, transforms


def _target_vectors(transforms: np.ndarray) -> np.ndarray:
    rotations = rotation_matrix_to_rotvec(transforms[:, :, :3, :3])
    translations = transforms[:, :, :3, 3]
    return np.concatenate([rotations, translations], axis=2).reshape(len(transforms), -1)


def _vectors_to_transforms(vectors: np.ndarray, bone_count: int) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64).reshape(len(vectors), bone_count, 6)
    result = np.tile(np.eye(4, dtype=np.float64), (len(values), bone_count, 1, 1))
    result[:, :, :3, :3] = rotvec_to_rotation_matrix(values[:, :, :3])
    result[:, :, :3, 3] = values[:, :, 3:]
    return result


def _fit_standardized(features: np.ndarray, targets: np.ndarray, train: np.ndarray, alpha: float, feature_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = features if feature_indices is None else features[:, feature_indices]
    mean = selected[train].mean(axis=0)
    scale = selected[train].std(axis=0)
    scale[scale < 1.0e-9] = 1.0
    normalized = (selected - mean) / scale
    design = np.concatenate([np.ones((len(train), 1)), normalized[train]], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * alpha
    regularizer[0, 0] = 0.0
    left = design.T @ design + regularizer
    right = design.T @ targets[train]
    coefficients = np.linalg.solve(left, right)
    return normalized, coefficients, mean


def _predict_standardized(normalized: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((len(normalized), 1)), normalized], axis=1) @ coefficients


def _fit_model(name: str, features: np.ndarray, targets: np.ndarray, train: np.ndarray, bone_regions: np.ndarray, alpha: float) -> np.ndarray:
    """Fit one model without exposing frame identity to the model."""

    if name in ("linear", "ridge", "polynomial"):
        normalized, coefficients, _ = _fit_standardized(features, targets, train, 1.0e-8 if name == "linear" else alpha)
        if name == "polynomial":
            expanded = np.concatenate([normalized, normalized * normalized], axis=1)
            fit, coefficients, _ = _fit_standardized(expanded, targets, train, alpha)
            return _predict_standardized(fit, coefficients)
        return _predict_standardized(normalized, coefficients)
    if name == "local-ridge":
        predictions = np.empty_like(targets)
        block_size = features.shape[1] // int(np.max(bone_regions) + 1)
        for bone_index, region in enumerate(bone_regions):
            columns = np.arange(region * block_size, (region + 1) * block_size, dtype=np.int64)
            normalized, coefficients, _ = _fit_standardized(features, targets[:, bone_index * 6 : bone_index * 6 + 6], train, alpha, columns)
            predictions[:, bone_index * 6 : bone_index * 6 + 6] = _predict_standardized(normalized, coefficients)
        return predictions
    if name in ("nearest", "rbf"):
        normalized_train = _zscore(features[train])
        normalized_all = (features - features[train].mean(axis=0)) / np.where(features[train].std(axis=0) < 1.0e-9, 1.0, features[train].std(axis=0))
        distances = np.sqrt(np.maximum(np.sum((normalized_all[:, None, :] - normalized_train[None, :, :]) ** 2, axis=2), 0.0))
        if name == "nearest":
            return targets[train[np.argmin(distances, axis=1)]]
        bandwidth = float(np.median(distances[distances > 1.0e-9])) if np.any(distances > 1.0e-9) else 1.0
        kernel = np.exp(-0.5 * (distances / max(bandwidth, 1.0e-6)) ** 2)
        fit_kernel = kernel[train]
        regularizer = np.eye(len(train), dtype=np.float64) * alpha
        coefficients = np.linalg.solve(fit_kernel + regularizer, targets[train])
        return kernel @ coefficients
    raise ValueError(f"Unknown model {name}")


def _split_indices(frame_count: int, name: str, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    indices = np.arange(frame_count, dtype=np.int64)
    if name == "contiguous_1_180":
        return indices[: min(180, frame_count - 1)], indices[min(180, frame_count - 1) :], {"type": name}
    if name in ("interleaved_5", "interleaved_10"):
        period = int(name.rsplit("_", 1)[1])
        validation = indices[(indices + 1) % period == 0]
        train = np.setdiff1d(indices, validation)
        return train, validation, {"type": name, "period": period}
    if name == "motion_space":
        labels = _kmeans(_zscore(features), min(4, frame_count))
        sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
        heldout_cluster = int(np.argmax(sizes))
        validation = indices[labels == heldout_cluster]
        train = indices[labels != heldout_cluster]
        return train, validation, {"type": name, "heldout_cluster": heldout_cluster, "cluster_sizes": sizes.tolist()}
    raise ValueError(f"Unknown split {name}")


def _metrics(predicted: np.ndarray, teacher: np.ndarray, indices: np.ndarray, include_per_frame: bool = False) -> dict[str, object]:
    errors = np.linalg.norm(predicted[indices] - teacher[indices], axis=2)
    frame_rms = np.sqrt(np.mean(errors * errors, axis=1))
    frame_max = np.max(errors, axis=1)
    result: dict[str, object] = {
        "frames": int(len(indices)),
        "mean_rms": float(frame_rms.mean()),
        "max_rms": float(frame_rms.max()),
        "max_point_error": float(frame_max.max()),
        "p95_point_error": float(np.percentile(errors, 95)),
        "mean_point_error": float(errors.mean()),
    }
    if include_per_frame:
        result["per_frame"] = [
            {"frame": int(frame + 1), "rms": float(frame_rms[position]), "max_point_error": float(frame_max[position])}
            for position, frame in enumerate(indices)
        ]
    return result


def _bone_subset(weights: np.ndarray, rest_vertices: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    bone_count = weights.shape[1]
    if count >= bone_count:
        return np.arange(bone_count, dtype=np.int64), weights.copy()
    mass = weights.sum(axis=0)
    centers = (weights.T @ rest_vertices) / np.maximum(mass[:, None], 1.0e-9)
    selected = [int(np.argmax(mass))]
    while len(selected) < count:
        distance = np.min(np.sum((centers[:, None, :] - centers[np.asarray(selected)][None, :, :]) ** 2, axis=2), axis=1)
        distance[np.asarray(selected)] = -1.0
        selected.append(int(np.argmax(distance)))
    selected_array = np.asarray(selected, dtype=np.int64)
    reduced = np.zeros((len(weights), count), dtype=np.float64)
    nearest = np.argmin(np.sum((centers[:, None, :] - centers[selected_array][None, :, :]) ** 2, axis=2), axis=1)
    for original in range(bone_count):
        reduced[:, nearest[original]] += weights[:, original]
    reduced /= np.maximum(reduced.sum(axis=1, keepdims=True), 1.0e-12)
    return selected_array, reduced


def _bone_region_assignment(rest_vertices: np.ndarray, weights: np.ndarray, layout: RegionLayout) -> np.ndarray:
    mass = weights.sum(axis=0)
    centers = (weights.T @ rest_vertices) / np.maximum(mass[:, None], 1.0e-9)
    return np.argmin(np.sum((centers[:, None, :] - layout.centroids[None, :, :]) ** 2, axis=2), axis=1)


def _serialize_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="PoC 3.1: body-local motion to cloth helper-bone transforms")
    parser.add_argument("--npz", type=Path, required=True, help="ClothTransformer-style NPZ")
    parser.add_argument("--poses", type=Path, required=True, help="Blender-exported clean-rig pose NPZ")
    parser.add_argument("--out-root", type=Path, required=True, help="Artifact directory outside the source repository")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge/RBF regularization")
    parser.add_argument("--regions", type=int, default=6, help="Number of automatically discovered body regions")
    parser.add_argument("--selected-model", choices=("linear", "ridge", "polynomial", "nearest", "rbf", "local-ridge"), default="ridge")
    parser.add_argument("--bone-counts", default="50,32,20,16,8", help="Comma-separated reduced-rig counts to benchmark")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    with np.load(args.npz, allow_pickle=False) as source:
        cloth_source = np.asarray(source["traj"], dtype=np.float64)
        body_source = np.asarray(source["collision_vertices"], dtype=np.float64)
        body_edges = np.asarray(source["collision_edges"], dtype=np.int64)
    with np.load(args.poses, allow_pickle=False) as pose_data:
        world_transforms = np.asarray(pose_data["bone_transforms"], dtype=np.float64)
        rest_vertices = np.asarray(pose_data["rest_vertices"], dtype=np.float64)
        weights = np.asarray(pose_data["weights"], dtype=np.float64)
        bone_names = [str(value) for value in pose_data["bone_names"].tolist()]
    axis_matrix = infer_axis_transform(cloth_source[0], rest_vertices)
    axis_offset = rest_vertices.mean(axis=0) - (cloth_source[0] @ axis_matrix.T).mean(axis=0)
    cloth_world = cloth_source @ axis_matrix.T + axis_offset
    body_world = body_source @ axis_matrix.T + axis_offset
    if cloth_world.shape[1:] != rest_vertices.shape or len(body_world) != len(cloth_world) or len(world_transforms) != len(cloth_world):
        raise ValueError("Body, cloth, and clean-rig pose sequences are not shape-compatible")
    if not np.allclose(cloth_world[0], rest_vertices, atol=2.0e-4):
        raise ValueError("The clean-rig rest mesh does not match NPZ traj[0] after axis conversion")

    global_transforms = np.stack([kabsch_transform(body_world[0], frame) for frame in body_world], axis=0)
    inverse_global = invert_rigid_transforms(global_transforms)
    body_local = transform_points(body_world, inverse_global)
    cloth_local = transform_points(cloth_world, inverse_global)
    local_world_transforms = np.einsum("tij,tbjk->tbik", inverse_global, world_transforms)
    teacher_from_local_bones = apply_skinning(rest_vertices, weights, local_world_transforms)
    reconstruction_error = np.linalg.norm(teacher_from_local_bones - cloth_local, axis=2)
    layout = auto_regions(body_local, body_edges, args.regions)
    features, feature_names, region_transforms = _region_features(body_local[0], body_local, layout)
    targets = _target_vectors(local_world_transforms)
    bone_regions = _bone_region_assignment(rest_vertices, weights, layout)
    model_names = ("linear", "ridge", "polynomial", "nearest", "rbf", "local-ridge")
    split_names = ("contiguous_1_180", "interleaved_5", "interleaved_10", "motion_space")
    model_results: dict[str, dict[str, object]] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    for model_name in model_names:
        split_results: dict[str, object] = {}
        split_scores = []
        for split_name in split_names:
            train, validation, split_info = _split_indices(len(features), split_name, features)
            predicted_vectors = _fit_model(model_name, features, targets, train, bone_regions, args.alpha)
            predicted_vectors[0] = targets[0]
            predicted_transforms = _vectors_to_transforms(predicted_vectors, len(bone_names))
            predicted_skin = apply_skinning(rest_vertices, weights, predicted_transforms)
            metrics = _metrics(predicted_skin, cloth_local, validation)
            split_results[split_name] = {"split": split_info, "train_frames": int(len(train)), "metrics": metrics}
            split_scores.append(float(metrics["mean_rms"]))
            if split_name == "contiguous_1_180":
                predictions_by_model[model_name] = predicted_transforms
        model_results[model_name] = {"splits": split_results, "mean_split_rms": float(np.mean(split_scores))}

    selected_model = args.selected_model
    selected_transforms = predictions_by_model[selected_model]
    selected_skin = apply_skinning(rest_vertices, weights, selected_transforms)
    all_indices = np.arange(len(features), dtype=np.int64)
    contiguous_train, contiguous_validation, contiguous_info = _split_indices(len(features), "contiguous_1_180", features)
    baseline_transforms = np.broadcast_to(np.eye(4, dtype=np.float64), (len(features), len(bone_names), 4, 4)).copy()
    baseline_skin = apply_skinning(rest_vertices, weights, baseline_transforms)
    bone_counts = [int(value) for value in args.bone_counts.split(",") if value.strip()]
    reduction_results: dict[str, object] = {}
    for bone_count in bone_counts:
        selected_indices, reduced_weights = _bone_subset(weights, rest_vertices, bone_count)
        reduced_targets = targets.reshape(len(targets), len(bone_names), 6)[:, selected_indices].reshape(len(targets), -1)
        reduced_regions = bone_regions[selected_indices]
        reduced_vectors = _fit_model(selected_model, features, reduced_targets, contiguous_train, reduced_regions, args.alpha)
        reduced_vectors[0] = reduced_targets[0]
        reduced_transforms = _vectors_to_transforms(reduced_vectors, bone_count)
        reduced_skin = apply_skinning(rest_vertices, reduced_weights, reduced_transforms)
        reduced_teacher = apply_skinning(rest_vertices, reduced_weights, local_world_transforms[:, selected_indices])
        reduction_results[str(bone_count)] = {
            "selected_original_bones": selected_indices.tolist(),
            "teacher_rig_reduction": _metrics(reduced_teacher, cloth_local, contiguous_validation),
            "predicted": _metrics(reduced_skin, cloth_local, contiguous_validation),
            "predicted_vs_reduced_teacher": _metrics(reduced_skin, reduced_teacher, contiguous_validation),
        }

    selected_vectors = _target_vectors(selected_transforms)
    per_bone_std = np.linalg.norm(selected_vectors.reshape(len(selected_vectors), len(bone_names), 6).std(axis=0), axis=1)
    teacher_bone_std = np.linalg.norm(targets.reshape(len(targets), len(bone_names), 6).std(axis=0), axis=1)
    feature_centered = features[contiguous_train] - features[contiguous_train].mean(axis=0)
    feature_norm = np.linalg.norm(feature_centered, axis=0)
    mapping_bones = []
    target_matrix = targets[contiguous_train].reshape(len(contiguous_train), len(bone_names), 6)
    for bone_index, bone_name in enumerate(bone_names):
        target_centered = target_matrix[:, bone_index] - target_matrix[:, bone_index].mean(axis=0)
        target_norm = np.linalg.norm(target_centered, axis=0)
        correlation = np.abs(feature_centered.T @ target_centered) / np.maximum(feature_norm[:, None] * target_norm[None, :], 1.0e-12)
        importance = np.max(np.nan_to_num(correlation), axis=1)
        top_features = np.argsort(importance)[-8:][::-1]
        mapping_bones.append(
            {
                "bone": bone_name,
                "assigned_body_region": layout.names[int(bone_regions[bone_index])],
                "teacher_motion_std_norm": float(teacher_bone_std[bone_index]),
                "predicted_motion_std_norm": float(per_bone_std[bone_index]),
                "top_body_features": [{"feature": feature_names[int(index)], "absolute_correlation": float(importance[index])} for index in top_features],
            }
        )
    global_rotation = rotation_matrix_to_rotvec(global_transforms[:, :3, :3])
    region_residuals = []
    for frame_index in range(len(body_local)):
        values = []
        for region_index, mask in enumerate(layout.masks):
            fitted = transform_points(body_local[0, mask], region_transforms[frame_index, region_index][None, ...])[0]
            values.append(np.sqrt(np.mean(np.sum((fitted - body_local[frame_index, mask]) ** 2, axis=1))))
        region_residuals.append(values)
    diagnostics = {
        "global_body_translation_norm": [float(np.linalg.norm(value)) for value in global_transforms[:, :3, 3]],
        "global_body_rotation_deg": [float(np.degrees(np.linalg.norm(value))) for value in global_rotation],
        "canonical_body_centroid_drift": [float(np.linalg.norm(frame.mean(axis=0) - body_local[0].mean(axis=0))) for frame in body_local],
        "canonical_body_residual_rms": [float(np.sqrt(np.mean(np.sum((frame - body_local[0]) ** 2, axis=1)))) for frame in body_local],
        "canonical_cloth_centroid_drift": [float(np.linalg.norm(frame.mean(axis=0) - cloth_local[0].mean(axis=0))) for frame in cloth_local],
        "canonical_cloth_deformation_rms": [float(np.sqrt(np.mean(np.sum((frame - cloth_local[0]) ** 2, axis=1)))) for frame in cloth_local],
        "canonical_teacher_bone_reconstruction_max_error": float(reconstruction_error.max()),
        "canonical_teacher_bone_reconstruction_mean_rms": float(np.sqrt(np.mean(reconstruction_error * reconstruction_error))),
        "region_rigid_fit_mean_rms": np.asarray(region_residuals).mean(axis=0).tolist(),
        "region_rigid_fit_max_rms": np.asarray(region_residuals).max(axis=0).tolist(),
    }
    region_info = {
        "names": list(layout.names),
        "sizes": [int(mask.sum()) for mask in layout.masks],
        "centroids": layout.centroids.tolist(),
        "connected_components": list(layout.connected_components),
        "bone_region_assignment": bone_regions.tolist(),
    }
    split_baselines = {}
    for split_name in split_names:
        _, validation, split_info = _split_indices(len(features), split_name, features)
        split_baselines[split_name] = {"split": split_info, "metrics": _metrics(baseline_skin, cloth_local, validation)}
    report = {
        "schema_version": 1,
        "source": {"npz": str(args.npz.resolve()), "poses": str(args.poses.resolve())},
        "frames": int(len(features)),
        "vertices": int(len(rest_vertices)),
        "bones": int(len(bone_names)),
        "bone_names": bone_names,
        "axis_matrix": axis_matrix.tolist(),
        "axis_offset": axis_offset.tolist(),
        "canonicalization": diagnostics,
        "regions": region_info,
        "features": {"count": int(features.shape[1]), "names": feature_names, "uses_frame_index": False},
        "targets": {"representation": "local rigid transform rotation-vector plus translation", "dimensions_per_bone": 6, "uses_cloth_features": False},
        "models": model_results,
        "selected_model": selected_model,
        "baseline": split_baselines,
        "selected_model_contiguous": {
            "split": contiguous_info,
            "train_frames": [1, int(contiguous_train[-1] + 1)],
            "validation_frames": [int(contiguous_validation[0] + 1), int(contiguous_validation[-1] + 1)],
            "metrics": _metrics(selected_skin, cloth_local, contiguous_validation, include_per_frame=True),
            "all_metrics": _metrics(selected_skin, cloth_local, all_indices),
        },
        "bone_motion": {
            "teacher_std_norm_per_bone": teacher_bone_std.tolist(),
            "predicted_std_norm_per_bone": per_bone_std.tolist(),
            "predicted_bones_with_nonzero_motion": int(np.count_nonzero(per_bone_std > 1.0e-5)),
            "predicted_all_bones_same_transform": bool(np.max(per_bone_std) < 1.0e-5),
        },
        "mapping": {"method": "nearest body-feature pose for selected model; train-only feature/target correlations for inspection", "bones": mapping_bones},
        "bone_count_comparison": reduction_results,
        "leakage_checks": {
            "frame_index_in_features": False,
            "time_or_progress_in_features": False,
            "cloth_teacher_in_features": False,
            "global_rigid_transform_removed_before_features": True,
            "contiguous_and_interleaved_and_motion_space_splits": True,
            "rest_identity_is_known_without_teacher": True,
        },
        "artifacts": {
            "features": "body_local_features.npz",
            "teacher_poses": "teacher_local_poses_50.npz",
            "predicted_poses": f"predicted_local_poses_{len(bone_names)}.npz",
            "per_frame_metrics": "per_frame_metrics.npz",
            "bone_mapping": "bone_mapping.json",
        },
    }
    np.savez_compressed(
        args.out_root / "body_local_features.npz",
        body_local=body_local,
        cloth_local=cloth_local,
        global_transforms=global_transforms,
        region_transforms=region_transforms,
        features=features,
        feature_names=np.asarray(feature_names),
        targets=targets,
        bone_regions=bone_regions,
    )
    np.savez_compressed(
        args.out_root / "teacher_local_poses_50.npz",
        bone_transforms=local_world_transforms,
        bone_names=np.asarray(bone_names),
        rest_vertices=rest_vertices,
        weights=weights,
    )
    np.savez_compressed(
        args.out_root / f"predicted_local_poses_{len(bone_names)}.npz",
        bone_transforms=selected_transforms,
        bone_names=np.asarray(bone_names),
        rest_vertices=rest_vertices,
        weights=weights,
        model=np.asarray([selected_model]),
    )
    per_frame = np.stack(
        [
            np.sqrt(np.mean(np.sum((selected_skin - cloth_local) ** 2, axis=2), axis=1)),
            np.sqrt(np.mean(np.sum((baseline_skin - cloth_local) ** 2, axis=2), axis=1)),
            np.max(np.linalg.norm(selected_skin - cloth_local, axis=2), axis=1),
        ],
        axis=1,
    )
    np.savez_compressed(args.out_root / "per_frame_metrics.npz", frames=np.arange(1, len(features) + 1), selected_rms=per_frame[:, 0], rest_rms=per_frame[:, 1], selected_max=per_frame[:, 2])
    (args.out_root / "bone_mapping.json").write_text(json.dumps({"selected_model": selected_model, "feature_names": feature_names, "bones": mapping_bones}, indent=2), encoding="utf-8")
    (args.out_root / "body_local_report.json").write_text(json.dumps(_serialize_metrics(report), indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
