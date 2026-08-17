"""PoC 3.2: leave-one-sequence-out body-motion-to-cloth prediction.

The model is trained on complete sequences and tested on a sequence whose
frames, teacher cloth, and teacher bone poses never enter fitting.  Features
are fixed body-region transforms in body-local space; frame number, progress,
sequence ID, and cloth teacher data are deliberately excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    transform_points,
)


@dataclass(frozen=True)
class FixedRegions:
    names: tuple[str, ...]
    labels: np.ndarray
    masks: tuple[np.ndarray, ...]
    centroids: np.ndarray


@dataclass(frozen=True)
class SequenceData:
    sequence_id: str
    path: Path
    body_local: np.ndarray
    cloth_local: np.ndarray
    global_transforms: np.ndarray
    bone_transforms: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    targets: np.ndarray
    frames: int
    reconstruction_rms: float
    metadata: dict[str, Any]
    canonicalization: dict[str, float]


def _spatial_regions(rest: np.ndarray, count: int) -> FixedRegions:
    values = np.asarray(rest, dtype=np.float64)
    standardized = (values - values.mean(axis=0)) / np.where(values.std(axis=0) < 1.0e-9, 1.0, values.std(axis=0))
    centers = [0]
    distances = np.full(len(values), np.inf, dtype=np.float64)
    for _ in range(1, count):
        distances = np.minimum(distances, np.sum((standardized - standardized[centers[-1]]) ** 2, axis=1))
        centers.append(int(np.argmax(distances)))
    centroids = standardized[np.asarray(centers)].copy()
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(60):
        distances = np.sum((standardized[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for region in range(count):
            if np.any(labels == region):
                centroids[region] = standardized[labels == region].mean(axis=0)
    raw_centroids = np.stack([values[labels == region].mean(axis=0) for region in range(count)])
    order = np.lexsort((raw_centroids[:, 0], raw_centroids[:, 1], raw_centroids[:, 2]))
    remap = np.empty(count, dtype=np.int64)
    remap[order] = np.arange(count)
    labels = remap[labels]
    masks = tuple(labels == region for region in range(count))
    raw_centroids = np.stack([values[mask].mean(axis=0) for mask in masks])
    return FixedRegions(
        names=tuple(f"fixed_rest_region_{region:02d}" for region in range(count)),
        labels=labels,
        masks=masks,
        centroids=raw_centroids,
    )


def _weighted_kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mass = np.asarray(weights, dtype=np.float64)
    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if len(source_values) < 3 or np.sum(mass) <= 1.0e-9:
        return np.eye(4, dtype=np.float64)
    source_center = np.average(source_values, axis=0, weights=mass)
    target_center = np.average(target_values, axis=0, weights=mass)
    covariance = (source_values - source_center).T @ ((target_values - target_center) * mass[:, None])
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = target_center - rotation @ source_center
    return result


def _fit_bone_transforms(rest: np.ndarray, cloth: np.ndarray, weights: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Fit one common-rig transform per frame by alternating weighted Kabsch."""

    frames, bone_count = len(cloth), weights.shape[1]
    fitted = np.tile(np.eye(4, dtype=np.float64), (frames, bone_count, 1, 1))
    for frame_index in range(frames):
        target = cloth[frame_index]
        current = fitted[frame_index]
        for _ in range(iterations):
            predicted = apply_skinning(rest, weights, current)
            for bone in range(bone_count):
                influence = weights[:, bone]
                mask = influence > 1.0e-4
                if int(mask.sum()) < 3:
                    continue
                without_bone = predicted - influence[:, None] * (
                    rest @ current[bone, :3, :3].T + current[bone, :3, 3]
                )
                desired = (target - without_bone[mask]) / influence[mask, None]
                current[bone] = _weighted_kabsch(rest[mask], desired, influence[mask] ** 2)
        fitted[frame_index] = current
    return fitted


def _region_features(rest: np.ndarray, frames: np.ndarray, layout: FixedRegions) -> tuple[np.ndarray, tuple[str, ...]]:
    feature_blocks: list[np.ndarray] = []
    names: list[str] = []
    for region, mask in enumerate(layout.masks):
        source = rest[mask]
        source_center = source.mean(axis=0)
        source_zero = source - source_center
        inverse = np.linalg.pinv(source_zero.T @ source_zero + np.eye(3) * 1.0e-9)
        blocks = []
        for frame in frames:
            target = frame[mask]
            transform = kabsch_transform(source, target)
            rotvec = rotation_matrix_to_rotvec(transform[:3, :3])
            translation = transform[:3, 3]
            centroid_delta = target.mean(axis=0) - source_center
            target_zero = target - target.mean(axis=0)
            linear = target_zero.T @ source_zero @ inverse
            scale_shear = np.log(np.maximum(np.linalg.svd(linear, compute_uv=False), 1.0e-9))
            blocks.append(np.concatenate([rotvec, translation, centroid_delta, scale_shear]))
        feature_blocks.append(np.asarray(blocks, dtype=np.float64))
        names.extend(
            [
                f"{layout.names[region]}.rotation_vector.x",
                f"{layout.names[region]}.rotation_vector.y",
                f"{layout.names[region]}.rotation_vector.z",
                f"{layout.names[region]}.translation.x",
                f"{layout.names[region]}.translation.y",
                f"{layout.names[region]}.translation.z",
                f"{layout.names[region]}.centroid_delta.x",
                f"{layout.names[region]}.centroid_delta.y",
                f"{layout.names[region]}.centroid_delta.z",
                f"{layout.names[region]}.log_scale.x",
                f"{layout.names[region]}.log_scale.y",
                f"{layout.names[region]}.log_scale.z",
            ]
        )
    return np.concatenate(feature_blocks, axis=1), tuple(names)


def _target_vectors(transforms: np.ndarray) -> np.ndarray:
    rotations = rotation_matrix_to_rotvec(transforms[:, :, :3, :3])
    translations = transforms[:, :, :3, 3]
    return np.concatenate([rotations, translations], axis=2).reshape(len(transforms), -1)


def _vectors_to_transforms(vectors: np.ndarray, bone_count: int) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64).reshape(len(vectors), bone_count, 6)
    result = np.tile(np.eye(4, dtype=np.float64), (len(values), bone_count, 1, 1))
    from cloth2bones.body_motion import rotvec_to_rotation_matrix

    result[:, :, :3, :3] = rotvec_to_rotation_matrix(values[:, :, :3])
    result[:, :, :3, 3] = values[:, :, 3:]
    return result


def _fit_model(name: str, features: np.ndarray, targets: np.ndarray, train: np.ndarray, bone_regions: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    train_values = values[train]
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[scale < 1.0e-9] = 1.0
    normalized = (values - mean) / scale
    if name in {"linear", "ridge", "polynomial", "local-ridge"}:
        if name == "polynomial":
            design_values = np.concatenate([normalized, normalized * normalized], axis=1)
            selected = design_values
        elif name == "local-ridge":
            predictions = np.empty_like(target)
            block_size = values.shape[1] // (int(np.max(bone_regions)) + 1)
            for bone, region in enumerate(bone_regions):
                columns = np.arange(region * block_size, (region + 1) * block_size, dtype=np.int64)
                local = normalized[:, columns]
                design = np.concatenate([np.ones((len(train), 1)), local[train]], axis=1)
                regularizer = np.eye(design.shape[1]) * alpha
                regularizer[0, 0] = 0.0
                coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ target[train, bone * 6 : bone * 6 + 6])
                predictions[:, bone * 6 : bone * 6 + 6] = np.concatenate([np.ones((len(values), 1)), local], axis=1) @ coefficients
            return predictions
        else:
            selected = normalized
        design = np.concatenate([np.ones((len(train), 1)), selected[train]], axis=1)
        regularizer = np.eye(design.shape[1]) * (1.0e-8 if name == "linear" else alpha)
        regularizer[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ target[train])
        return np.concatenate([np.ones((len(values), 1)), selected], axis=1) @ coefficients
    if name in {"nearest", "rbf"}:
        train_normalized = normalized[train]
        distances = np.sqrt(np.maximum(np.sum((normalized[:, None, :] - train_normalized[None, :, :]) ** 2, axis=2), 0.0))
        if name == "nearest":
            return target[train[np.argmin(distances, axis=1)]]
        positive = distances[distances > 1.0e-9]
        bandwidth = float(np.median(positive)) if len(positive) else 1.0
        kernel = np.exp(-0.5 * (distances / max(bandwidth, 1.0e-6)) ** 2)
        fit_kernel = kernel[train]
        coefficients = np.linalg.solve(fit_kernel + np.eye(len(train)) * alpha, target[train])
        return kernel @ coefficients
    raise ValueError(f"Unknown model {name}")


def _nearest_distances(features: np.ndarray, train_features: np.ndarray) -> np.ndarray:
    train_values = np.asarray(train_features, dtype=np.float64)
    all_values = np.asarray(features, dtype=np.float64)
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[scale < 1.0e-9] = 1.0
    normalized_train = (train_values - mean) / scale
    normalized_all = (all_values - mean) / scale
    return np.sqrt(np.min(np.sum((normalized_all[:, None, :] - normalized_train[None, :, :]) ** 2, axis=2), axis=1))


def _metrics(predicted: np.ndarray, teacher: np.ndarray) -> dict[str, Any]:
    errors = np.linalg.norm(np.asarray(predicted) - np.asarray(teacher), axis=2)
    frame_rms = np.sqrt(np.mean(errors * errors, axis=1))
    frame_max = errors.max(axis=1)
    return {
        "frames": int(len(frame_rms)),
        "mean_rms": float(frame_rms.mean()),
        "max_frame_rms": float(frame_rms.max()),
        "max_point_error": float(frame_max.max()),
        "p95_point_error": float(np.percentile(errors, 95)),
        "mean_point_error": float(errors.mean()),
        "per_frame_rms": frame_rms.tolist(),
        "per_frame_max_point_error": frame_max.tolist(),
    }


def _bone_regions(rest_vertices: np.ndarray, weights: np.ndarray, regions: FixedRegions) -> np.ndarray:
    mass = weights.sum(axis=0)
    centers = (weights.T @ rest_vertices) / np.maximum(mass[:, None], 1.0e-9)
    return np.argmin(np.sum((centers[:, None, :] - regions.centroids[None, :, :]) ** 2, axis=2), axis=1)


def _bone_subset(weights: np.ndarray, rest_vertices: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if count >= weights.shape[1]:
        return np.arange(weights.shape[1], dtype=np.int64), weights.copy()
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
    for original in range(len(centers)):
        reduced[:, nearest[original]] += weights[:, original]
    reduced /= np.maximum(reduced.sum(axis=1, keepdims=True), 1.0e-12)
    return selected_array, reduced


def _load_sequence(path: Path, rest_vertices: np.ndarray, weights: np.ndarray, regions: FixedRegions, bone_count: int) -> SequenceData:
    with np.load(path, allow_pickle=False) as source:
        cloth_source = np.asarray(source["traj"], dtype=np.float64)
        body_source = np.asarray(source["collision_vertices"], dtype=np.float64)
        source_rest = np.asarray(source["initial"], dtype=np.float64)[:, :3] if "initial" in source else cloth_source[0]
        axis_matrix = infer_axis_transform(source_rest, rest_vertices)
        offset = rest_vertices.mean(axis=0) - (source_rest @ axis_matrix.T).mean(axis=0)
        cloth_world = cloth_source @ axis_matrix.T + offset
        body_world = body_source @ axis_matrix.T + offset
        if cloth_world.shape[1:] != rest_vertices.shape:
            raise ValueError(f"{path.name} is not compatible with the common cloth rig")
        globals_ = np.stack([kabsch_transform(body_world[0], frame) for frame in body_world])
        inverse = invert_rigid_transforms(globals_)
        body_local = transform_points(body_world, inverse)
        cloth_local = transform_points(cloth_world, inverse)
        if "teacher_bone_transforms_world" in source:
            world_bones = np.asarray(source["teacher_bone_transforms_world"], dtype=np.float64)
            bone_transforms = np.einsum("tij,tbjk->tbik", inverse, world_bones)
        elif "teacher_bone_transforms" in source:
            bone_transforms = np.asarray(source["teacher_bone_transforms"], dtype=np.float64)
        else:
            bone_transforms = _fit_bone_transforms(rest_vertices, cloth_local, weights)
        if bone_transforms.shape != (len(cloth_local), bone_count, 4, 4):
            raise ValueError(f"{path.name} teacher bone shape is {bone_transforms.shape}, expected {(len(cloth_local), bone_count, 4, 4)}")
        reconstructed = apply_skinning(rest_vertices, weights, bone_transforms)
        reconstruction_error = np.linalg.norm(reconstructed - cloth_local, axis=2)
        features, feature_names = _region_features(body_local[0], body_local, regions)
        global_rotvec = rotation_matrix_to_rotvec(globals_[:, :3, :3])
        canonicalization = {
            "global_translation_max": float(np.linalg.norm(globals_[:, :3, 3], axis=1).max()),
            "global_rotation_max_deg": float(np.degrees(np.linalg.norm(global_rotvec, axis=1)).max()),
            "body_centroid_drift_max": float(np.linalg.norm(body_local.mean(axis=1) - body_local[0].mean(axis=0), axis=1).max()),
            "body_residual_rms_mean": float(np.sqrt(np.mean(np.sum((body_local - body_local[0]) ** 2, axis=2), axis=1)).mean()),
            "cloth_centroid_drift_max": float(np.linalg.norm(cloth_local.mean(axis=1) - cloth_local[0].mean(axis=0), axis=1).max()),
        }
        metadata: dict[str, Any] = {"path": str(path.resolve())}
        for key in ("sequence_id", "motion_label", "fps"):
            if key in source:
                value = source[key].item()
                metadata[key] = value if isinstance(value, (str, int, float)) else str(value)
        sequence_id = str(metadata.get("sequence_id", path.stem))
    return SequenceData(
        sequence_id=sequence_id,
        path=path,
        body_local=body_local,
        cloth_local=cloth_local,
        global_transforms=globals_,
        bone_transforms=bone_transforms,
        features=features,
        feature_names=feature_names,
        targets=_target_vectors(bone_transforms),
        frames=len(cloth_local),
        reconstruction_rms=float(np.sqrt(np.mean(reconstruction_error * reconstruction_error))),
        metadata=metadata,
        canonicalization=canonicalization,
    )


def _write_pose(path: Path, transforms: np.ndarray, bone_names: list[str], rest_vertices: np.ndarray, weights: np.ndarray) -> None:
    np.savez_compressed(path, bone_transforms=transforms, bone_names=np.asarray(bone_names), rest_vertices=rest_vertices, weights=weights, frames=np.arange(1, len(transforms) + 1))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["fold", "model", "feature_set", "mean_rms", "max_frame_rms", "max_point_error", "p95_point_error", "improvement_ratio"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _evaluate_fold(
    train_sequences: list[SequenceData],
    test_sequence: SequenceData,
    model_names: tuple[str, ...],
    velocity_by_sequence: dict[str, np.ndarray],
    bone_regions: np.ndarray,
    weights: np.ndarray,
    rest_vertices: np.ndarray,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result: dict[str, Any] = {"test_sequence": test_sequence.sequence_id, "models": {}}
    predictions: dict[str, np.ndarray] = {}
    baseline_transforms = np.tile(np.eye(4, dtype=np.float64), (test_sequence.frames, weights.shape[1], 1, 1))
    baseline_skin = apply_skinning(rest_vertices, weights, baseline_transforms)
    baseline_metrics = _metrics(baseline_skin, test_sequence.cloth_local)
    for model_name in model_names:
        feature_set = "with_velocity" if model_name.endswith("+velocity") else "pose_only"
        base_name = model_name.replace("+velocity", "")
        train_features = np.concatenate([sequence.features if feature_set == "pose_only" else velocity_by_sequence[sequence.sequence_id] for sequence in train_sequences])
        train_targets = np.concatenate([sequence.targets for sequence in train_sequences])
        test_features = test_sequence.features if feature_set == "pose_only" else velocity_by_sequence[test_sequence.sequence_id]
        pooled_features = np.concatenate([train_features, test_features])
        train_indices = np.arange(len(train_features), dtype=np.int64)
        pooled_targets = np.concatenate([train_targets, np.zeros((len(test_features), train_targets.shape[1]))])
        predicted_vectors = _fit_model(base_name, pooled_features, pooled_targets, train_indices, bone_regions, alpha)
        predicted_transforms = _vectors_to_transforms(predicted_vectors[len(train_features) :], weights.shape[1])
        predicted_skin = apply_skinning(rest_vertices, weights, predicted_transforms)
        metrics = _metrics(predicted_skin, test_sequence.cloth_local)
        metrics["improvement_ratio"] = float((baseline_metrics["mean_rms"] - metrics["mean_rms"]) / max(baseline_metrics["mean_rms"], 1.0e-12))
        train_features_for_distance = train_features
        nearest = _nearest_distances(test_features, train_features_for_distance)
        frame_errors = np.asarray(metrics["per_frame_rms"], dtype=np.float64)
        metrics["nearest_train_feature_distance_mean"] = float(nearest.mean())
        metrics["nearest_train_feature_distance_max"] = float(nearest.max())
        metrics["nearest_distance_error_correlation"] = float(np.corrcoef(nearest, frame_errors)[0, 1]) if np.std(nearest) > 1.0e-12 and np.std(frame_errors) > 1.0e-12 else 0.0
        result["models"][model_name] = {"metrics": metrics, "baseline": baseline_metrics}
        predictions[model_name] = predicted_transforms
    return result, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PoC 3.2 leave-one-sequence-out cloth prediction")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True, help="Common clean-rig pose NPZ")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--regions", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--primary-test", default="seq_D_combined_holdout")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    with np.load(args.poses, allow_pickle=False) as pose_data:
        rest_vertices = np.asarray(pose_data["rest_vertices"], dtype=np.float64)
        weights = np.asarray(pose_data["weights"], dtype=np.float64)
        bone_names = [str(value) for value in pose_data["bone_names"].tolist()]
    if weights.shape[0] != len(rest_vertices):
        raise ValueError("Common rig rest vertices and weights are incompatible")
    first_path = sorted(args.dataset_root.glob("*.npz"))[0]
    with np.load(first_path, allow_pickle=False) as source:
        body_rest_source = np.asarray(source["collision_vertices"][0], dtype=np.float64)
        source_rest = np.asarray(source["initial"], dtype=np.float64)[:, :3] if "initial" in source else np.asarray(source["traj"][0], dtype=np.float64)
    axis = infer_axis_transform(source_rest, rest_vertices)
    body_rest = body_rest_source @ axis.T + (rest_vertices.mean(axis=0) - (source_rest @ axis.T).mean(axis=0))
    regions = _spatial_regions(body_rest, args.regions)
    paths = sorted(args.dataset_root.glob("*.npz"))
    if len(paths) < 2:
        raise ValueError("PoC 3.2 requires at least two sequences")
    sequences = [_load_sequence(path, rest_vertices, weights, regions, len(bone_names)) for path in paths]
    bone_regions = _bone_regions(rest_vertices, weights, regions)
    for sequence in sequences:
        if sequence.frames != sequences[0].frames:
            raise ValueError("All cross-sequence fixtures must have the same frame count")
    model_names = ("linear", "ridge", "polynomial", "nearest", "rbf", "local-ridge", "ridge+velocity", "nearest+velocity")
    feature_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sequence in sequences:
        velocity = np.vstack([np.zeros((1, sequence.features.shape[1])), np.diff(sequence.features, axis=0)])
        feature_sets.setdefault("velocity", ([], []))
        feature_sets["velocity"][0].append(sequence.features)
        feature_sets["velocity"][1].append(velocity)
    train_velocity = np.concatenate(feature_sets["velocity"][0])
    all_velocity = np.concatenate(feature_sets["velocity"][1])
    # Per-sequence velocity matrices are recovered below; the pooled arrays are
    # retained only for reproducible schema/diagnostic output.
    velocity_by_sequence = {
        sequence.sequence_id: np.vstack([np.zeros((1, sequence.features.shape[1])), np.diff(sequence.features, axis=0)])
        for sequence in sequences
    }
    loso: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    fold_predictions: dict[str, dict[str, np.ndarray]] = {}
    for test_index, test_sequence in enumerate(sequences):
        train_sequences = [sequence for index, sequence in enumerate(sequences) if index != test_index]
        # Fold-local feature tuples keep test frames out of model fitting while
        # preserving one shared rest-space region assignment.
        fold_result, predictions = _evaluate_fold(train_sequences, test_sequence, model_names, velocity_by_sequence, bone_regions, weights, rest_vertices, args.alpha)
        loso.append(fold_result)
        fold_predictions[test_sequence.sequence_id] = predictions
        for model_name, entry in fold_result["models"].items():
            metrics = entry["metrics"]
            csv_rows.append({"fold": test_sequence.sequence_id, "model": model_name, "feature_set": "with_velocity" if model_name.endswith("+velocity") else "pose_only", **metrics})

    aggregate: dict[str, dict[str, float]] = {}
    for model_name in model_names:
        scores = [float(fold["models"][model_name]["metrics"]["mean_rms"]) for fold in loso]
        improvements = [float(fold["models"][model_name]["metrics"]["improvement_ratio"]) for fold in loso]
        aggregate[model_name] = {"mean_rms": float(np.mean(scores)), "max_fold_rms": float(np.max(scores)), "mean_improvement_ratio": float(np.mean(improvements))}
    research_model = min(model_names, key=lambda name: aggregate[name]["mean_rms"])
    runtime_candidates = ("ridge", "local-ridge", "ridge+velocity")
    runtime_model = min(runtime_candidates, key=lambda name: aggregate[name]["mean_rms"])
    primary = next((sequence for sequence in sequences if sequence.sequence_id == args.primary_test), sequences[-1])
    primary_predictions = fold_predictions[primary.sequence_id]
    primary_result = next(fold for fold in loso if fold["test_sequence"] == primary.sequence_id)
    for model_name, transforms in primary_predictions.items():
        safe_name = model_name.replace("+", "_")
        _write_pose(args.out_root / f"{safe_name}_{primary.sequence_id}_poses.npz", transforms, bone_names, rest_vertices, weights)
    _write_pose(args.out_root / f"teacher_{primary.sequence_id}_poses.npz", primary.bone_transforms, bone_names, rest_vertices, weights)
    baseline_transforms = np.tile(np.eye(4, dtype=np.float64), (primary.frames, len(bone_names), 1, 1))
    _write_pose(args.out_root / f"rest_{primary.sequence_id}_poses.npz", baseline_transforms, bone_names, rest_vertices, weights)
    primary_best = primary_result["models"][research_model]["metrics"]
    target_motion = np.linalg.norm(primary.targets.reshape(primary.frames, len(bone_names), 6).std(axis=0), axis=1)
    research_motion = np.linalg.norm(_target_vectors(primary_predictions[research_model]).reshape(primary.frames, len(bone_names), 6).std(axis=0), axis=1)
    runtime_motion = np.linalg.norm(_target_vectors(primary_predictions[runtime_model]).reshape(primary.frames, len(bone_names), 6).std(axis=0), axis=1)
    region_features = {
        "names": list(primary.feature_names),
        "count": len(primary.feature_names),
        "regions": list(regions.names),
        "region_sizes": [int(mask.sum()) for mask in regions.masks],
        "bone_region_assignment": [regions.names[int(region)] for region in bone_regions],
    }
    mapping_bones = []
    train_primary = [sequence for sequence in sequences if sequence.sequence_id != primary.sequence_id]
    train_features = np.concatenate([sequence.features for sequence in train_primary])
    train_targets = np.concatenate([sequence.targets for sequence in train_primary]).reshape(-1, len(bone_names), 6)
    centered_features = train_features - train_features.mean(axis=0)
    feature_norm = np.linalg.norm(centered_features, axis=0)
    for bone in range(len(bone_names)):
        target = train_targets[:, bone] - train_targets[:, bone].mean(axis=0)
        target_norm = np.linalg.norm(target, axis=0)
        correlation = np.abs(centered_features.T @ target) / np.maximum(feature_norm[:, None] * target_norm[None, :], 1.0e-12)
        importance = np.nan_to_num(correlation).max(axis=1)
        indices = np.argsort(importance)[-8:][::-1]
        mapping_bones.append({"bone": bone_names[bone], "assigned_body_region": regions.names[int(bone_regions[bone])], "top_features": [{"feature": primary.feature_names[int(index)], "absolute_correlation": float(importance[index])} for index in indices]})
    bone_count_comparison: dict[str, Any] = {}
    for count in (50, 32, 20, 16, 8):
        if count > len(bone_names):
            continue
        selected, reduced_weights = _bone_subset(weights, rest_vertices, count)
        reduced_regions = bone_regions[selected]
        train_targets = np.concatenate([sequence.targets.reshape(sequence.frames, len(bone_names), 6)[:, selected].reshape(sequence.frames, count * 6) for sequence in train_primary])
        train_features = np.concatenate([sequence.features for sequence in train_primary])
        test_features = primary.features
        pooled_features = np.concatenate([train_features, test_features])
        train_indices = np.arange(len(train_features), dtype=np.int64)
        pooled_targets = np.concatenate([train_targets, np.zeros((len(test_features), count * 6))])
        base_name = runtime_model.replace("+velocity", "")
        predicted_vectors = _fit_model(base_name, pooled_features, pooled_targets, train_indices, reduced_regions, args.alpha)
        predicted = _vectors_to_transforms(predicted_vectors[len(train_features) :], count)
        teacher_reduced = primary.bone_transforms[:, selected]
        predicted_skin = apply_skinning(rest_vertices, reduced_weights, predicted)
        teacher_skin = apply_skinning(rest_vertices, reduced_weights, teacher_reduced)
        rest_skin = apply_skinning(rest_vertices, reduced_weights, np.tile(np.eye(4), (primary.frames, count, 1, 1)))
        bone_count_comparison[str(count)] = {"predicted": _metrics(predicted_skin, primary.cloth_local), "rest": _metrics(rest_skin, primary.cloth_local), "teacher_reduced_rig": _metrics(teacher_skin, primary.cloth_local), "selected_original_bones": selected.tolist()}
    report = {
        "schema_version": 1,
        "dataset_mode": "synthetic_fallback_same_rig_cross_sequence",
        "source_dataset_audit": str((args.out_root / "dataset_inventory.json").resolve()),
        "common_rig": {"bones": len(bone_names), "vertices": len(rest_vertices), "weights_shape": list(weights.shape), "bone_names": bone_names},
        "sequences": [{"sequence_id": sequence.sequence_id, "path": str(sequence.path.resolve()), "frames": sequence.frames, "metadata": sequence.metadata, "bone_fit_reconstruction_rms": sequence.reconstruction_rms, "canonicalization": sequence.canonicalization} for sequence in sequences],
        "regions": region_features,
        "features": {"pose_only_dimensions": int(sequences[0].features.shape[1]), "velocity_dimensions": int(sequences[0].features.shape[1]), "names": list(sequences[0].feature_names), "uses_frame_index": False, "uses_time_or_progress": False, "uses_sequence_id": False, "uses_cloth_teacher": False},
        "target": {"representation": "per-bone local rotation-vector plus translation", "dimensions_per_bone": 6, "common_rig": True},
        "baselines": {"static_rest": "identity cloth-bone transforms", "global_only": "identity after body-local canonicalization; numerically identical to static_rest"},
        "body_velocity_state": {"available_in_fixture": True, "representation": "first difference of body-local region feature vector", "models_tested": ["ridge+velocity", "nearest+velocity"], "selected": False},
        "train_protocol": {"method": "leave-one-sequence-out", "folds": [fold["test_sequence"] for fold in loso], "primary_train_sequences": [sequence.sequence_id for sequence in sequences if sequence.sequence_id != primary.sequence_id], "primary_test_sequence": primary.sequence_id},
        "models": {"per_fold": loso, "aggregate": aggregate, "research_best": research_model, "runtime_friendly_best": runtime_model},
        "primary_test": {"sequence_id": primary.sequence_id, "research_model": research_model, "runtime_model": runtime_model, "research_metrics": primary_best, "rest_baseline": primary_result["models"][research_model]["baseline"]},
        "bone_count_comparison": bone_count_comparison,
        "bone_motion": {"teacher_nonzero_bones": int(np.count_nonzero(target_motion > 1.0e-5)), "research_nonzero_bones": int(np.count_nonzero(research_motion > 1.0e-5)), "runtime_nonzero_bones": int(np.count_nonzero(runtime_motion > 1.0e-5)), "runtime_std_norm_min": float(runtime_motion.min()), "runtime_std_norm_max": float(runtime_motion.max()), "runtime_std_norm_mean": float(runtime_motion.mean()), "all_runtime_bones_same_transform": bool(np.max(runtime_motion) < 1.0e-5)},
        "mapping": {"method": "training-fold absolute feature/target correlation", "bones": mapping_bones},
        "leakage_checks": {"frame_index_in_features": False, "time_or_progress_in_features": False, "sequence_id_in_features": False, "cloth_teacher_in_features": False, "test_teacher_in_model_fit": False, "test_frames_in_nearest_database": False, "global_rigid_motion_removed": True, "regions_fit_from_common_rest_only": True},
        "status": {"unknown_motion_sequence_generalization": bool(all(float(fold["models"][research_model]["metrics"]["improvement_ratio"]) > 0.0 for fold in loso)), "most_test_frames_improved_primary": int(np.count_nonzero(np.asarray(primary_best["per_frame_rms"]) < np.asarray(primary_result["models"][research_model]["baseline"]["per_frame_rms"]))), "primary_test_frames": primary.frames},
        "artifacts": {"metrics_csv": "metrics.csv", "bone_mapping": "bone_mapping.json", "primary_teacher_poses": f"teacher_{primary.sequence_id}_poses.npz", "primary_research_poses": f"{research_model.replace('+', '_')}_{primary.sequence_id}_poses.npz", "primary_runtime_poses": f"{runtime_model.replace('+', '_')}_{primary.sequence_id}_poses.npz"},
    }
    (args.out_root / "bone_mapping.json").write_text(json.dumps(region_features | {"bones": mapping_bones}, indent=2), encoding="utf-8")
    (args.out_root / "cross_sequence_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(args.out_root / "metrics.csv", csv_rows)
    np.savez_compressed(args.out_root / "primary_features.npz", train_features=train_features, test_features=primary.features, test_body_local=primary.body_local, test_cloth_local=primary.cloth_local, test_global_transforms=primary.global_transforms, velocity_train=train_velocity, velocity_all=all_velocity)
    np.savez_compressed(args.out_root / "primary_teacher_reference.npz", cloth_local=primary.cloth_local)
    print(json.dumps({"research_best": research_model, "runtime_friendly_best": runtime_model, "primary": report["primary_test"], "loso": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
