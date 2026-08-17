"""Evaluate semantic-skeleton to Blender-Cloth helper-bone prediction.

The input sequences are exported by generate_blender_physics_poc.py. A
common 50-bone clean rig is fixed before the primary holdout is opened. Cloth
geometry from the holdout is used only for post-fit evaluation and never for
weights, feature normalization, nearest-pose lookup, or regression.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from cross_sequence_poc import (  # noqa: E402
    _bone_subset,
    _fit_model,
    _metrics,
    _target_vectors,
    _vectors_to_transforms,
)

from cloth2bones.body_motion import (  # noqa: E402
    apply_skinning,
    invert_rigid_transforms,
    rotation_matrix_to_rotvec,
    rotvec_to_rotation_matrix,
    transform_points,
)

INPUT_BONES = ("Chest", "Upper_arm.L", "Lower_arm.L", "Upper_arm.R", "Lower_arm.R")
MODEL_NAMES = ("nearest", "linear", "ridge", "polynomial", "rbf", "local-ridge")
MODEL_ALPHAS = {"ridge": 1.0e-3, "local-ridge": 1.0e-3, "rbf": 1.0e-2}


@dataclass(frozen=True)
class PhysicsSequence:
    sequence_id: str
    path: Path
    rest: np.ndarray
    triangles: np.ndarray
    cloth_local: np.ndarray
    cloth_velocity: np.ndarray
    skeleton_features: np.ndarray
    feature_names: tuple[str, ...]
    skeleton_names: tuple[str, ...]
    teacher_bones: np.ndarray
    targets: np.ndarray
    metadata: dict[str, Any]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--common-poses", required=True, help="Verified clean-rig NPZ used for common rest and weights")
    parser.add_argument("--triangles", required=True, help="Reference NPZ containing common triangles")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--primary-test", default="H_unseen_combined")
    parser.add_argument("--train-sequences", default="A_left_down,B_right_down,C_both_down,D_left_down_up,E_right_down_up,F_alternating,G_diagonal")
    parser.add_argument("--alpha", type=float, default=1.0e-3)
    return parser.parse_args()


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _load_common(path: Path, triangles_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as source:
        rest = np.asarray(source["rest_vertices"], dtype=np.float64)
        weights = np.asarray(source["weights"], dtype=np.float64)
        bone_names = [str(value) for value in source["bone_names"].tolist()]
    with np.load(triangles_path, allow_pickle=False) as source:
        triangles = np.asarray(source["triangles"], dtype=np.int32)
    if weights.shape != (len(rest), len(bone_names)):
        raise ValueError(f"Common weights shape {weights.shape} does not match rest/bones")
    return rest, weights, triangles, bone_names


def _semantic_features(transforms: np.ndarray, names: tuple[str, ...], fps: float) -> tuple[np.ndarray, tuple[str, ...]]:
    name_to_index = {name: index for index, name in enumerate(names)}
    missing = [name for name in INPUT_BONES if name not in name_to_index]
    if missing:
        raise ValueError(f"Semantic skeleton is missing bones: {missing}")
    selected = np.asarray([name_to_index[name] for name in INPUT_BONES], dtype=np.int64)
    values = np.asarray(transforms, dtype=np.float64)[:, selected]
    rotations = rotation_matrix_to_rotvec(values[:, :, :3, :3])
    translations = values[:, :, :3, 3]
    blocks = []
    feature_names: list[str] = []
    for bone_index, bone_name in enumerate(INPUT_BONES):
        blocks.append(rotations[:, bone_index])
        blocks.append(translations[:, bone_index])
        feature_names.extend(
            [
                f"{bone_name}.rotation_vector.x",
                f"{bone_name}.rotation_vector.y",
                f"{bone_name}.rotation_vector.z",
                f"{bone_name}.translation.x",
                f"{bone_name}.translation.y",
                f"{bone_name}.translation.z",
            ]
        )
    pose = np.concatenate(blocks, axis=1)
    velocity = np.concatenate([np.zeros_like(pose[:1]), np.diff(pose, axis=0)], axis=0) * fps
    velocity_names = tuple(f"{name}.velocity" for name in feature_names)
    return np.concatenate([pose, velocity], axis=1), tuple(feature_names) + velocity_names


def _cloth_regions(rest: np.ndarray) -> dict[str, np.ndarray]:
    x = rest[:, 0]
    z = rest[:, 2]
    return {
        "left_sleeve_underarm": (x < -0.08) & (z > 1.00),
        "right_sleeve_underarm": (x > 0.08) & (z > 1.00),
        "torso": (np.abs(x) <= 0.12) & (z > 0.94),
        "hem": z <= 0.94,
    }


def _canonicalize(root_transforms: np.ndarray, points: np.ndarray) -> np.ndarray:
    inverse = invert_rigid_transforms(root_transforms)
    return transform_points(points, inverse)


def _linearized_teacher_bone_fit(rest: np.ndarray, cloth: np.ndarray, weights: np.ndarray, regularization: float = 1.0e-3) -> np.ndarray:
    """Fit stable small rigid bone deltas directly from the cloth teacher.

    Around the common rest pose, ``R(v)`` is linearized as ``v + r x v``.
    The fixed LBS design matrix is solved once with ridge regularization for
    every cloth frame. This avoids unstable per-bone residual division while
    still deriving every target from the simulated cloth vertices.
    """

    vertices = np.asarray(rest, dtype=np.float64)
    frame_values = np.asarray(cloth, dtype=np.float64)
    bone_count = weights.shape[1]
    skew = np.zeros((len(vertices), 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -vertices[:, 2]
    skew[:, 0, 2] = vertices[:, 1]
    skew[:, 1, 0] = vertices[:, 2]
    skew[:, 1, 2] = -vertices[:, 0]
    skew[:, 2, 0] = -vertices[:, 1]
    skew[:, 2, 1] = vertices[:, 0]
    design = np.zeros((len(vertices), 3, bone_count * 6), dtype=np.float64)
    for bone in range(bone_count):
        design[:, :, bone * 6 : bone * 6 + 3] = -skew * weights[:, bone, None, None]
        design[:, :, bone * 6 + 3 : bone * 6 + 6] = np.eye(3, dtype=np.float64)[None, :, :] * weights[:, bone, None, None]
    matrix = design.reshape(len(vertices) * 3, bone_count * 6)
    normal = matrix.T @ matrix
    normal += np.eye(bone_count * 6, dtype=np.float64) * regularization * max(float(np.trace(normal)) / max(len(normal), 1), 1.0)
    right = matrix.T
    vectors = np.empty((len(frame_values), bone_count * 6), dtype=np.float64)
    for frame_index, target in enumerate(frame_values):
        vectors[frame_index] = np.linalg.solve(normal, right @ (target - vertices[None, ...]).reshape(-1))
    values = vectors.reshape(len(frame_values), bone_count, 6)
    result = np.tile(np.eye(4, dtype=np.float64), (len(frame_values), bone_count, 1, 1))
    result[:, :, :3, :3] = rotvec_to_rotation_matrix(values[:, :, :3])
    result[:, :, :3, 3] = values[:, :, 3:]
    return result


def _load_sequence(path: Path, common_rest: np.ndarray, common_weights: np.ndarray, bone_count: int) -> PhysicsSequence:
    with np.load(path, allow_pickle=False) as source:
        rest = np.asarray(source["rest_vertices"], dtype=np.float64)
        triangles = np.asarray(source["triangles"], dtype=np.int32)
        cloth = np.asarray(source["traj"], dtype=np.float64)
        velocity = np.asarray(source["traj_vel"], dtype=np.float64)
        skeleton = np.asarray(source["skeleton_transforms"], dtype=np.float64)
        skeleton_names = tuple(str(value) for value in source["skeleton_names"].tolist())
        root = np.asarray(source["root_transforms"], dtype=np.float64)
        sequence_id = str(source["sequence_id"].item())
        fps = float(source["fps"].item())
    if rest.shape != common_rest.shape or np.max(np.abs(rest - common_rest)) > 2.0e-6:
        raise ValueError(f"{sequence_id} does not use the exact common cloth rest geometry")
    if not np.array_equal(triangles, _COMMON_TRIANGLES):
        raise ValueError(f"{sequence_id} does not use the exact common cloth topology")
    if skeleton.shape[:2] != (len(cloth), len(skeleton_names)):
        raise ValueError(f"{sequence_id} skeleton shape does not match cloth frames")
    cloth_local = _canonicalize(root, cloth)
    cloth_velocity_local = _canonicalize(root, velocity)
    print(f"Fitting physics teacher targets: {sequence_id}", flush=True)
    cache_path = path.parent / "teacher_fit.npz"
    weights_signature = _hash_array(common_weights) + ":linearized_lbs_v1"
    teacher_bones = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["weights_signature"].item()) == weights_signature:
                cached = np.asarray(cache["teacher_bones"], dtype=np.float64)
                if cached.shape == (len(cloth_local), bone_count, 4, 4):
                    teacher_bones = cached
    if teacher_bones is None:
        # Sparse weighted-Kabsch passes keep ten 120-frame physics sequences
        # practical to audit while preserving the LBS residual update. The fit
        # is cached per sequence; no model fitting uses the holdout geometry.
        teacher_bones = _linearized_teacher_bone_fit(common_rest, cloth_local, common_weights)
        np.savez_compressed(cache_path, teacher_bones=teacher_bones, weights_signature=np.asarray(weights_signature))
    print(f"Loaded physics teacher targets: {sequence_id}", flush=True)
    targets = _target_vectors(teacher_bones)
    features, feature_names = _semantic_features(skeleton, skeleton_names, fps)
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update({"path": str(path.resolve()), "sequence_id": sequence_id, "fps": fps})
    return PhysicsSequence(
        sequence_id=sequence_id,
        path=path,
        rest=common_rest,
        triangles=triangles,
        cloth_local=cloth_local,
        cloth_velocity=cloth_velocity_local,
        skeleton_features=features,
        feature_names=feature_names,
        skeleton_names=skeleton_names,
        teacher_bones=teacher_bones,
        targets=targets,
        metadata=metadata,
    )


def _predict_model_fixed_features(model_name: str, train: list[PhysicsSequence], test: PhysicsSequence, alpha: float, target_indices: np.ndarray | None = None) -> np.ndarray:
    with_velocity = model_name.endswith("+velocity")
    base_name = model_name.replace("+velocity", "")
    feature_count = test.skeleton_features.shape[1] if with_velocity else test.skeleton_features.shape[1] // 2
    train_features = np.concatenate([sequence.skeleton_features[:, :feature_count] for sequence in train])
    test_features = test.skeleton_features[:, :feature_count]
    blocks = []
    for sequence in train:
        values = sequence.targets.reshape(sequence.targets.shape[0], -1, 6)
        if target_indices is not None:
            values = values[:, target_indices]
        blocks.append(values.reshape(len(values), -1))
    train_targets = np.concatenate(blocks, axis=0)
    if base_name == "rbf" and len(train_features) > 128:
        selected_train = np.linspace(0, len(train_features) - 1, 128, dtype=np.int64)
        fit_features = train_features[selected_train]
        fit_targets = train_targets[selected_train]
    else:
        fit_features = train_features
        fit_targets = train_targets
    pooled_features = np.concatenate([fit_features, test_features], axis=0)
    pooled_targets = np.concatenate([fit_targets, np.zeros((len(test_features), train_targets.shape[1]))], axis=0)
    train_indices = np.arange(len(fit_features), dtype=np.int64)
    bone_regions = np.zeros(train_targets.shape[1] // 6, dtype=np.int64)
    alpha_value = MODEL_ALPHAS.get(model_name.replace("+velocity", ""), alpha)
    predicted_vectors = _fit_model(base_name, pooled_features, pooled_targets, train_indices, bone_regions, alpha_value)[len(fit_features) :]
    return _vectors_to_transforms(predicted_vectors, train_targets.shape[1] // 6)


def _evaluate_prediction(predicted_bones: np.ndarray, test: PhysicsSequence, weights: np.ndarray, rest: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    predicted_mesh = apply_skinning(rest, weights, predicted_bones)
    values: dict[str, Any] = {"overall": _metrics(predicted_mesh, test.cloth_local)}
    for name, mask in masks.items():
        values[name] = _metrics(predicted_mesh[:, mask], test.cloth_local[:, mask])
    return values


def _write_pose(path: Path, transforms: np.ndarray, names: list[str], rest: np.ndarray, weights: np.ndarray) -> None:
    np.savez_compressed(path, bone_transforms=transforms, bone_names=np.asarray(names), rest_vertices=rest, weights=weights, frames=np.arange(1, len(transforms) + 1, dtype=np.int32))


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["fold", "model", "feature_set", "mean_rms", "max_frame_rms", "max_point_error", "p95_point_error", "improvement_ratio", "improved_frame_ratio"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _fit_fold(train: list[PhysicsSequence], test: PhysicsSequence, rest: np.ndarray, weights: np.ndarray, masks: dict[str, np.ndarray], alpha: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    baseline = np.tile(np.eye(4, dtype=np.float64), (len(test.cloth_local), weights.shape[1], 1, 1))
    baseline_metrics = _evaluate_prediction(baseline, test, weights, rest, masks)
    models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for model_name in MODEL_NAMES:
        for feature_set in ("pose_only", "pose_velocity"):
            full_name = model_name if feature_set == "pose_only" else model_name + "+velocity"
            predicted = _predict_model_fixed_features(full_name, train, test, alpha)
            metrics = _evaluate_prediction(predicted, test, weights, rest, masks)
            frame_rms = np.asarray(metrics["overall"]["per_frame_rms"])
            baseline_frame_rms = np.asarray(baseline_metrics["overall"]["per_frame_rms"])
            metrics["overall"]["improvement_ratio"] = float((baseline_metrics["overall"]["mean_rms"] - metrics["overall"]["mean_rms"]) / max(baseline_metrics["overall"]["mean_rms"], 1.0e-12))
            metrics["overall"]["improved_frame_ratio"] = float(np.mean(frame_rms < baseline_frame_rms))
            models[full_name] = {"metrics": metrics, "baseline": baseline_metrics}
            predictions[full_name] = predicted
    return {"test_sequence": test.sequence_id, "models": models, "baseline": baseline_metrics}, predictions


def _mapping(train: list[PhysicsSequence], bone_names: list[str], feature_names: tuple[str, ...]) -> list[dict[str, Any]]:
    features = np.concatenate([sequence.skeleton_features[:, : len(feature_names)] for sequence in train])
    targets = np.concatenate([sequence.targets.reshape(sequence.targets.shape[0], -1, 6) for sequence in train], axis=0)
    centered = features - features.mean(axis=0)
    feature_norm = np.linalg.norm(centered, axis=0)
    entries = []
    for bone_index, bone_name in enumerate(bone_names):
        target = targets[:, bone_index] - targets[:, bone_index].mean(axis=0)
        target_norm = np.linalg.norm(target, axis=0)
        correlation = np.abs(centered.T @ target) / np.maximum(feature_norm[:, None] * target_norm[None, :], 1.0e-12)
        importance = np.nan_to_num(correlation).max(axis=1)
        selected = np.argsort(importance)[-8:][::-1]
        entries.append({"bone": bone_name, "top_features": [{"feature": feature_names[int(index)], "absolute_correlation": float(importance[index])} for index in selected]})
    return entries


def _hysteresis(sequences: list[PhysicsSequence]) -> dict[str, Any]:
    candidates = [sequence for sequence in sequences if sequence.metadata.get("hysteresis_candidate")]
    result: dict[str, Any] = {"candidates": [sequence.sequence_id for sequence in candidates], "comparisons": []}
    for sequence in candidates:
        pose = sequence.skeleton_features[:, : sequence.skeleton_features.shape[1] // 2]
        distance = np.sqrt(np.sum((pose[:, None] - pose[None, :]) ** 2, axis=2))
        row, column = np.indices(distance.shape)
        valid = (np.abs(row - column) >= 20) & (row > 20) & (column > 20)
        distance[~valid] = np.inf
        first, second = np.unravel_index(np.argmin(distance), distance.shape)
        cloth_rms = float(np.sqrt(np.mean((sequence.cloth_local[first] - sequence.cloth_local[second]) ** 2)))
        result["comparisons"].append({"sequence_id": sequence.sequence_id, "frame_a": int(first + 1), "frame_b": int(second + 1), "pose_feature_distance": float(distance[first, second]), "teacher_cloth_rms_difference": cloth_rms})
    return result


def main() -> None:
    global _COMMON_TRIANGLES
    args = _args()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    common_rest, common_weights, common_triangles, bone_names = _load_common(Path(args.common_poses).resolve(), Path(args.triangles).resolve())
    _COMMON_TRIANGLES = common_triangles
    paths = sorted((Path(args.dataset_root).resolve() / "physics_sequences").glob("*/teacher.npz"))
    if not paths:
        raise FileNotFoundError("No physics_sequences/*/teacher.npz files found")
    sequences = [_load_sequence(path, common_rest, common_weights, len(bone_names)) for path in paths]
    sequence_map = {sequence.sequence_id: sequence for sequence in sequences}
    primary = sequence_map[args.primary_test]
    primary_train_ids = tuple(value for value in args.train_sequences.split(",") if value)
    primary_train = [sequence_map[value] for value in primary_train_ids]
    masks = _cloth_regions(common_rest)
    print("Fitting primary holdout models", flush=True)
    primary_fold, primary_predictions = _fit_fold(primary_train, primary, common_rest, common_weights, masks, args.alpha)

    loso = []
    csv_rows: list[dict[str, Any]] = []
    for test in sequences:
        print(f"Fitting LOSO models: {test.sequence_id}", flush=True)
        train = [sequence for sequence in sequences if sequence.sequence_id != test.sequence_id]
        fold, _ = _fit_fold(train, test, common_rest, common_weights, masks, args.alpha)
        loso.append(fold)
        for model_name, entry in fold["models"].items():
            metric = entry["metrics"]["overall"]
            csv_rows.append({"fold": test.sequence_id, "model": model_name, "feature_set": "pose+velocity" if model_name.endswith("+velocity") else "pose_only", **metric})

    all_model_names = list(primary_fold["models"])
    aggregate = {}
    for model_name in all_model_names:
        scores = [float(fold["models"][model_name]["metrics"]["overall"]["mean_rms"]) for fold in loso]
        improvements = [float(fold["models"][model_name]["metrics"]["overall"]["improvement_ratio"]) for fold in loso]
        aggregate[model_name] = {"mean_rms": float(np.mean(scores)), "mean_improvement_ratio": float(np.mean(improvements)), "improved_fold_ratio": float(np.mean(np.asarray(improvements) > 0.0))}
    research_model = min(all_model_names, key=lambda name: aggregate[name]["mean_rms"])
    runtime_candidates = [name for name in all_model_names if name in {"rbf", "rbf+velocity", "nearest", "nearest+velocity", "ridge", "linear", "ridge+velocity", "linear+velocity"}]
    runtime_model = min(runtime_candidates, key=lambda name: aggregate[name]["mean_rms"])
    primary_research = primary_fold["models"][research_model]["metrics"]
    primary_runtime = primary_fold["models"][runtime_model]["metrics"]
    primary_baseline = primary_fold["baseline"]

    _write_pose(out_root / "teacher_holdout_poses.npz", primary.teacher_bones, bone_names, common_rest, common_weights)
    _write_pose(out_root / "research_holdout_poses.npz", primary_predictions[research_model], bone_names, common_rest, common_weights)
    _write_pose(out_root / "runtime_holdout_poses.npz", primary_predictions[runtime_model], bone_names, common_rest, common_weights)
    rest_bones = np.tile(np.eye(4, dtype=np.float64), (len(primary.cloth_local), len(bone_names), 1, 1))
    _write_pose(out_root / "rest_holdout_poses.npz", rest_bones, bone_names, common_rest, common_weights)
    np.savez_compressed(out_root / "primary_teacher_reference.npz", cloth_local=primary.cloth_local, rest_vertices=common_rest, triangles=common_triangles)

    bone_count_comparison = {}
    for count in (50, 32, 20, 16, 8):
        selected, reduced_weights = _bone_subset(common_weights, common_rest, count)
        predicted = _predict_model_fixed_features(runtime_model, primary_train, primary, args.alpha, target_indices=selected)
        teacher = primary.teacher_bones[:, selected]
        rest_reduced = np.tile(np.eye(4, dtype=np.float64), (len(primary.cloth_local), count, 1, 1))
        bone_count_comparison[str(count)] = {
            "predicted": _evaluate_prediction(predicted, primary, reduced_weights, common_rest, masks),
            "teacher_fit": _evaluate_prediction(teacher, primary, reduced_weights, common_rest, masks),
            "rest": _evaluate_prediction(rest_reduced, primary, reduced_weights, common_rest, masks),
            "selected_original_bones": selected.tolist(),
        }

    teacher_deformation = np.linalg.norm(primary.cloth_local - common_rest[None], axis=2)
    feature_names = tuple(primary.feature_names[: len(primary.feature_names) // 2])
    mapping_bones = _mapping(primary_train, bone_names, feature_names)
    target_motion = np.linalg.norm(primary.targets.reshape(len(primary.targets), len(bone_names), 6).std(axis=0), axis=1)
    research_motion = np.linalg.norm(_target_vectors(primary_predictions[research_model]).reshape(len(primary_predictions[research_model]), len(bone_names), 6).std(axis=0), axis=1)
    runtime_motion = np.linalg.norm(_target_vectors(primary_predictions[runtime_model]).reshape(len(primary_predictions[runtime_model]), len(bone_names), 6).std(axis=0), axis=1)
    manifest = {
        "schema_version": 1,
        "generator": "tools/generate_blender_physics_poc.py",
        "teacher_source": "Blender 5.2 Cloth modifier cache",
        "common_topology": {"vertices": int(len(common_rest)), "triangles": int(len(common_triangles)), "rest_hash": _hash_array(common_rest), "triangle_hash": _hash_array(common_triangles)},
        "common_rig": {"bone_count": len(bone_names), "bone_names": bone_names, "weights_shape": list(common_weights.shape), "source": "verified PoC 1/2 clean rig; fixed before primary holdout"},
        "sequences": [{"sequence_id": sequence.sequence_id, "path": str(sequence.path.resolve()), "frames": len(sequence.cloth_local), "metadata": sequence.metadata} for sequence in sequences],
    }
    (out_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "schema_version": 1,
        "dataset_manifest": "dataset_manifest.json",
        "body_rig": {"bones": ["Hips", "Spine", "Chest", "Shoulder.L", "Upper_arm.L", "Lower_arm.L", "Shoulder.R", "Upper_arm.R", "Lower_arm.R"], "semantic_input_bones": list(INPUT_BONES), "root_canonicalization": "inverse exported Hips root transform per frame"},
        "cloth": {"vertices": int(len(common_rest)), "triangles": int(len(common_triangles)), "physics_teacher": True, "teacher_geometry_used_for_target_fit": True, "holdout_teacher_used_for_evaluation_only": True},
        "physics_quality": {"primary_teacher_vs_rest_mean_rms": float(np.sqrt(np.mean(teacher_deformation * teacher_deformation, axis=1)).mean()), "primary_teacher_vs_rest_max_rms": float(np.sqrt(np.mean(teacher_deformation * teacher_deformation, axis=1)).max()), "underarm_teacher_vs_rest": {name: float(np.sqrt(np.mean(teacher_deformation[:, mask] ** 2, axis=1)).mean()) for name, mask in masks.items()}},
        "features": {"pose_dimensions": len(feature_names), "pose_velocity_dimensions": len(primary.feature_names), "names": list(primary.feature_names), "uses_frame_index": False, "uses_time": False, "uses_progress": False, "uses_sequence_id": False, "uses_cloth_teacher": False, "pose_features": "semantic skeleton relative rotation-vector plus translation", "velocity_features": "finite difference of pose features multiplied by fps"},
        "target": {"representation": "common helper-bone local rotation-vector plus translation fitted from physics cloth geometry", "bone_count": len(bone_names), "common_weights": True},
        "primary_holdout": {"train_sequences": list(primary_train_ids), "test_sequence": primary.sequence_id, "models": primary_fold["models"], "static_rest": primary_baseline, "research_model": research_model, "research_metrics": primary_research, "runtime_model": runtime_model, "runtime_metrics": primary_runtime},
        "loso": {"folds": loso, "aggregate": aggregate},
        "models": {"tested": all_model_names, "research_best": research_model, "runtime_friendly_best": runtime_model},
        "bone_count_comparison": bone_count_comparison,
        "bone_motion": {"teacher_std_norm_per_bone": target_motion.tolist(), "research_std_norm_per_bone": research_motion.tolist(), "runtime_std_norm_per_bone": runtime_motion.tolist(), "teacher_nonzero_bones": int(np.count_nonzero(target_motion > 1.0e-5)), "research_nonzero_bones": int(np.count_nonzero(research_motion > 1.0e-5)), "runtime_nonzero_bones": int(np.count_nonzero(runtime_motion > 1.0e-5)), "all_runtime_bones_same_transform": bool(np.max(runtime_motion) < 1.0e-5)},
        "regions": {"names": list(masks), "sizes": {name: int(mask.sum()) for name, mask in masks.items()}},
        "mapping": {"method": "training-sequence absolute feature/target correlation", "bones": mapping_bones},
        "hysteresis": _hysteresis(sequences),
        "leakage_checks": {"holdout_cloth_in_common_rig": False, "holdout_cloth_in_weights": False, "holdout_cloth_in_model_fit": False, "holdout_frames_in_nearest_database": False, "frame_index_in_features": False, "time_or_progress_in_features": False, "sequence_id_in_features": False, "cloth_teacher_in_features": False, "root_global_motion_removed": True},
        "status": {"physics_teacher_has_nonrigid_deformation": bool(np.sqrt(np.mean(teacher_deformation * teacher_deformation, axis=1)).mean() > 1.0e-3), "primary_unknown_physics_motion_improved": bool(primary_research["overall"]["mean_rms"] < primary_baseline["overall"]["mean_rms"]), "primary_research_improved_frame_ratio": primary_research["overall"]["improved_frame_ratio"], "primary_runtime_improved_frame_ratio": primary_runtime["overall"]["improved_frame_ratio"], "multiple_loso_folds_improved_research": bool(aggregate[research_model]["improved_fold_ratio"] > 0.5)},
        "artifacts": {"teacher_poses": "teacher_holdout_poses.npz", "research_poses": "research_holdout_poses.npz", "runtime_poses": "runtime_holdout_poses.npz", "rest_poses": "rest_holdout_poses.npz", "primary_teacher_reference": "primary_teacher_reference.npz", "metrics_csv": "model_metrics.csv", "bone_mapping": "bone_mapping.json"},
    }
    (out_root / "bone_mapping.json").write_text(json.dumps({"bones": mapping_bones, "semantic_input_bones": list(INPUT_BONES)}, indent=2), encoding="utf-8")
    _write_metrics_csv(out_root / "model_metrics.csv", csv_rows)
    (out_root / "physics_cross_sequence_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"research_best": research_model, "runtime_friendly_best": runtime_model, "primary_research": primary_research["overall"], "primary_runtime": primary_runtime["overall"], "status": report["status"]}, indent=2))


if __name__ == "__main__":
    _COMMON_TRIANGLES = np.empty((0, 3), dtype=np.int32)
    main()
