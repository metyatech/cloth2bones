"""Fit cloth bone transforms from collider motion and evaluate held-out frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloth2bones.body_motion import (  # noqa: E402
    apply_skinning,
    body_driver_features,
    fit_ridge_mapping,
    frame_metrics,
    infer_axis_transform,
    kabsch_transform,
    predict_ridge_mapping,
    project_transform_vectors,
)


def _target_vectors(transforms: np.ndarray) -> np.ndarray:
    rotation_delta = transforms[:, :, :3, :3] - np.eye(3, dtype=np.float64)[None, None, :, :]
    translation = transforms[:, :, :3, 3]
    return np.concatenate([rotation_delta.reshape(len(transforms), transforms.shape[1], 9), translation], axis=2).reshape(len(transforms), -1)


def _aggregate_metrics(predicted: np.ndarray, teacher: np.ndarray, indices: np.ndarray) -> dict[str, object]:
    values = [frame_metrics(predicted[i], teacher[i]) for i in indices]
    rms = np.asarray([value["rms"] for value in values], dtype=np.float64)
    max_error = np.asarray([value["max_point_error"] for value in values], dtype=np.float64)
    return {
        "frames": int(len(values)),
        "mean_rms": float(rms.mean()),
        "max_rms": float(rms.max()),
        "mean_max_point_error": float(max_error.mean()),
        "max_point_error": float(max_error.max()),
        "per_frame": [dict(frame=int(i + 1), **values[position]) for position, i in enumerate(indices)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Body collider motion -> cloth bone regression PoC")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True, help="NPZ exported by dump_blender_rig_poses.py")
    parser.add_argument("--out", type=Path, required=True, help="Predicted bone transform NPZ")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--model", choices=("global-kabsch", "linear"), default="global-kabsch")
    args = parser.parse_args()
    with np.load(args.npz, allow_pickle=False) as source:
        collision = np.asarray(source["collision_vertices"], dtype=np.float64)
        teacher_vertices = np.asarray(source["traj"], dtype=np.float64)
    with np.load(args.poses, allow_pickle=False) as pose_data:
        transforms = np.asarray(pose_data["bone_transforms"], dtype=np.float64)
        rest_vertices = np.asarray(pose_data["rest_vertices"], dtype=np.float64)
        weights = np.asarray(pose_data["weights"], dtype=np.float64)
        bone_names = [str(name) for name in pose_data["bone_names"].tolist()]
    axis_matrix = infer_axis_transform(teacher_vertices[0], rest_vertices)
    axis_offset = rest_vertices.mean(axis=0) - (teacher_vertices[0] @ axis_matrix.T).mean(axis=0)
    teacher_vertices = teacher_vertices @ axis_matrix.T + axis_offset
    collision = collision @ axis_matrix.T + axis_offset
    if len(collision) != len(teacher_vertices) or len(transforms) != len(teacher_vertices):
        raise ValueError("Body, cloth, and bone pose sequences must have the same frame count")
    if teacher_vertices.shape[1:] != rest_vertices.shape:
        raise ValueError(f"Cloth rest shape {rest_vertices.shape} does not match trajectory {teacher_vertices.shape[1:]}")
    if not np.allclose(rest_vertices, teacher_vertices[0], atol=2.0e-4):
        raise ValueError("Blender pose export rest vertices do not match NPZ traj[0]")
    features, layout = body_driver_features(collision)
    frame_count = len(features)
    train_count = max(3, min(frame_count - 1, int(round(frame_count * args.train_fraction))))
    train_indices = np.arange(train_count, dtype=np.int64)
    validation_indices = np.arange(train_count, frame_count, dtype=np.int64)
    global_transforms = np.stack([kabsch_transform(collision[0], frame) for frame in collision], axis=0)
    if args.model == "global-kabsch":
        predicted = np.broadcast_to(global_transforms[:, None], transforms.shape).copy()
    else:
        target_vectors = _target_vectors(transforms)
        coefficients = fit_ridge_mapping(features, target_vectors, train_indices, args.alpha)
        predicted = project_transform_vectors(predict_ridge_mapping(features, coefficients), len(bone_names))
    predicted[0] = transforms[0]
    predicted_skin = apply_skinning(rest_vertices, weights, predicted)
    baseline = np.broadcast_to(transforms[0], transforms.shape).copy()
    baseline_skin = apply_skinning(rest_vertices, weights, baseline)
    report = {
        "npz": str(args.npz.resolve()),
        "poses": str(args.poses.resolve()),
        "frames": frame_count,
        "bones": len(bone_names),
        "bone_names": bone_names,
        "driver_regions": list(layout.names),
        "driver_feature_count": int(features.shape[1]),
        "train_frames": [1, train_count],
        "validation_frames": [train_count + 1, frame_count],
        "model": args.model,
        "alpha": args.alpha,
        "source_to_blender_axis_matrix": axis_matrix.tolist(),
        "source_to_blender_axis_offset": axis_offset.tolist(),
        "baseline": {
            "all": _aggregate_metrics(baseline_skin, teacher_vertices, np.arange(frame_count)),
            "validation": _aggregate_metrics(baseline_skin, teacher_vertices, validation_indices),
        },
        "model_metrics": {
            "all": _aggregate_metrics(predicted_skin, teacher_vertices, np.arange(frame_count)),
            "train": _aggregate_metrics(predicted_skin, teacher_vertices, train_indices),
            "validation": _aggregate_metrics(predicted_skin, teacher_vertices, validation_indices),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, bone_transforms=predicted, bone_names=np.asarray(bone_names), features=features)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
