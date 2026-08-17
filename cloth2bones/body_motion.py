"""Body-motion features and deterministic cloth-bone regression.

The module intentionally has no Blender dependency.  It consumes the body
collider vertices from a ClothTransformer-style NPZ and bone transforms
exported by Blender, then predicts bone transforms from body motion only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np


@dataclass(frozen=True)
class DriverLayout:
    """Stable region masks used to derive body drivers."""

    names: tuple[str, ...]
    masks: tuple[np.ndarray, ...]


def infer_axis_transform(source_rest: np.ndarray, target_rest: np.ndarray) -> np.ndarray:
    """Infer a signed axis-permutation matrix from same-index rest points."""

    source = np.asarray(source_rest, dtype=np.float64)
    target = np.asarray(target_rest, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Axis inference expects two arrays with shape (N, 3), got {source.shape} and {target.shape}")
    best_error = float("inf")
    best_matrix = None
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for output_axis, source_axis in enumerate(permutation):
                matrix[output_axis, source_axis] = signs[output_axis]
            candidate = source @ matrix.T
            offset = target.mean(axis=0) - candidate.mean(axis=0)
            error = float(np.sqrt(np.mean((candidate + offset - target) ** 2)))
            if error < best_error:
                best_error = error
                best_matrix = matrix
    if best_matrix is None or best_error > 1.0e-4:
        raise ValueError(f"Could not infer a signed axis permutation; best rest RMS was {best_error}")
    return best_matrix


def _region_masks(rest: np.ndarray) -> DriverLayout:
    """Create deterministic torso/side masks from a rest collider.

    Human Garment colliders are Y-up.  The positive and negative X side masks
    are deliberately named by side rather than left/right because the source
    dataset does not provide semantic joint labels.
    """

    if rest.ndim != 2 or rest.shape[1] != 3:
        raise ValueError(f"Expected collider rest points with shape (N, 3), got {rest.shape}")
    x, y, _ = rest.T
    width = float(np.ptp(x))
    height = float(np.ptp(y))
    x_threshold = max(0.18, width * 0.25)
    y_low = float(y.min()) + height * 0.15
    y_high = float(y.max()) - height * 0.05
    masks = (
        np.ones(rest.shape[0], dtype=bool),
        (np.abs(x) <= x_threshold) & (y >= y_low) & (y <= y_high),
        x > x_threshold,
        x < -x_threshold,
    )
    names = ("body", "torso", "side_positive_x", "side_negative_x")
    fallback = []
    for mask in masks:
        if int(mask.sum()) < 12:
            fallback.append(False)
        else:
            fallback.append(True)
    if not all(fallback):
        side_cut = max(width * 0.18, 1.0e-6)
        masks = (
            masks[0],
            np.abs(x) <= side_cut,
            x >= np.quantile(x, 0.70),
            x <= np.quantile(x, 0.30),
        )
    if any(int(mask.sum()) < 3 for mask in masks):
        raise ValueError("Collider masks must each contain at least three points for Kabsch fitting")
    return DriverLayout(names=names, masks=tuple(masks))


def kabsch_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the rigid transform mapping source points to target points."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Kabsch inputs must both have shape (N, 3), got {source.shape} and {target.shape}")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def body_driver_features(collision_vertices: np.ndarray) -> tuple[np.ndarray, DriverLayout]:
    """Fit region transforms and encode them as frame-wise body features.

    For each region, the nine rotation entries are encoded as ``R - I`` and
    the three translations are kept in world units.  The first feature row is
    therefore close to zero for an unchanged rest frame, which is convenient
    for a linear mapping with an intercept.
    """

    vertices = np.asarray(collision_vertices, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError(f"Expected collider animation with shape (T, N, 3), got {vertices.shape}")
    layout = _region_masks(vertices[0])
    feature_blocks = []
    for mask in layout.masks:
        source = vertices[0, mask]
        transforms = np.stack([kabsch_transform(source, frame[mask]) for frame in vertices], axis=0)
        rotation_delta = transforms[:, :3, :3] - np.eye(3, dtype=np.float64)[None, :, :]
        translation = transforms[:, :3, 3]
        feature_blocks.append(np.concatenate([rotation_delta.reshape(len(vertices), 9), translation], axis=1))
    return np.concatenate(feature_blocks, axis=1), layout


def apply_skinning(rest_vertices: np.ndarray, weights: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Evaluate linear blend skinning for one or more transform frames."""

    rest = np.asarray(rest_vertices, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    t = np.asarray(transforms, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1] != 3:
        raise ValueError("rest_vertices must have shape (V, 3)")
    if w.shape != (rest.shape[0], t.shape[-3]):
        raise ValueError(f"weights shape {w.shape} does not match vertices/bones {(rest.shape[0], t.shape[-3])}")
    if t.ndim == 3:
        t = t[None, ...]
    if t.ndim != 4 or t.shape[1:] != (w.shape[1], 4, 4):
        raise ValueError(f"transforms must have shape (T, B, 4, 4), got {t.shape}")
    transformed = np.einsum("tbij,vj->tbvi", t[:, :, :3, :3], rest) + t[:, :, :3, 3][:, :, None, :]
    return np.einsum("vb,tbvi->tvi", w, transformed)


def fit_ridge_mapping(features: np.ndarray, targets: np.ndarray, train_indices: np.ndarray, alpha: float) -> np.ndarray:
    """Fit an intercept-plus-linear ridge mapping from body features to targets."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    train = np.asarray(train_indices, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("features and targets must be two-dimensional arrays with equal frame counts")
    if not len(train) or np.any(train < 0) or np.any(train >= len(x)):
        raise ValueError("train_indices must select at least one valid frame")
    design = np.concatenate([np.ones((len(train), 1)), x[train]], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    regularizer[0, 0] = 0.0
    lhs = design.T @ design + regularizer
    rhs = design.T @ y[train]
    return np.linalg.solve(lhs, rhs)


def predict_ridge_mapping(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict target vectors from a fitted intercept-plus-linear mapping."""

    x = np.asarray(features, dtype=np.float64)
    c = np.asarray(coefficients, dtype=np.float64)
    return np.concatenate([np.ones((len(x), 1)), x], axis=1) @ c


def project_transform_vectors(vectors: np.ndarray, bone_count: int) -> np.ndarray:
    """Convert predicted 12-value bone blocks to valid rigid matrices."""

    values = np.asarray(vectors, dtype=np.float64).reshape(len(vectors), bone_count, 12)
    transforms = np.tile(np.eye(4, dtype=np.float64), (len(values), bone_count, 1, 1))
    for frame_index in range(len(values)):
        for bone_index in range(bone_count):
            rotation = values[frame_index, bone_index, :9].reshape(3, 3)
            u, _, vt = np.linalg.svd(rotation)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0.0:
                u[:, -1] *= -1.0
                rotation = u @ vt
            transforms[frame_index, bone_index, :3, :3] = rotation
            transforms[frame_index, bone_index, :3, 3] = values[frame_index, bone_index, 9:12]
    return transforms


def frame_metrics(predicted: np.ndarray, teacher: np.ndarray) -> dict[str, float]:
    """Return RMS and maximum point error for two vertex arrays."""

    distances = np.linalg.norm(np.asarray(predicted) - np.asarray(teacher), axis=1)
    return {
        "rms": float(np.sqrt(np.mean(distances * distances))),
        "max_point_error": float(np.max(distances)),
        "mean": float(np.mean(distances)),
    }
