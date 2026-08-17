from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from audit_cross_sequence_dataset import _topology_digests  # noqa: E402
from cross_sequence_poc import _fit_model, _metrics, _spatial_regions, _vectors_to_transforms  # noqa: E402


def test_topology_canonical_hash_ignores_face_order_and_winding() -> None:
    first = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    second = np.asarray([[2, 0, 1], [3, 2, 1]], dtype=np.int64)
    first_hashes = _topology_digests(first)
    second_hashes = _topology_digests(second)
    assert first_hashes["raw"] != second_hashes["raw"]
    assert first_hashes["canonical_connectivity"] == second_hashes["canonical_connectivity"]


def test_fixed_regions_are_deterministic_and_cover_body() -> None:
    rest = np.asarray([[x, y, z] for x in (-1.0, -0.5, 0.5, 1.0) for y in (0.0, 1.0) for z in (0.0, 0.5)], dtype=np.float64)
    first = _spatial_regions(rest, 4)
    second = _spatial_regions(rest, 4)
    assert np.array_equal(first.labels, second.labels)
    assert sum(int(mask.sum()) for mask in first.masks) == len(rest)
    assert all(int(mask.sum()) > 0 for mask in first.masks)


def test_cross_sequence_ridge_uses_train_frames_only() -> None:
    train_features = np.asarray([[value, value * value] for value in np.linspace(-1.0, 1.0, 12)])
    train_targets = np.concatenate([train_features[:, :1] * 0.25, train_features[:, :1] * -0.4], axis=1)
    test_features = np.asarray([[0.25, 0.25**2], [-0.75, 0.75**2]])
    pooled_features = np.concatenate([train_features, test_features])
    pooled_targets = np.concatenate([train_targets, np.zeros((len(test_features), 2))])
    prediction = _fit_model("ridge", pooled_features, pooled_targets, np.arange(len(train_features)), np.zeros(1, dtype=np.int64), 1.0e-8)
    assert np.allclose(prediction[-2:, 0], test_features[:, 0] * 0.25, atol=1.0e-3)
    assert np.allclose(prediction[-2:, 1], test_features[:, 0] * -0.4, atol=1.0e-3)


def test_prediction_vectors_project_to_rigid_transforms_and_metrics_are_complete() -> None:
    vectors = np.zeros((3, 2, 6), dtype=np.float64)
    vectors[:, 0, 3] = [0.0, 0.1, 0.2]
    transforms = _vectors_to_transforms(vectors.reshape(3, -1), 2)
    assert transforms.shape == (3, 2, 4, 4)
    assert np.allclose(np.linalg.det(transforms[:, :, :3, :3]), 1.0)
    metrics = _metrics(np.zeros((3, 4, 3)), np.ones((3, 4, 3)))
    assert len(metrics["per_frame_rms"]) == 3
    assert "max_point_error" in metrics
