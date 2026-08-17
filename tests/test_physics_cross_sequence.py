from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from physics_cross_sequence_poc import _cloth_regions, _linearized_teacher_bone_fit, _semantic_features  # noqa: E402

from cloth2bones.body_motion import apply_skinning  # noqa: E402


def test_semantic_features_have_pose_and_velocity_without_frame_feature() -> None:
    names = ("Hips", "Spine", "Chest", "Shoulder.L", "Upper_arm.L", "Lower_arm.L", "Shoulder.R", "Upper_arm.R", "Lower_arm.R")
    transforms = np.tile(np.eye(4, dtype=np.float64), (4, len(names), 1, 1))
    transforms[2, names.index("Upper_arm.L"), :3, 3] = (0.1, 0.0, 0.0)
    features, feature_names = _semantic_features(transforms, names, 30.0)
    assert features.shape == (4, 60)
    assert len(feature_names) == 60
    assert np.allclose(features[0, 30:], 0.0)
    assert not any("frame" in name or "time" in name or "progress" in name for name in feature_names)


def test_linearized_teacher_fit_is_finite_and_reconstructs_small_lbs_motion() -> None:
    rest = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    weights = np.zeros((len(rest), 2), dtype=np.float64)
    weights[:4, 0] = 1.0
    weights[4:, 1] = 1.0
    transforms = np.tile(np.eye(4, dtype=np.float64), (3, 2, 1, 1))
    transforms[:, 0, :3, 3] = (0.02, -0.01, 0.03)
    transforms[:, 1, :3, 3] = (-0.01, 0.04, 0.01)
    teacher = apply_skinning(rest, weights, transforms)
    fitted = _linearized_teacher_bone_fit(rest, teacher, weights, regularization=1.0e-8)
    reconstructed = apply_skinning(rest, weights, fitted)
    assert np.isfinite(fitted).all()
    assert float(np.sqrt(np.mean((reconstructed - teacher) ** 2))) < 1.0e-4


def test_cloth_regions_cover_expected_vertices() -> None:
    rest = np.asarray([[x, 0.0, z] for x in (-0.2, -0.1, 0.0, 0.1, 0.2) for z in (0.8, 1.1)], dtype=np.float64)
    regions = _cloth_regions(rest)
    assert all(mask.dtype == bool for mask in regions.values())
    assert all(int(mask.sum()) > 0 for mask in regions.values())
