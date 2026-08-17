from __future__ import annotations

import numpy as np

from cloth2bones.body_motion import (
    apply_skinning,
    body_driver_features,
    fit_ridge_mapping,
    infer_axis_transform,
    invert_rigid_transforms,
    kabsch_transform,
    predict_ridge_mapping,
    project_transform_vectors,
    rotation_matrix_to_rotvec,
    rotvec_to_rotation_matrix,
    transform_points,
    validate_rig_pose_contract,
)


def test_kabsch_recovers_rigid_transform() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    target = source @ np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T + np.asarray([2.0, -1.0, 0.5])
    transform = kabsch_transform(source, target)
    actual = source @ transform[:3, :3].T + transform[:3, 3]
    assert np.allclose(actual, target, atol=1.0e-10)


def test_body_features_and_ridge_do_not_use_frame_index() -> None:
    rest = np.asarray(
        [[x, y, z] for x in (-0.3, -0.1, 0.1, 0.3) for y in (0.2, 0.8, 1.4) for z in (-0.1, 0.1)],
        dtype=np.float64,
    )
    collision = np.stack([rest + np.asarray([0.0, 0.0, frame * 0.05]) for frame in range(8)])
    features, layout = body_driver_features(collision)
    assert features.shape[0] == 8
    assert features.shape[1] == 48
    assert layout.names[0] == "body"
    targets = np.concatenate([features[:, :9], features[:, 9:12]], axis=1)
    coefficients = fit_ridge_mapping(features, targets, np.arange(6), 1.0e-8)
    predictions = predict_ridge_mapping(features, coefficients)
    assert np.allclose(predictions[6:], targets[6:], atol=1.0e-6)


def test_projected_transforms_skin_vertices() -> None:
    raw = np.zeros((2, 12), dtype=np.float64)
    raw[:, :9] = np.eye(3).reshape(1, 9)
    raw[1, 9:] = [1.0, 2.0, 3.0]
    transforms = project_transform_vectors(raw, 1)
    vertices = apply_skinning(np.asarray([[0.0, 0.0, 0.0]]), np.ones((1, 1)), transforms)
    assert np.allclose(vertices[1, 0], [1.0, 2.0, 3.0])

    rotated = np.zeros((1, 12), dtype=np.float64)
    rotated[0, :9] = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).reshape(-1)
    rotation_transform = project_transform_vectors(rotated, 1)
    rotated_vertex = apply_skinning(np.asarray([[1.0, 0.0, 0.0]]), np.ones((1, 1)), rotation_transform)
    assert np.allclose(rotated_vertex[0, 0], [0.0, 1.0, 0.0])


def test_infer_signed_axis_permutation() -> None:
    source = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    matrix = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    target = source @ matrix.T
    assert np.allclose(infer_axis_transform(source, target), matrix)


def test_rigid_canonicalization_removes_global_motion() -> None:
    rest = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    transforms = np.tile(np.eye(4), (2, 1, 1))
    transforms[1, :3, :3] = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transforms[1, :3, 3] = [4.0, -2.0, 1.0]
    animated = transform_points(rest, transforms)
    recovered = np.stack([kabsch_transform(rest, frame) for frame in animated])
    canonical = transform_points(animated, invert_rigid_transforms(recovered))
    assert np.allclose(canonical, rest[None, ...], atol=1.0e-10)


def test_rotation_vector_round_trip() -> None:
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    vector = rotation_matrix_to_rotvec(rotation)
    assert np.allclose(rotvec_to_rotation_matrix(vector), rotation, atol=1.0e-10)


def test_pose_contract_rejects_mismatched_weights() -> None:
    rest = np.zeros((2, 3), dtype=np.float64)
    weights = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    mismatched = weights[:, ::-1]
    try:
        validate_rig_pose_contract(rest, mismatched, rest, weights)
    except ValueError as error:
        assert "weights" in str(error)
    else:
        raise AssertionError("Mismatched rig weights were accepted")
