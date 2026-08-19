from __future__ import annotations

import numpy as np
import pytest

from cloth2bones.pbd_settle import PBDSettleParameters, settle_pbd


def _parameters(**overrides: object) -> PBDSettleParameters:
    values: dict[str, object] = {
        "frames": 0,
        "dt": 1.0 / 60.0,
        "solver_iterations": 1,
        "collision_interval": 1,
        "final_projection_iterations": 0,
        "gravity": 0.0,
        "damping": 1.0,
        "stretch_stiffness": 1.0,
        "bend_stiffness": 1.0,
        "attachment_stiffness": 1.0,
    }
    values.update(overrides)
    return PBDSettleParameters(**values)


def _identity_collision(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return positions, np.zeros(len(positions), dtype=bool)


def test_single_edge_projection_reduces_rest_length_error() -> None:
    rest = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    base = np.asarray([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]])
    result = settle_pbd(rest, base, np.asarray([[0, 1]]), np.empty((0, 2)), np.zeros(2), np.zeros(2, dtype=bool), _parameters(frames=1), _identity_collision)
    initial_error = abs(np.linalg.norm(base[1] - base[0]) - 1.0)
    final_error = abs(np.linalg.norm(result.positions[1] - result.positions[0]) - 1.0)
    assert final_error < initial_error


def test_hard_pin_endpoint_is_completely_unchanged() -> None:
    rest = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = settle_pbd(rest, rest, np.asarray([[0, 1]]), np.empty((0, 2)), np.zeros(2), np.asarray([True, False]), _parameters(frames=4, gravity=-9.81), _identity_collision)
    assert np.array_equal(result.positions[0], rest[0])
    assert np.array_equal(result.velocity[0], np.zeros(3))


def test_full_attachment_returns_gravity_displaced_vertex_toward_base() -> None:
    rest = np.asarray([[0.0, 0.0, 0.0]])
    base = rest.copy()
    attached = settle_pbd(rest, base, np.empty((0, 2)), np.empty((0, 2)), np.ones(1), np.zeros(1, dtype=bool), _parameters(frames=1, gravity=-9.81, attachment_stiffness=0.5), _identity_collision)
    free = settle_pbd(rest, base, np.empty((0, 2)), np.empty((0, 2)), np.zeros(1), np.zeros(1, dtype=bool), _parameters(frames=1, gravity=-9.81, attachment_stiffness=0.0), _identity_collision)
    assert abs(attached.positions[0, 2] - base[0, 2]) < abs(free.positions[0, 2] - base[0, 2])


def test_bend_pair_projection_reduces_rest_distance_error() -> None:
    rest = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    base = np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    result = settle_pbd(rest, base, np.empty((0, 2)), np.asarray([[0, 1]]), np.zeros(2), np.zeros(2, dtype=bool), _parameters(), _identity_collision)
    assert abs(np.linalg.norm(result.positions[1] - result.positions[0]) - 1.0) < 0.4


def test_gravity_moves_unpinned_vertex_downward() -> None:
    rest = np.asarray([[0.0, 0.0, 1.0]])
    result = settle_pbd(rest, rest, np.empty((0, 2)), np.empty((0, 2)), np.zeros(1), np.zeros(1, dtype=bool), _parameters(frames=1, gravity=-9.81), _identity_collision)
    assert result.positions[0, 2] < rest[0, 2]


def test_hard_pin_stays_exact_after_multiple_frames() -> None:
    rest = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = settle_pbd(rest, rest, np.asarray([[0, 1]]), np.empty((0, 2)), np.zeros(2), np.asarray([True, False]), _parameters(frames=8, gravity=-9.81), _identity_collision)
    assert float(np.max(np.linalg.norm(result.positions[0] - rest[0]))) <= 1.0e-12


def test_collision_callback_is_called_at_interval() -> None:
    calls = 0

    def callback(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        return positions, np.zeros(len(positions), dtype=bool)

    result = settle_pbd(np.zeros((1, 3)), np.zeros((1, 3)), np.empty((0, 2)), np.empty((0, 2)), np.zeros(1), np.zeros(1, dtype=bool), _parameters(frames=3, collision_interval=1), callback)
    assert calls == 3
    assert result.collision_projection_count == 3


def test_plane_projector_removes_penetration_with_clearance() -> None:
    def plane_projector(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        corrected = positions[:, 2] < 0.0015
        projected = positions.copy()
        projected[corrected, 2] = 0.0015
        return projected, corrected

    positions = np.asarray([[0.0, 0.0, -0.01]])
    result = settle_pbd(positions, positions, np.empty((0, 2)), np.empty((0, 2)), np.zeros(1), np.zeros(1, dtype=bool), _parameters(frames=1, final_projection_iterations=1), plane_projector)
    assert result.positions[0, 2] >= 0.0015


def test_same_input_and_parameters_are_deterministic() -> None:
    rest = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    base = rest + np.asarray([[0.0, 0.0, 0.02], [0.02, 0.0, -0.01], [0.0, -0.01, 0.03]])
    kwargs = (rest, base, np.asarray([[0, 1], [1, 2], [2, 0]]), np.asarray([[0, 2]]), np.asarray([0.5, 0.8, 1.0]), np.asarray([False, False, True]), _parameters(frames=4, gravity=-1.0, final_projection_iterations=2), _identity_collision)
    first = settle_pbd(*kwargs)
    second = settle_pbd(*kwargs)
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.velocity, second.velocity)


def test_small_grid_has_no_nan_or_inf() -> None:
    rest = np.asarray([[x, y, 0.0] for y in range(3) for x in range(3)], dtype=np.float64)
    edges = np.asarray([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [0, 3], [3, 6], [1, 4], [4, 7], [2, 5], [5, 8]])
    result = settle_pbd(rest, rest + np.asarray([0.0, 0.0, 0.02]), edges, np.empty((0, 2)), np.zeros(9), np.zeros(9, dtype=bool), _parameters(frames=3, gravity=-1.0), _identity_collision)
    assert np.isfinite(result.positions).all()
    assert np.isfinite(result.velocity).all()


def test_zero_length_constraint_does_not_create_nan() -> None:
    rest = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = settle_pbd(rest, rest, np.asarray([[0, 1]]), np.asarray([[0, 1]]), np.zeros(2), np.zeros(2, dtype=bool), _parameters(frames=2), _identity_collision)
    assert np.isfinite(result.positions).all()
    assert np.isfinite(result.velocity).all()


def test_nonfinite_input_fails_immediately() -> None:
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        settle_pbd(np.asarray([[np.nan, 0.0, 0.0]]), np.zeros((1, 3)), np.empty((0, 2)), np.empty((0, 2)), np.zeros(1), np.zeros(1, dtype=bool), _parameters(), _identity_collision)
