from __future__ import annotations

import numpy as np
import pytest

from cloth2bones.static_equilibrium import (
    EquilibriumParameters,
    adam_minimize,
    collision_energy_gradient,
    edge_energy_gradient,
    gravity_energy_gradient,
    optimize_static_equilibrium,
    smooth_energy_gradient,
    tether_energy_gradient,
)


def _finite_difference(function, value: np.ndarray, index: tuple[int, ...], step: float = 1.0e-6) -> float:
    plus = value.copy()
    minus = value.copy()
    plus[index] += step
    minus[index] -= step
    return float((function(plus) - function(minus)) / (2.0 * step))


def test_edge_energy_gradient_matches_finite_difference() -> None:
    rest = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
    positions = rest + np.asarray([[0.02, -0.01, 0.01], [0.04, 0.01, -0.02], [-0.01, 0.03, 0.02]])
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    energy, gradient = edge_energy_gradient(rest, positions, edges)
    numerical = _finite_difference(lambda candidate: edge_energy_gradient(rest, candidate, edges)[0], positions, (1, 0))
    assert np.isfinite(energy)
    assert gradient[1, 0] == pytest.approx(numerical, rel=1.0e-5, abs=1.0e-7)


def test_smooth_gradient_matches_finite_difference() -> None:
    displacement = np.asarray([[0.02, 0.0, 0.01], [0.04, -0.01, 0.0], [-0.01, 0.03, 0.02]], dtype=np.float64)
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    _, gradient = smooth_energy_gradient(displacement, edges)
    numerical = _finite_difference(lambda candidate: smooth_energy_gradient(candidate, edges)[0], displacement, (1, 1))
    assert gradient[1, 1] == pytest.approx(numerical, rel=1.0e-5, abs=1.0e-7)


def test_tether_gradient_matches_finite_difference() -> None:
    displacement = np.asarray([[0.02, 0.0, 0.01], [0.04, -0.01, 0.0]], dtype=np.float64)
    anchors = np.asarray([0.25, 1.0], dtype=np.float64)
    _, gradient = tether_energy_gradient(displacement, anchors)
    numerical = _finite_difference(lambda candidate: tether_energy_gradient(candidate, anchors)[0], displacement, (0, 2))
    assert gradient[0, 2] == pytest.approx(numerical, rel=1.0e-5, abs=1.0e-7)


def test_collision_gradient_matches_finite_difference() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0002], [0.1, 0.0, 0.002]], dtype=np.float64)
    points = np.zeros_like(positions)
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (2, 1))
    _, gradient = collision_energy_gradient(positions, points, normals, 0.001)
    numerical = _finite_difference(lambda candidate: collision_energy_gradient(candidate, points, normals, 0.001)[0], positions, (0, 2))
    assert gradient[0, 2] == pytest.approx(numerical, rel=1.0e-5, abs=1.0e-7)


def test_gravity_gradient_is_constant_world_z() -> None:
    displacement = np.asarray([[0.02, 0.0, 0.01], [0.04, -0.01, -0.2]], dtype=np.float64)
    energy, gradient = gravity_energy_gradient(displacement)
    assert energy == pytest.approx(-0.095)
    assert np.allclose(gradient, [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])


def test_adam_is_deterministic() -> None:
    initial = np.asarray([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(value * value)), 2.0 * value

    first = adam_minimize(initial, objective, iterations=25, learning_rate=0.01)
    second = adam_minimize(initial, objective, iterations=25, learning_rate=0.01)
    assert np.array_equal(first, second)


def _fixture_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rest = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]], dtype=np.float64)
    base = rest + np.asarray([[0.0, 0.0, -0.015], [0.0, 0.0, -0.015], [0.0, 0.0, -0.015], [0.0, 0.0, -0.015]])
    edges = np.asarray([[0, 1], [1, 3], [3, 2], [2, 0], [0, 3]], dtype=np.int64)
    points = np.zeros_like(rest)
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (len(rest), 1))
    anchors = np.asarray([1.0, 1.0, 0.5, 0.5], dtype=np.float64)
    return rest, base, edges, points, normals, anchors


def test_small_fixture_has_no_nan_and_strong_collision_reduces_penetration() -> None:
    rest, base, edges, points, normals, anchors = _fixture_inputs()
    parameters = EquilibriumParameters(800.0, 2.0, 80.0, 0.1, 5000.0, iterations=80, learning_rate=2.0e-4)
    result = optimize_static_equilibrium(rest, base, edges, points, normals, anchors, parameters)
    initial_penetration = np.maximum(parameters.clearance - base[:, 2], 0.0)
    final_penetration = np.maximum(parameters.clearance - result.positions[:, 2], 0.0)
    assert np.isfinite(result.positions).all()
    assert float(np.mean(final_penetration)) < float(np.mean(initial_penetration))


def test_strong_edge_weight_moves_fixture_toward_rest_length() -> None:
    rest, base, edges, points, normals, anchors = _fixture_inputs()
    stretched = base.copy()
    stretched[1, 0] += 0.03
    parameters = EquilibriumParameters(8000.0, 2.0, 80.0, 0.0, 0.0, iterations=120, learning_rate=2.0e-4)
    result = optimize_static_equilibrium(rest, stretched, edges, points, normals, anchors, parameters)
    initial_error = np.mean(np.abs(np.linalg.norm(stretched[edges[:, 0]] - stretched[edges[:, 1]], axis=1) - np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)))
    final_error = np.mean(np.abs(np.linalg.norm(result.positions[edges[:, 0]] - result.positions[edges[:, 1]], axis=1) - np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)))
    assert final_error < initial_error
