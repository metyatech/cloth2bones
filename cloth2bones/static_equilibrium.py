"""NumPy-only static garment equilibrium optimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray
ObjectiveGradient = Callable[[Array], tuple[float, Array]]


@dataclass(frozen=True)
class EquilibriumParameters:
    """Weights and optimizer settings for one static equilibrium candidate."""

    edge: float
    smooth: float
    tether: float
    gravity: float
    collision: float
    clearance: float = 0.0005
    iterations: int = 1200
    learning_rate: float = 2.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8

    def validate(self) -> None:
        values = (self.edge, self.smooth, self.tether, self.gravity, self.collision, self.clearance, self.learning_rate, self.beta1, self.beta2, self.epsilon)
        if not np.isfinite(values).all():
            raise ValueError("Equilibrium parameters must be finite")
        if any(value < 0.0 for value in (self.edge, self.smooth, self.tether, self.gravity, self.collision, self.clearance)):
            raise ValueError("Energy weights and clearance must be non-negative")
        if self.iterations <= 0 or self.learning_rate <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("iterations, learning_rate, and epsilon must be positive")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("Adam beta values must be in [0, 1)")


@dataclass(frozen=True)
class EquilibriumResult:
    """Optimized positions and unweighted energy terms."""

    positions: Array
    displacement: Array
    total_energy: float
    energies: dict[str, float]


def _vertices(value: Array, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _edges(value: Array, vertex_count: int) -> Array:
    result = np.asarray(value, dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"edges must have shape (E, 2), got {result.shape}")
    if result.size and (result.min() < 0 or result.max() >= vertex_count):
        raise ValueError("edges contain an out-of-range vertex index")
    return result


def edge_energy_gradient(rest_vertices: Array, positions: Array, edges: Array) -> tuple[float, Array]:
    """Return rest-length energy and its analytic position gradient."""

    rest = _vertices(rest_vertices, "rest_vertices")
    current = _vertices(positions, "positions")
    if rest.shape != current.shape:
        raise ValueError("rest_vertices and positions must have the same shape")
    edge_array = _edges(edges, len(rest))
    gradient = np.zeros_like(current)
    if len(edge_array) == 0:
        return 0.0, gradient
    left = edge_array[:, 0]
    right = edge_array[:, 1]
    delta = current[left] - current[right]
    lengths = np.linalg.norm(delta, axis=1)
    rest_lengths = np.linalg.norm(rest[left] - rest[right], axis=1)
    residual = lengths - rest_lengths
    energy = float(np.mean(residual * residual))
    safe_lengths = np.maximum(lengths, 1.0e-12)
    scale = 2.0 * residual / safe_lengths / len(edge_array)
    scale[lengths <= 1.0e-12] = 0.0
    contribution = delta * scale[:, None]
    np.add.at(gradient, left, contribution)
    np.add.at(gradient, right, -contribution)
    return energy, gradient


def smooth_energy_gradient(displacement: Array, edges: Array) -> tuple[float, Array]:
    """Return edge displacement smoothness energy and analytic gradient."""

    correction = _vertices(displacement, "displacement")
    edge_array = _edges(edges, len(correction))
    gradient = np.zeros_like(correction)
    if len(edge_array) == 0:
        return 0.0, gradient
    left = edge_array[:, 0]
    right = edge_array[:, 1]
    delta = correction[left] - correction[right]
    energy = float(np.mean(np.sum(delta * delta, axis=1)))
    contribution = 2.0 * delta / len(edge_array)
    np.add.at(gradient, left, contribution)
    np.add.at(gradient, right, -contribution)
    return energy, gradient


def tether_energy_gradient(displacement: Array, anchor_weights: Array) -> tuple[float, Array]:
    """Return anchor tether energy and analytic gradient."""

    correction = _vertices(displacement, "displacement")
    anchors = np.asarray(anchor_weights, dtype=np.float64)
    if anchors.shape != (len(correction),):
        raise ValueError(f"anchor_weights must have shape ({len(correction)},), got {anchors.shape}")
    if not np.isfinite(anchors).all() or (anchors < 0.0).any():
        raise ValueError("anchor_weights must be finite and non-negative")
    energy = float(np.mean(anchors * np.sum(correction * correction, axis=1)))
    gradient = 2.0 * anchors[:, None] * correction / len(correction)
    return energy, gradient


def gravity_energy_gradient(displacement: Array) -> tuple[float, Array]:
    """Return the world-Z gravity energy and constant analytic gradient."""

    correction = _vertices(displacement, "displacement")
    energy = float(np.mean(correction[:, 2]))
    gradient = np.zeros_like(correction)
    gradient[:, 2] = 1.0 / len(correction)
    return energy, gradient


def collision_energy_gradient(
    positions: Array,
    collision_points: Array,
    collision_normals: Array,
    clearance: float,
) -> tuple[float, Array]:
    """Return fixed-point surface collision energy and analytic gradient."""

    current = _vertices(positions, "positions")
    points = _vertices(collision_points, "collision_points")
    normals = _vertices(collision_normals, "collision_normals")
    if points.shape != current.shape or normals.shape != current.shape:
        raise ValueError("collision arrays must have the same shape as positions")
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance must be finite and non-negative")
    if (np.linalg.norm(normals, axis=1) <= 1.0e-12).any():
        raise ValueError("collision_normals must be non-zero")
    signed = np.sum((current - points) * normals, axis=1)
    penetration = np.maximum(clearance - signed, 0.0)
    energy = float(np.mean(penetration * penetration))
    gradient = -2.0 * penetration[:, None] * normals / len(current)
    return energy, gradient


def energy_and_gradient(
    rest_vertices: Array,
    base_vertices: Array,
    edges: Array,
    collision_points: Array,
    collision_normals: Array,
    anchor_weights: Array,
    displacement: Array,
    parameters: EquilibriumParameters,
) -> tuple[float, dict[str, float], Array]:
    """Evaluate the weighted energy and analytic gradient at a displacement."""

    parameters.validate()
    rest = _vertices(rest_vertices, "rest_vertices")
    base = _vertices(base_vertices, "base_vertices")
    correction = _vertices(displacement, "displacement")
    if rest.shape != base.shape or rest.shape != correction.shape:
        raise ValueError("rest_vertices, base_vertices, and displacement must have the same shape")
    position = base + correction
    edge_energy, edge_gradient = edge_energy_gradient(rest, position, edges)
    smooth_energy, smooth_gradient = smooth_energy_gradient(correction, edges)
    tether_energy, tether_gradient = tether_energy_gradient(correction, anchor_weights)
    gravity_energy, gravity_gradient = gravity_energy_gradient(correction)
    collision_energy, collision_gradient = collision_energy_gradient(position, collision_points, collision_normals, parameters.clearance)
    energies = {
        "edge": edge_energy,
        "smooth": smooth_energy,
        "tether": tether_energy,
        "gravity": gravity_energy,
        "collision": collision_energy,
    }
    total = (
        parameters.edge * edge_energy
        + parameters.smooth * smooth_energy
        + parameters.tether * tether_energy
        + parameters.gravity * gravity_energy
        + parameters.collision * collision_energy
    )
    gradient = (
        parameters.edge * edge_gradient
        + parameters.smooth * smooth_gradient
        + parameters.tether * tether_gradient
        + parameters.gravity * gravity_gradient
        + parameters.collision * collision_gradient
    )
    if not np.isfinite(total) or not np.isfinite(gradient).all():
        raise FloatingPointError("Non-finite static equilibrium energy or gradient")
    return float(total), energies, gradient


def adam_minimize(
    initial: Array,
    objective_gradient: ObjectiveGradient,
    iterations: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
) -> Array:
    """Minimize a finite objective with deterministic, in-place Adam steps."""

    position = np.asarray(initial, dtype=np.float64).copy()
    if not np.isfinite(position).all():
        raise ValueError("initial must contain only finite values")
    if iterations <= 0 or learning_rate <= 0.0 or epsilon <= 0.0:
        raise ValueError("iterations, learning_rate, and epsilon must be positive")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("Adam beta values must be in [0, 1)")
    first_moment = np.zeros_like(position)
    second_moment = np.zeros_like(position)
    for step in range(1, iterations + 1):
        _, gradient = objective_gradient(position)
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.shape != position.shape or not np.isfinite(gradient).all():
            raise FloatingPointError(f"Non-finite or malformed gradient at Adam iteration {step}")
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
        bias_first = 1.0 - beta1**step
        bias_second = 1.0 - beta2**step
        direction = (first_moment / bias_first) / (np.sqrt(second_moment / bias_second) + epsilon)
        position -= learning_rate * direction
        if not np.isfinite(position).all():
            raise FloatingPointError(f"Non-finite position at Adam iteration {step}")
    return position


def optimize_static_equilibrium(
    rest_vertices: Array,
    base_vertices: Array,
    edges: Array,
    collision_points: Array,
    collision_normals: Array,
    anchor_weights: Array,
    parameters: EquilibriumParameters,
) -> EquilibriumResult:
    """Optimize displacement from ``base_vertices`` under the five energies."""

    parameters.validate()
    rest = _vertices(rest_vertices, "rest_vertices")
    base = _vertices(base_vertices, "base_vertices")
    if rest.shape != base.shape:
        raise ValueError("rest_vertices and base_vertices must have the same shape")
    initial = np.zeros_like(base)

    def objective_gradient(correction: Array) -> tuple[float, Array]:
        total, _, gradient = energy_and_gradient(
            rest,
            base,
            edges,
            collision_points,
            collision_normals,
            anchor_weights,
            correction,
            parameters,
        )
        return total, gradient

    displacement = adam_minimize(
        initial,
        objective_gradient,
        parameters.iterations,
        parameters.learning_rate,
        parameters.beta1,
        parameters.beta2,
        parameters.epsilon,
    )
    total, energies, _ = energy_and_gradient(
        rest,
        base,
        edges,
        collision_points,
        collision_normals,
        anchor_weights,
        displacement,
        parameters,
    )
    positions = base + displacement
    if not np.isfinite(positions).all():
        raise FloatingPointError("Non-finite optimized positions")
    return EquilibriumResult(positions, displacement, total, energies)
