"""Deterministic NumPy-only position-based cloth settling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

CollisionProjector = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class PBDSettleParameters:
    frames: int
    dt: float
    solver_iterations: int
    collision_interval: int
    final_projection_iterations: int
    gravity: float
    damping: float
    stretch_stiffness: float
    bend_stiffness: float
    attachment_stiffness: float


@dataclass(frozen=True)
class PBDSettleResult:
    positions: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    frames: int
    collision_projection_count: int


def _finite(value: np.ndarray, label: str) -> None:
    if not np.isfinite(value).all():
        raise FloatingPointError(f"PBD {label} contains NaN or Inf")


def _validate_parameters(parameters: PBDSettleParameters) -> None:
    integer_fields = (
        ("frames", parameters.frames),
        ("solver_iterations", parameters.solver_iterations),
        ("collision_interval", parameters.collision_interval),
        ("final_projection_iterations", parameters.final_projection_iterations),
    )
    for name, value in integer_fields:
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"PBD {name} must be a non-negative integer")
    if parameters.dt <= 0.0 or not np.isfinite(parameters.dt):
        raise ValueError("PBD dt must be finite and positive")
    if parameters.collision_interval == 0 and parameters.frames > 0:
        raise ValueError("PBD collision_interval must be positive when frames are simulated")
    for name in ("damping", "stretch_stiffness", "bend_stiffness", "attachment_stiffness"):
        value = float(getattr(parameters, name))
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"PBD {name} must be finite and in [0, 1]")
    if not np.isfinite(parameters.gravity):
        raise ValueError("PBD gravity must be finite")


def _pairs(value: np.ndarray, vertex_count: int, label: str) -> np.ndarray:
    pairs = np.asarray(value, dtype=np.int64)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"PBD {label} must have shape (N, 2)")
    if np.any(pairs < 0) or np.any(pairs >= vertex_count):
        raise ValueError(f"PBD {label} contains an out-of-range vertex index")
    return pairs


def _pair_lengths(vertices: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if not len(pairs):
        return np.empty(0, dtype=np.float64)
    lengths = np.linalg.norm(vertices[pairs[:, 0]] - vertices[pairs[:, 1]], axis=1)
    _finite(lengths, "constraint lengths")
    return lengths


def _reset_hard_pins(positions: np.ndarray, velocity: np.ndarray, base: np.ndarray, hard_pin: np.ndarray) -> None:
    if np.any(hard_pin):
        positions[hard_pin] = base[hard_pin]
        velocity[hard_pin] = 0.0


def _project_distance_constraints(
    positions: np.ndarray,
    pairs: np.ndarray,
    rest_lengths: np.ndarray,
    inverse_mass: np.ndarray,
    stiffness: float,
) -> None:
    if not len(pairs) or stiffness <= 0.0:
        return
    left = pairs[:, 0]
    right = pairs[:, 1]
    delta = positions[right] - positions[left]
    lengths = np.linalg.norm(delta, axis=1)
    _finite(lengths, "constraint lengths")
    valid = lengths > 1.0e-12
    if not np.any(valid):
        return
    left = left[valid]
    right = right[valid]
    directions = delta[valid] / lengths[valid, None]
    correction = (lengths[valid] - rest_lengths[valid]) * float(stiffness)
    left_weight = inverse_mass[left]
    right_weight = inverse_mass[right]
    total_weight = left_weight + right_weight
    valid_mass = total_weight > 0.0
    if not np.any(valid_mass):
        return
    left = left[valid_mass]
    right = right[valid_mass]
    directions = directions[valid_mass]
    correction = correction[valid_mass]
    left_weight = left_weight[valid_mass]
    right_weight = right_weight[valid_mass]
    total_weight = total_weight[valid_mass]
    accumulator = np.zeros_like(positions)
    contribution = np.zeros(len(positions), dtype=np.float64)
    left_correction = directions * (correction * left_weight / total_weight)[:, None]
    right_correction = -directions * (correction * right_weight / total_weight)[:, None]
    np.add.at(accumulator, left, left_correction)
    np.add.at(accumulator, right, right_correction)
    np.add.at(contribution, left, (left_weight > 0.0).astype(np.float64))
    np.add.at(contribution, right, (right_weight > 0.0).astype(np.float64))
    affected = contribution > 0.0
    positions[affected] += accumulator[affected] / contribution[affected, None]
    _finite(positions, "constraint projection")


def _apply_collision(
    positions: np.ndarray,
    velocity: np.ndarray,
    collision_projector: CollisionProjector,
) -> np.ndarray:
    projected, corrected = collision_projector(positions.copy())
    projected = np.asarray(projected, dtype=np.float64)
    corrected = np.asarray(corrected, dtype=bool)
    if projected.shape != positions.shape:
        raise ValueError("PBD collision projector returned positions with the wrong shape")
    if corrected.shape != (len(positions),):
        raise ValueError("PBD collision projector returned a mask with the wrong shape")
    _finite(projected, "collision projection")
    positions[:] = projected
    velocity[corrected] = 0.0
    return corrected


def settle_pbd(
    rest_vertices: np.ndarray,
    base_vertices: np.ndarray,
    stretch_edges: np.ndarray,
    bend_pairs: np.ndarray,
    attachment_weights: np.ndarray,
    hard_pin_mask: np.ndarray,
    parameters: PBDSettleParameters,
    collision_projector: CollisionProjector,
) -> PBDSettleResult:
    """Settle a cloth mesh from an A-pose baseline using Jacobi PBD projections."""

    _validate_parameters(parameters)
    rest = np.asarray(rest_vertices, dtype=np.float64).copy()
    base = np.asarray(base_vertices, dtype=np.float64).copy()
    if rest.ndim != 2 or rest.shape[1] != 3 or base.shape != rest.shape:
        raise ValueError("PBD rest_vertices and base_vertices must both have shape (N, 3)")
    _finite(rest, "rest vertices")
    _finite(base, "base vertices")
    vertex_count = len(rest)
    edges = _pairs(stretch_edges, vertex_count, "stretch_edges")
    bends = _pairs(bend_pairs, vertex_count, "bend_pairs")
    attachments = np.asarray(attachment_weights, dtype=np.float64)
    hard_pin = np.asarray(hard_pin_mask, dtype=bool)
    if attachments.shape != (vertex_count,) or hard_pin.shape != (vertex_count,):
        raise ValueError("PBD attachment_weights and hard_pin_mask must have shape (N,)")
    _finite(attachments, "attachment weights")
    if np.any(attachments < 0.0) or np.any(attachments > 1.0):
        raise ValueError("PBD attachment weights must be in [0, 1]")
    if not callable(collision_projector):
        raise TypeError("PBD collision_projector must be callable")
    stretch_rest_lengths = _pair_lengths(rest, edges)
    bend_rest_lengths = _pair_lengths(rest, bends)
    inverse_mass = (~hard_pin).astype(np.float64)
    positions = base.copy()
    velocity = np.zeros_like(positions)
    collision_projection_count = 0

    for frame in range(parameters.frames):
        previous = positions.copy()
        velocity[:, 2] += float(parameters.gravity) * float(parameters.dt)
        positions += velocity * float(parameters.dt)
        _reset_hard_pins(positions, velocity, base, hard_pin)
        for _ in range(parameters.solver_iterations):
            _project_distance_constraints(positions, edges, stretch_rest_lengths, inverse_mass, parameters.stretch_stiffness)
            _project_distance_constraints(positions, bends, bend_rest_lengths, inverse_mass, parameters.bend_stiffness)
            _reset_hard_pins(positions, velocity, base, hard_pin)
        positions += float(parameters.attachment_stiffness) * attachments[:, None] * (base - positions)
        _reset_hard_pins(positions, velocity, base, hard_pin)
        corrected = np.zeros(vertex_count, dtype=bool)
        if parameters.collision_interval and ((frame + 1) % parameters.collision_interval == 0 or frame + 1 == parameters.frames):
            corrected = _apply_collision(positions, velocity, collision_projector)
            collision_projection_count += 1
        _reset_hard_pins(positions, velocity, base, hard_pin)
        velocity = float(parameters.damping) * (positions - previous) / float(parameters.dt)
        velocity[corrected] = 0.0
        velocity[hard_pin] = 0.0
        _finite(positions, "frame positions")
        _finite(velocity, "frame velocity")

    for _ in range(parameters.final_projection_iterations):
        _project_distance_constraints(positions, edges, stretch_rest_lengths, inverse_mass, parameters.stretch_stiffness)
        _project_distance_constraints(positions, bends, bend_rest_lengths, inverse_mass, parameters.bend_stiffness)
        positions += float(parameters.attachment_stiffness) * attachments[:, None] * (base - positions)
        _reset_hard_pins(positions, velocity, base, hard_pin)
        corrected = _apply_collision(positions, velocity, collision_projector)
        collision_projection_count += 1
        _reset_hard_pins(positions, velocity, base, hard_pin)
        velocity[corrected] = 0.0
        _finite(positions, "final positions")
        _finite(velocity, "final velocity")

    _reset_hard_pins(positions, velocity, base, hard_pin)
    displacement = positions - base
    _finite(displacement, "displacement")
    return PBDSettleResult(
        positions=positions.copy(),
        displacement=displacement.copy(),
        velocity=velocity.copy(),
        frames=parameters.frames,
        collision_projection_count=collision_projection_count,
    )
