"""Generate a tiny legal fixture for testing body-driver extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()
    rest_body = np.asarray([[x, y, z] for x in (-0.3, -0.1, 0.1, 0.3) for y in (0.2, 0.8, 1.4) for z in (-0.1, 0.1)], dtype=np.float32)
    body = np.stack([rest_body + np.asarray([0.0, 0.0, frame * 0.02], dtype=np.float32) for frame in range(args.frames)])
    cloth = np.asarray([[-0.2, 0.8, 0.0], [0.2, 0.8, 0.0], [-0.2, 1.2, 0.0], [0.2, 1.2, 0.0]], dtype=np.float32)
    traj = np.stack([cloth + np.asarray([0.0, 0.0, frame * 0.02], dtype=np.float32) for frame in range(args.frames)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        initial=np.concatenate([cloth, np.zeros_like(cloth)], axis=1),
        traj=traj,
        collision_vertices=body,
        triangles=np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
        collision_triangles=np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
    )
    print(f"Wrote synthetic fixture to {args.out}")


if __name__ == "__main__":
    main()
