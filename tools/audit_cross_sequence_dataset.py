"""Inventory ClothTransformer-style sequences without committing dataset data.

The audit is intentionally metadata-first: it hashes rest geometry and
connectivity, records every NPZ field, and groups samples by exact cloth
correspondence.  It can also consume a directory containing only the small
rest OBJ files, which is useful before downloading the simulation NPZ files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DATASET_CARD_URL = "https://huggingface.co/datasets/YuCrazing1/ClothTransformer-dataset"


def _digest(values: np.ndarray, *, decimals: int | None = None) -> str:
    array = np.asarray(values)
    if decimals is not None and np.issubdtype(array.dtype, np.floating):
        array = np.round(array.astype(np.float64), decimals)
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype("<i8", copy=False)
    elif np.issubdtype(array.dtype, np.floating):
        array = array.astype("<f8", copy=False)
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _topology_digests(values: np.ndarray) -> dict[str, str]:
    topology = np.asarray(values, dtype=np.int64)
    if topology.ndim != 2:
        raise ValueError(f"Topology must be two-dimensional, got {topology.shape}")
    canonical = np.sort(topology, axis=1)
    canonical = canonical[np.lexsort(canonical.T[::-1])]
    return {"raw": _digest(topology), "canonical_connectivity": _digest(canonical)}


def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f" and len(fields) >= 4:
            indices: list[int] = []
            for token in fields[1:]:
                raw = int(token.split("/", 1)[0])
                indices.append(raw - 1 if raw > 0 else len(vertices) + raw)
            for offset in range(1, len(indices) - 1):
                faces.append([indices[0], indices[offset], indices[offset + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _scalar_metadata(source: np.lib.npyio.NpzFile) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in source.files:
        value = source[key]
        if value.ndim == 0:
            scalar = value.item()
            if isinstance(scalar, (str, int, float, bool)):
                metadata[key] = scalar
        elif value.dtype.kind in {"U", "S"} and value.size <= 16:
            metadata[key] = [str(item) for item in value.reshape(-1).tolist()]
    return metadata


def _field_inventory(source: np.lib.npyio.NpzFile) -> dict[str, dict[str, Any]]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key in source.files
        for value in [source[key]]
    }


def _record(path: Path, obj_path: Path | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sample_id": path.stem,
        "path": str(path.resolve()),
        "metadata": {},
    }
    if path.suffix.lower() == ".obj":
        rest, triangles = _parse_obj(path)
        record["cloth"] = {
            "vertices": int(len(rest)),
            "triangles": int(len(triangles)),
            "rest_geometry_hash": _digest(rest, decimals=8),
            "triangle_hashes": _topology_digests(triangles),
            "obj_only": True,
        }
        return record
    with np.load(path, allow_pickle=False) as source:
        if "traj" not in source or "triangles" not in source or "collision_vertices" not in source:
            raise ValueError(f"{path} is missing a required ClothTransformer field")
        traj = np.asarray(source["traj"], dtype=np.float64)
        initial = np.asarray(source["initial"], dtype=np.float64) if "initial" in source else traj[0]
        cloth_rest = initial[:, :3] if initial.ndim == 2 and initial.shape[1] >= 3 else traj[0]
        cloth_triangles = np.asarray(source["triangles"], dtype=np.int64)
        cloth_edges = np.asarray(source["edges"], dtype=np.int64) if "edges" in source else None
        body = np.asarray(source["collision_vertices"], dtype=np.float64)
        body_triangles = np.asarray(source["collision_triangles"], dtype=np.int64) if "collision_triangles" in source else None
        body_edges = np.asarray(source["collision_edges"], dtype=np.int64) if "collision_edges" in source else None
        record["metadata"] = _scalar_metadata(source)
        record["fields"] = _field_inventory(source)
        record["frames"] = int(traj.shape[0])
        record["cloth"] = {
            "vertices": int(traj.shape[1]),
            "triangles": int(len(cloth_triangles)),
            "topology_hashes": _topology_digests(cloth_triangles),
            "edge_hashes": _topology_digests(cloth_edges) if cloth_edges is not None else None,
            "rest_geometry_hash": _digest(cloth_rest, decimals=8),
            "rest_centroid": cloth_rest.mean(axis=0).tolist(),
            "rest_bounds": np.ptp(cloth_rest, axis=0).tolist(),
            "initial_matches_traj0_max_abs": float(np.max(np.abs(cloth_rest - traj[0]))),
        }
        record["body"] = {
            "vertices": int(body.shape[1]),
            "triangles": int(len(body_triangles)) if body_triangles is not None else None,
            "topology_hashes": _topology_digests(body_triangles) if body_triangles is not None else None,
            "edge_hashes": _topology_digests(body_edges) if body_edges is not None else None,
            "rest_centroid": body[0].mean(axis=0).tolist(),
            "rest_bounds": np.ptp(body[0], axis=0).tolist(),
        }
    return record


def _group(records: list[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for record in records:
        value: Any = record
        for key in key_path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, dict):
            value = value.get("canonical_connectivity")
        if value is None:
            continue
        groups.setdefault(str(value), []).append(str(record["sample_id"]))
    return groups


def audit(root: Path, output: Path | None = None) -> dict[str, Any]:
    paths = sorted(root.glob("*.npz")) + sorted(root.glob("*.obj"))
    if not paths:
        raise FileNotFoundError(f"No .npz or .obj files found under {root}")
    obj_by_sample = {path.name.split("_processed_cloth", 1)[0]: path for path in root.glob("*.obj")}
    records = []
    for path in sorted(root.glob("*.npz")):
        records.append(_record(path, obj_by_sample.get(path.stem)))
    for path in sorted(root.glob("*.obj")):
        if path.stem.split("_processed_cloth", 1)[0] not in {record["sample_id"] for record in records}:
            records.append(_record(path, None))
    inventory = {
        "schema_version": 1,
        "dataset": {
            "name": "ClothTransformer Human Garment",
            "dataset_card": DATASET_CARD_URL,
            "license": "CC BY 4.0 (with third-party asset terms documented by the dataset card)",
            "root": str(root.resolve()),
        },
        "sample_count": len(records),
        "samples": records,
        "groups": {
            "cloth_topology": _group(records, ("cloth", "topology_hashes")),
            "cloth_rest_geometry": _group(records, ("cloth", "rest_geometry_hash")),
            "body_topology": _group(records, ("body", "topology_hashes")),
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cloth sequence topology, rest geometry, and NPZ fields")
    parser.add_argument("--root", type=Path, required=True, help="Directory containing sequence NPZ/OBJ files")
    parser.add_argument("--out", type=Path, required=True, help="JSON inventory output path")
    args = parser.parse_args()
    result = audit(args.root, args.out)
    print(json.dumps({"sample_count": result["sample_count"], "groups": result["groups"]}, indent=2))


if __name__ == "__main__":
    main()
