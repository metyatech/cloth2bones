"""Inspect a ClothTransformer-style NPZ without loading object arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Print keys, shapes, dtypes, and numeric ranges in a cloth NPZ.")
    parser.add_argument("npz", type=Path, help="Input .npz file")
    parser.add_argument("--json", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    with np.load(args.npz, allow_pickle=False) as data:
        fields = []
        for key in data.files:
            value = data[key]
            item = {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
            if np.issubdtype(value.dtype, np.number):
                item["min"] = float(np.nanmin(value))
                item["max"] = float(np.nanmax(value))
            fields.append(item)
    report = {"path": str(args.npz.resolve()), "fields": fields}
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
