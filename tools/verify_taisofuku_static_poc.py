"""Verify a static-equilibrium review Blend in a separate Blender process."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy

SOURCE_SHA256 = "00eace7d79ed201dd0c5684c3301f9af6c5653bff05c215e4d8ed5c6433801df"
EXPECTED_VERTICES = 121471
EXPECTED_FACES = 115103
COLLECTIONS = ("RYUON_A", "A_LBS_BASELINE", "A_OPT_STIFF", "A_OPT_BALANCED", "A_OPT_SOFT")
CANDIDATE_COLLECTIONS = ("A_LBS_BASELINE", "A_OPT_STIFF", "A_OPT_BALANCED", "A_OPT_SOFT")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_mesh(obj: bpy.types.Object) -> bool:
    return all(math.isfinite(float(value)) for vertex in obj.data.vertices for value in vertex.co)


def main() -> None:
    args = _args()
    blend = args.blend.resolve()
    report_path = args.report.resolve()
    source = args.source.resolve()
    if not blend.is_file() or not report_path.is_file() or not source.is_file():
        raise FileNotFoundError(f"Missing verification input: blend={blend}, report={report_path}, source={source}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    missing_collections = [name for name in COLLECTIONS if bpy.data.collections.get(name) is None]
    if missing_collections:
        raise RuntimeError(f"Review Blend is missing collections: {missing_collections}")
    if report.get("review_blend") and Path(report["review_blend"]).resolve() != blend:
        raise RuntimeError("Report review_blend path does not match the verified Blend")
    if report.get("collections") != list(COLLECTIONS):
        raise RuntimeError("Report collection list is inconsistent")
    if report.get("previews") is None or len(report["previews"]) != 16:
        raise RuntimeError("Report must list exactly 16 preview PNGs")
    missing_previews = [path for path in report["previews"] if not Path(path).is_file()]
    if missing_previews:
        raise RuntimeError(f"Missing preview PNGs: {missing_previews}")
    candidate_records = report.get("candidates", {})
    for collection_name in CANDIDATE_COLLECTIONS:
        collection = bpy.data.collections[collection_name]
        mesh_objects = [obj for obj in collection.objects if obj.type == "MESH"]
        if len(mesh_objects) != 1:
            raise RuntimeError(f"Collection {collection_name} must contain one mesh, got {len(mesh_objects)}")
        candidate = mesh_objects[0]
        if len(candidate.data.vertices) != EXPECTED_VERTICES or len(candidate.data.polygons) != EXPECTED_FACES:
            raise RuntimeError(f"{candidate.name} topology mismatch")
        if candidate.modifiers:
            raise RuntimeError(f"{candidate.name} still has modifiers")
        if not _finite_mesh(candidate):
            raise RuntimeError(f"{candidate.name} contains NaN or Inf vertex coordinates")
        report_name = "balanced" if collection_name == "A_OPT_BALANCED" else "baseline" if collection_name == "A_LBS_BASELINE" else collection_name.removeprefix("A_OPT_").lower()
        record = candidate_records.get(report_name)
        if not record or not record.get("topology_preserved") or not record.get("valid_numeric"):
            raise RuntimeError(f"Report does not validate candidate {report_name}")
        if record.get("output_vertices") != EXPECTED_VERTICES or record.get("output_faces") != EXPECTED_FACES:
            raise RuntimeError(f"Report topology counts are inconsistent for {report_name}")

    expected_visibility = {"RYUON_A": False, "A_LBS_BASELINE": True, "A_OPT_STIFF": True, "A_OPT_BALANCED": False, "A_OPT_SOFT": True}
    for name, hidden in expected_visibility.items():
        if bpy.data.collections[name].hide_viewport != hidden or bpy.data.collections[name].hide_render != hidden:
            raise RuntimeError(f"Collection visibility mismatch for {name}")
    scene = bpy.context.scene
    if scene.get("dem_bones_executed") is not False or scene.get("weight_generation_executed") is not False:
        raise RuntimeError("Review Blend indicates a forbidden downstream operation ran")

    source_stat = source.stat()
    source_hash = _sha256(source)
    source_record = report.get("source", {})
    source_unchanged = (
        source_hash == SOURCE_SHA256
        and source_record.get("sha256") == SOURCE_SHA256
        and source_record.get("sha256_after") == SOURCE_SHA256
        and source_record.get("size") == source_stat.st_size
        and source_record.get("size_after") == source_stat.st_size
        and source_record.get("mtime_ns") == source_stat.st_mtime_ns
        and source_record.get("mtime_ns_after") == source_stat.st_mtime_ns
        and source_record.get("unchanged") is True
    )
    if not source_unchanged:
        raise RuntimeError("Source Blend hash, size, or mtime is not unchanged")
    if report.get("original_total_vertex_count") != EXPECTED_VERTICES or report.get("original_total_face_count") != EXPECTED_FACES:
        raise RuntimeError("Report original topology counts are inconsistent")
    result = {
        "valid": True,
        "blend": str(blend),
        "report": str(report_path),
        "collections": list(COLLECTIONS),
        "candidate_topology": {name: {"vertices": EXPECTED_VERTICES, "faces": EXPECTED_FACES} for name in CANDIDATE_COLLECTIONS},
        "nan_inf_free": True,
        "report_consistent": True,
        "source_unchanged": True,
        "preview_count": 16,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
