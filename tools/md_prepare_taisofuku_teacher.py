"""Marvelous Designer 2026 in-app preparation script; does not simulate or record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ApiTypes
import export_api
import import_api
import pattern_api
import utility_api

OUTPUT_ROOT = Path("D:/") / "Users" / "Origin" / "Downloads" / "cloth_poc_out" / "taisofuku_md_teacher"
SOURCE_ZPRJ = Path("D:/") / "VRChatProjects" / "mybooth" / "RyuonTaisofuku" / "01_marvelous_designer" / "RyuonTaisofuku.zprj"
MOTION_FBX = OUTPUT_ROOT / "motion" / "Ryuon_MD_BothDown.fbx"
PREPARED_ZPRJ = OUTPUT_ROOT / "RyuonTaisofuku_MD_Prepared.zprj"
REPORT_PATH = OUTPUT_ROOT / "md_prepare_report.json"


def _fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"path": str(path), "sha256": digest.hexdigest(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _require_members(module: object, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Required Marvelous Designer 2026 API members missing from {label}: {missing}")


def _json_or_raw(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value, "json_parse_failed": True}


def _pattern_inventory() -> list[dict[str, object]]:
    required = (
        "GetPatternCount",
        "GetPatternPieceName",
        "GetPatternInformation",
        "GetPatternInputInformation",
        "GetParticleDistanceOfPattern",
        "GetPatternPieceFabricIndex",
        "GetPatternLayer",
    )
    _require_members(pattern_api, required, "pattern_api")
    optional_hide = getattr(pattern_api, "GetPatternHide3D", None)
    inventory = []
    for index in range(int(pattern_api.GetPatternCount())):
        record: dict[str, object] = {
            "index": index,
            "pattern_piece_name": pattern_api.GetPatternPieceName(index),
            "pattern_information": _json_or_raw(pattern_api.GetPatternInformation(index)),
            "pattern_input_information": _json_or_raw(pattern_api.GetPatternInputInformation(index)),
            "particle_distance": pattern_api.GetParticleDistanceOfPattern(index),
            "fabric_index": pattern_api.GetPatternPieceFabricIndex(index),
            "layer": pattern_api.GetPatternLayer(index),
            "three_d_hidden": optional_hide(index) if callable(optional_hide) else None,
            "three_d_hidden_api_available": callable(optional_hide),
        }
        inventory.append(record)
    return inventory


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not SOURCE_ZPRJ.is_file():
        raise FileNotFoundError(SOURCE_ZPRJ)
    if not MOTION_FBX.is_file():
        raise FileNotFoundError(MOTION_FBX)
    source_before = _fingerprint(SOURCE_ZPRJ)

    _require_members(utility_api, ("NewProject", "SetStartAnimationFrame", "SetEndAnimationFrame", "SetCurrentAnimationFrame", "SetSimulationQuality", "SetSimulationTimeStep", "GetAvatarCount", "GetAvatarNameList"), "utility_api")
    _require_members(import_api, ("ImportZprj", "ImportFBX"), "import_api")
    _require_members(export_api, ("ExportZPrj",), "export_api")
    _require_members(ApiTypes, ("ImportZPRJOption", "ImportExportOption"), "ApiTypes")

    utility_api.NewProject()
    zprj_option = ApiTypes.ImportZPRJOption()
    for name, value in {
        "bAppend": False,
        "bLoadGarment": True,
        "bLoadAvatar": False,
        "bLoadSceneAndProps": True,
        "bLoadRenderProperties": True,
        "bLoadCustomView": True,
        "translationValueX": 0.0,
        "translationValueY": 0.0,
        "translationValueZ": 0.0,
    }.items():
        if not hasattr(zprj_option, name):
            raise RuntimeError(f"Required ImportZPRJOption property missing: {name}")
        setattr(zprj_option, name, value)
    if not import_api.ImportZprj(str(SOURCE_ZPRJ), zprj_option):
        raise RuntimeError("import_api.ImportZprj failed")

    fbx_option = ApiTypes.ImportExportOption()
    for name, value in {
        "ImportObjectType": 0,
        "bAdd": True,
        "bCreateAnimation": True,
        "bCreateCacheAnimation": False,
        "bMoveGarment": False,
        "bAddArrangementPoints": False,
        "bAutoCreateFittingSuit": False,
        "bAutoTranslate": False,
        "scale": 1.0,
        "translationValueX": 0.0,
        "translationValueY": 0.0,
        "translationValueZ": 0.0,
    }.items():
        if not hasattr(fbx_option, name):
            raise RuntimeError(f"Required ImportExportOption property missing: {name}")
        setattr(fbx_option, name, value)
    if not import_api.ImportFBX(str(MOTION_FBX), fbx_option):
        raise RuntimeError("import_api.ImportFBX failed")

    avatar_count = int(utility_api.GetAvatarCount())
    avatar_names = list(utility_api.GetAvatarNameList())
    if avatar_count != 1:
        raise RuntimeError(f"Expected exactly one imported Avatar, found {avatar_count}: {avatar_names}")
    utility_api.SetStartAnimationFrame(0)
    utility_api.SetEndAnimationFrame(119)
    utility_api.SetCurrentAnimationFrame(0)
    utility_api.SetSimulationQuality(1, 0)
    utility_api.SetSimulationTimeStep(1.0 / 30.0)
    inventory = _pattern_inventory()

    exported = export_api.ExportZPrj(str(PREPARED_ZPRJ))
    if not exported or not PREPARED_ZPRJ.is_file():
        raise RuntimeError(f"export_api.ExportZPrj failed: return={exported!r}, expected={PREPARED_ZPRJ}")
    source_after = _fingerprint(SOURCE_ZPRJ)
    report = {
        "source_zprj_before": source_before,
        "source_zprj_after": source_after,
        "source_unchanged": source_before == source_after,
        "imported_motion_fbx": str(MOTION_FBX),
        "avatar_count": avatar_count,
        "avatar_names": avatar_names,
        "animation_start": 0,
        "animation_end": 119,
        "simulation_quality": {"quality": 1, "mode": 0, "description": "Animation (Stable), CPU"},
        "simulation_timestep": 1.0 / 30.0,
        "pattern_inventory": inventory,
        "prepared_zprj": str(PREPARED_ZPRJ),
        "prepared_zprj_size": PREPARED_ZPRJ.stat().st_size,
        "simulation_executed": False,
        "animation_recording_executed": False,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["source_unchanged"]:
        raise RuntimeError("Original ZPRJ changed during preparation")
    print(json.dumps({"report": str(REPORT_PATH), "prepared_zprj": str(PREPARED_ZPRJ), "source_unchanged": True, "simulation_executed": False}, indent=2))


if __name__ == "__main__":
    main()
