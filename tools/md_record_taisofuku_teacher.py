"""Marvelous Designer 2026 in-app recording script for the prepared project."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ApiTypes
import export_api
import import_api
import utility_api

OUTPUT_ROOT = Path("D:/") / "Users" / "Origin" / "Downloads" / "cloth_poc_out" / "taisofuku_md_teacher"
SOURCE_ZPRJ = Path("D:/") / "VRChatProjects" / "mybooth" / "RyuonTaisofuku" / "01_marvelous_designer" / "RyuonTaisofuku.zprj"
PREPARED_ZPRJ = OUTPUT_ROOT / "RyuonTaisofuku_MD_Prepared.zprj"
RECORDED_ZPRJ = OUTPUT_ROOT / "RyuonTaisofuku_MD_Recorded.zprj"
REPORT_PATH = OUTPUT_ROOT / "md_record_report.json"


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


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not SOURCE_ZPRJ.is_file():
        raise FileNotFoundError(SOURCE_ZPRJ)
    if not PREPARED_ZPRJ.is_file():
        raise FileNotFoundError(PREPARED_ZPRJ)
    original_before = _fingerprint(SOURCE_ZPRJ)
    prepared_before = _fingerprint(PREPARED_ZPRJ)

    _require_members(utility_api, ("NewProject", "GetAvatarCount", "GetAvatarNameList", "SetStartAnimationFrame", "SetEndAnimationFrame", "SetCurrentAnimationFrame", "SetSimulationQuality", "SetSimulationTimeStep", "RunAnimationRecording"), "utility_api")
    _require_members(import_api, ("ImportZprj",), "import_api")
    _require_members(export_api, ("ExportZPrj",), "export_api")
    _require_members(ApiTypes, ("ImportZPRJOption",), "ApiTypes")

    utility_api.NewProject()
    zprj_option = ApiTypes.ImportZPRJOption()
    for name, value in {
        "bAppend": False,
        "bLoadGarment": True,
        "bLoadAvatar": True,
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
    if not import_api.ImportZprj(str(PREPARED_ZPRJ), zprj_option):
        raise RuntimeError("import_api.ImportZprj failed for prepared project")

    avatar_count = int(utility_api.GetAvatarCount())
    avatar_names = list(utility_api.GetAvatarNameList())
    if avatar_count != 1:
        raise RuntimeError(f"Expected exactly one Avatar in prepared project, found {avatar_count}: {avatar_names}")
    utility_api.SetStartAnimationFrame(0)
    utility_api.SetEndAnimationFrame(119)
    utility_api.SetCurrentAnimationFrame(0)
    utility_api.SetSimulationQuality(1, 0)
    utility_api.SetSimulationTimeStep(1.0 / 30.0)
    utility_api.RunAnimationRecording(0, 119)

    exported = export_api.ExportZPrj(str(RECORDED_ZPRJ))
    if not exported or not RECORDED_ZPRJ.is_file():
        raise RuntimeError(f"export_api.ExportZPrj failed: return={exported!r}, expected={RECORDED_ZPRJ}")
    original_after = _fingerprint(SOURCE_ZPRJ)
    report = {
        "prepared_zprj_before": prepared_before,
        "prepared_zprj": str(PREPARED_ZPRJ),
        "original_zprj_before": original_before,
        "original_zprj_after": original_after,
        "original_unchanged": original_before == original_after,
        "avatar_count": avatar_count,
        "avatar_names": avatar_names,
        "animation_start": 0,
        "animation_end": 119,
        "simulation_quality": {"quality": 1, "mode": 0, "description": "Animation (Stable), CPU"},
        "simulation_timestep": 1.0 / 30.0,
        "recording_executed": True,
        "recorded_zprj": str(RECORDED_ZPRJ),
        "recorded_zprj_size": RECORDED_ZPRJ.stat().st_size,
        "alembic_exported": False,
        "cloth2bones_executed": False,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["original_unchanged"]:
        raise RuntimeError("Original ZPRJ changed during recording")
    print(json.dumps({"report": str(REPORT_PATH), "recorded_zprj": str(RECORDED_ZPRJ), "original_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
