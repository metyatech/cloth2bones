"""Create compact numeric plots for a PoC 3.2 report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloth2bones.body_motion import apply_skinning, rotation_matrix_to_rotvec  # noqa: E402


def _pose(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["bone_transforms"], dtype=np.float64), np.asarray(data["rest_vertices"], dtype=np.float64), np.asarray(data["weights"], dtype=np.float64)


def _svg(path: Path, title: str, body: str, width: int = 1000, height: int = 480) -> None:
    path.write_text(
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"<rect width='100%' height='100%' fill='#202124'/><text x='30' y='34' fill='#f5f5f5' font-family='sans-serif' font-size='20'>{title}</text>{body}</svg>",
        encoding="utf-8",
    )


def _scale(values: np.ndarray, low: float, high: float, start: float, length: float) -> np.ndarray:
    value_range = max(high - low, 1.0e-12)
    return start + (values - low) / value_range * length


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PoC 3.2 metrics and bone motion")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    aggregate = report["models"]["aggregate"]
    names = list(aggregate)
    values = np.asarray([aggregate[name]["mean_rms"] for name in names], dtype=np.float64)
    plot_left, plot_top, plot_width, plot_height = 70, 70, 880, 330
    x_positions = np.linspace(plot_left + 20, plot_left + plot_width - 20, len(names))
    y_positions = _scale(values, 0.0, float(values.max()), plot_top + plot_height, -plot_height)
    bars = "".join(f"<rect x='{x - 28:.1f}' y='{y:.1f}' width='56' height='{plot_top + plot_height - y:.1f}' fill='#3f7cac'/><text x='{x - 30:.1f}' y='{plot_top + plot_height + 22}' fill='#d7d7d7' font-size='11' transform='rotate(35 {x - 30:.1f},{plot_top + plot_height + 22})'>{name}</text>" for x, y, name in zip(x_positions, y_positions, names, strict=True))
    body = f"<line x1='{plot_left}' y1='{plot_top + plot_height}' x2='{plot_left + plot_width}' y2='{plot_top + plot_height}' stroke='#aaa'/><text x='70' y='455' fill='#ddd' font-size='14'>LOSO mean RMS (lower is better)</text>{bars}"
    _svg(args.out / "loso_model_mean_rms.svg", "Cross-sequence model comparison", body)

    primary = report["primary_test"]
    primary_id = primary["sequence_id"]
    features_path = args.report.parent / "primary_features.npz"
    with np.load(features_path, allow_pickle=False) as data:
        teacher = np.asarray(data["test_cloth_local"], dtype=np.float64)
    rest_pose, rest_vertices, weights = _pose(args.report.parent / f"rest_{primary_id}_poses.npz")
    research_name = primary["research_model"].replace("+", "_")
    runtime_name = primary["runtime_model"].replace("+", "_")
    research_pose, _, _ = _pose(args.report.parent / f"{research_name}_{primary_id}_poses.npz")
    runtime_pose, _, _ = _pose(args.report.parent / f"{runtime_name}_{primary_id}_poses.npz")
    rest_mesh = apply_skinning(rest_vertices, weights, rest_pose)
    research_mesh = apply_skinning(rest_vertices, weights, research_pose)
    runtime_mesh = apply_skinning(rest_vertices, weights, runtime_pose)
    frame_axis = np.arange(1, len(teacher) + 1, dtype=np.float64)

    def frame_rms(values: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(np.sum((values - teacher) ** 2, axis=2), axis=1))

    series = [("Rest baseline", frame_rms(rest_mesh), "#42a35a"), (f"Research: {primary['research_model']}", frame_rms(research_mesh), "#3f7cac"), (f"Runtime: {primary['runtime_model']}", frame_rms(runtime_mesh), "#e07a5f")]
    maximum = max(float(values.max()) for _, values, _ in series)
    lines = []
    for label, values, color in series:
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(_scale(frame_axis, 1.0, float(frame_axis[-1]), 70, 880), _scale(values, 0.0, maximum, 400, -300), strict=True))
        lines.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{points}'/><text x='760' y='{75 + len(lines) * 22}' fill='{color}' font-size='13'>{label}</text>")
    body = "<line x1='70' y1='400' x2='950' y2='400' stroke='#aaa'/><line x1='70' y1='100' x2='70' y2='400' stroke='#aaa'/>" + "".join(lines) + "<text x='70' y='445' fill='#ddd' font-size='14'>Held-out sequence frame</text><text x='70' y='90' fill='#ddd' font-size='14'>RMS point error</text>"
    _svg(args.out / "primary_per_frame_rms.svg", f"Held-out sequence: {primary_id}", body)

    points = []
    distances = []
    errors = []
    for fold in report["models"]["per_fold"]:
        metrics = fold["models"][primary["research_model"]]["metrics"]
        distances.append(metrics["nearest_train_feature_distance_mean"])
        errors.append(metrics["mean_rms"])
    x = _scale(np.asarray(distances), min(distances), max(distances), 90, 820)
    y = _scale(np.asarray(errors), 0.0, max(errors), 390, -280)
    body = "<line x1='70' y1='400' x2='950' y2='400' stroke='#aaa'/><line x1='70' y1='100' x2='70' y2='400' stroke='#aaa'/>" + "".join(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='7' fill='#e07a5f'/><text x='{px + 9:.1f}' y='{py - 8:.1f}' fill='#ddd' font-size='11'>{fold['test_sequence'].replace('seq_', '')}</text>" for fold, px, py in zip(report["models"]["per_fold"], x, y, strict=True)) + "<text x='90' y='445' fill='#ddd' font-size='14'>Mean nearest train feature distance</text><text x='90' y='90' fill='#ddd' font-size='14'>Held-out mean RMS</text>"
    _svg(args.out / "nearest_distance_vs_error.svg", "Feature coverage versus prediction error", body, width=1000, height=480)

    motion_vectors = rotation_matrix_to_rotvec(runtime_pose[:, :, :3, :3])
    motion = np.concatenate([motion_vectors, runtime_pose[:, :, :3, 3]], axis=2)
    motion_std = np.linalg.norm(motion.std(axis=0), axis=1)
    x_positions = np.linspace(70, 950, len(motion_std))
    y_positions = _scale(motion_std, 0.0, float(motion_std.max()), 400, -300)
    bars = "".join(f"<rect x='{x - 6:.1f}' y='{y:.1f}' width='12' height='{400 - y:.1f}' fill='#8e6c8a'/>" for x, y in zip(x_positions, y_positions, strict=True))
    _svg(args.out / "runtime_bone_motion_diversity.svg", "Runtime-friendly per-bone motion diversity", f"<line x1='70' y1='400' x2='950' y2='400' stroke='#aaa'/>{bars}<text x='70' y='445' fill='#ddd' font-size='14'>Cloth helper bone index</text><text x='70' y='90' fill='#ddd' font-size='14'>Transform motion std norm</text>")
    print(f"Wrote PoC 3.2 plots to {args.out.resolve()}")


if __name__ == "__main__":
    main()
