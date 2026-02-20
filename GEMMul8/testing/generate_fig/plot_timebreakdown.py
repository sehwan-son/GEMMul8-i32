#!/usr/bin/env python3
"""Python port of plot_timebreakdown.m for oz2_results_{f,d}_time CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from plot_oz2_common import (
    PRECISION_CONFIG,
    apply_paper_style,
    env_label_from_filename,
    load_csv_rows,
    parse_int,
    parse_float,
    replace_stem_token,
    resolve_input_files,
    sparse_tick_labels,
    unique_stable,
)


STAGE_COLUMNS: List[str] = [
    "conv_64f_2_8i",
    "cublasGemmEx",
    "conv_32i_2_8u",
    "inverse_scaling",
]

STAGE_LABELS: Dict[str, str] = {
    "conv_64f_2_8i": "conv_64f_2_8i",
    "cublasGemmEx": "cublasGemmEx",
    "conv_32i_2_8u": "conv_32i_2_8u",
    "inverse_scaling": "inverse_scaling",
}

STAGE_COLORS: List[str] = ["#F6AA00", "#03AF7A", "#005AFF", "#FF4B00"]


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
        return None
    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stage time-breakdown (%) from oz2_results_{f,d}_time_NVIDIA_*.csv."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Input CSV files. If omitted, auto-loads ../oz2_results_<precision>_time_NVIDIA_*.csv.",
    )
    parser.add_argument("--precision", choices=["d", "f"], default="d")
    parser.add_argument(
        "--glob",
        default="",
        help="Override glob pattern searched in ../ (relative to this script).",
    )
    parser.add_argument("--bar-width", type=float, default=0.82)
    parser.add_argument("--output-dir", default=".", help="Directory to save images.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Display plot windows.")
    return parser.parse_args()


def _find_row(rows: Sequence[Dict[str, str]], n: int, function_name: str) -> Dict[str, str] | None:
    for row in rows:
        n_val = parse_int(row.get("n", ""))
        if n_val != n:
            continue
        if row.get("function", "") == function_name:
            return row
    return None


def _percentages_for_function(
    rows: Sequence[Dict[str, str]],
    *,
    n: int,
    moduli: Sequence[int],
    prefix: str,
) -> List[List[float]]:
    out: List[List[float]] = []
    for m in moduli:
        row = _find_row(rows, n, f"{prefix}-{m}")
        vals: List[float] = []
        for col in STAGE_COLUMNS:
            v = parse_float(row.get(col, "")) if row is not None else None
            vals.append(0.0 if v is None else v)
        s = sum(vals)
        if s > 0.0:
            out.append([100.0 * v / s for v in vals])
        else:
            out.append([0.0 for _ in vals])
    return out


def _plot_stacked(
    ax,
    percentages: Sequence[Sequence[float]],
    x_vals: Sequence[int],
    add_legend: bool,
    bar_width: float,
) -> None:
    bottom = [0.0] * len(x_vals)
    for i, col in enumerate(STAGE_COLUMNS):
        heights = [row[i] for row in percentages]
        label = STAGE_LABELS[col] if add_legend else None
        ax.bar(
            x_vals,
            heights,
            bottom=bottom,
            color=STAGE_COLORS[i],
            width=bar_width,
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
        bottom = [b + h for b, h in zip(bottom, heights)]


def _plot_one_file(plt, path: Path, args: argparse.Namespace) -> Path:
    cfg = PRECISION_CONFIG[args.precision]
    _, rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in CSV: {path}")

    n_list = unique_stable(
        n for row in rows if (n := parse_int(row.get("n", ""))) is not None
    )
    if not n_list:
        raise ValueError(f"Missing valid n column values: {path}")

    moduli = list(range(cfg.xlim_min, cfg.moduli_max + 1))
    x_pos = list(range(1, len(moduli) + 1))
    x_tick_labels = sparse_tick_labels(moduli)

    fig, axes = plt.subplots(2, len(n_list), figsize=(1.95 * len(n_list) + 1.7, 3.4), squeeze=False)

    for col_idx, n in enumerate(n_list):
        ax_fast = axes[0][col_idx]
        fast = _percentages_for_function(rows, n=n, moduli=moduli, prefix="OS2-fast")
        _plot_stacked(ax_fast, fast, x_pos, add_legend=(col_idx == 0), bar_width=args.bar_width)
        ax_fast.set_ylim(0, 100)
        ax_fast.set_yticks([0, 25, 50, 75, 100])
        ax_fast.set_xticks(x_pos)
        ax_fast.set_xticklabels(x_tick_labels)
        ax_fast.tick_params(direction="out", labelsize=8)
        ax_fast.grid(True, axis="y", alpha=0.3)
        ax_fast.set_axisbelow(True)
        ax_fast.spines["top"].set_visible(False)
        ax_fast.spines["right"].set_visible(False)
        ax_fast.set_title(f"n={n}", fontsize=11)
        if col_idx == 0:
            ax_fast.set_ylabel("% (fast)", fontsize=9)
        else:
            ax_fast.set_yticklabels([])

        ax_accu = axes[1][col_idx]
        accu = _percentages_for_function(rows, n=n, moduli=moduli, prefix="OS2-accu")
        _plot_stacked(ax_accu, accu, x_pos, add_legend=False, bar_width=args.bar_width)
        ax_accu.set_ylim(0, 100)
        ax_accu.set_yticks([0, 25, 50, 75, 100])
        ax_accu.set_xticks(x_pos)
        ax_accu.set_xticklabels(x_tick_labels)
        ax_accu.tick_params(direction="out", labelsize=8)
        ax_accu.grid(True, axis="y", alpha=0.3)
        ax_accu.set_axisbelow(True)
        ax_accu.spines["top"].set_visible(False)
        ax_accu.spines["right"].set_visible(False)
        if col_idx == 0:
            ax_accu.set_ylabel("% (accurate)", fontsize=9)
        else:
            ax_accu.set_yticklabels([])

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
        )

    fig.supxlabel("Number of moduli", fontsize=10)
    fig.suptitle(env_label_from_filename(path), fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = replace_stem_token(path.stem, "_time_", "_timebreakdown_", "_timebreakdown")
    out_path = out_dir / f"{out_stem}.png"
    fig.savefig(out_path, dpi=args.dpi)
    if not args.show:
        plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    plt = _import_matplotlib()
    if plt is None:
        return 1
    apply_paper_style(plt)

    script_dir = Path(__file__).resolve().parent
    default_glob = args.glob or f"oz2_results_{args.precision}_time_NVIDIA_*.csv"
    input_files = resolve_input_files(args.files, default_glob, script_dir)
    if not input_files:
        print(f"No input CSV files matched: ../{default_glob}", file=sys.stderr)
        return 1

    failures = 0
    for path in input_files:
        try:
            out = _plot_one_file(plt, path, args)
            print(f"Saved plot: {out}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {path}: {exc}", file=sys.stderr)

    if args.show:
        plt.show()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
