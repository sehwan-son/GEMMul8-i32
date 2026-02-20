#!/usr/bin/env python3
"""Python port of plot_accuracy.m for oz2_results_{f,d}_accuracy CSV files."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from plot_oz2_common import (
    PRECISION_CONFIG,
    apply_paper_style,
    env_label_from_filename,
    legend_ncol,
    load_csv_rows,
    parse_float,
    replace_stem_token,
    robust_log_bounds,
    resolve_input_files,
    sparse_tick_labels,
    split_function_and_k,
    unique_stable,
    wrapped_mark,
)


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
        return None
    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot max relative error from oz2_results_{f,d}_accuracy_NVIDIA_*.csv."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Input CSV files. If omitted, auto-loads ../oz2_results_<precision>_accuracy_NVIDIA_*.csv.",
    )
    parser.add_argument("--precision", choices=["d", "f"], default="d")
    parser.add_argument(
        "--glob",
        default="",
        help="Override glob pattern searched in ../ (relative to this script).",
    )
    parser.add_argument(
        "--k-values",
        default="",
        help="Comma-separated k values to plot. Default: first and last available k.",
    )
    parser.add_argument("--ylim-mode", choices=["robust", "full"], default="robust")
    parser.add_argument("--q-low", type=float, default=0.02, help="Lower quantile for robust log y-limit.")
    parser.add_argument("--q-high", type=float, default=0.98, help="Upper quantile for robust log y-limit.")
    parser.add_argument("--output-dir", default=".", help="Directory to save images.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Display plot windows.")
    return parser.parse_args()


def _preferred_base_order(precision: str) -> List[str]:
    if precision == "d":
        return [
            "DGEMM",
            "ozIMMU_EF-8",
            "ozIMMU_EF-9",
            "OS1-7",
            "OS1-8",
            "OS1-9",
            "OS1-10",
            "OS2-fast",
            "OS2-accu",
        ]
    return [
        "SGEMM",
        "SGEMM-TF32",
        "SGEMM-BF16X9",
        "FP16TCEC_SCALING",
        "OS2-fast",
        "OS2-accu",
    ]


def _build_ordered_base_names(base_names: Sequence[str], precision: str) -> List[str]:
    base_set = set(base_names)
    out: List[str] = []
    for name in _preferred_base_order(precision):
        if name in base_set:
            out.append(name)
    for name in base_names:
        if name not in out:
            out.append(name)
    return out


def _parse_k_values_arg(arg: str) -> List[int]:
    if not arg.strip():
        return []
    out: List[int] = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            raise ValueError(f"Invalid --k-values token: {token}") from None
    return out


def _select_k_values(available: Sequence[int], requested: Sequence[int]) -> List[int]:
    if not available:
        return []
    available_sorted = sorted(set(available))
    if requested:
        req = [k for k in requested if k in available_sorted]
        return req if req else [available_sorted[0]]
    if len(available_sorted) == 1:
        return [available_sorted[0]]
    return [available_sorted[0], available_sorted[-1]]


def _plot_one_file(plt, path: Path, args: argparse.Namespace) -> Path:
    cfg = PRECISION_CONFIG[args.precision]
    header, rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in CSV: {path}")

    moduli = [int(h) for h in header if h.isdigit()]
    if not moduli:
        raise ValueError(f"No numeric moduli columns in CSV: {path}")

    # (phi, base_name, k) -> y-values
    rec: Dict[Tuple[float, str, int], List[float]] = {}
    phi_list = unique_stable(
        phi for row in rows if (phi := parse_float(row.get("phi", ""))) is not None
    )
    base_names_seen: List[str] = []
    k_seen: List[int] = []

    for row in rows:
        phi = parse_float(row.get("phi", ""))
        if phi is None:
            continue
        function_name = row.get("function", "")
        base, k = split_function_and_k(function_name)
        if k is None:
            continue
        y_vals = [parse_float(row.get(str(m), "")) for m in moduli]
        rec[(phi, base, k)] = [math.nan if v is None else v for v in y_vals]
        base_names_seen.append(base)
        k_seen.append(k)

    if not rec:
        raise ValueError(f"No (phi, function(k), values) rows found: {path}")

    base_names = _build_ordered_base_names(unique_stable(base_names_seen), args.precision)
    requested_k = _parse_k_values_arg(args.k_values)
    k_values = _select_k_values(k_seen, requested_k)
    if not k_values:
        raise ValueError(f"No usable k values in CSV: {path}")

    import matplotlib.ticker as mticker

    ncols = len(phi_list)
    fig, axes = plt.subplots(1, ncols, figsize=(1.95 * ncols + 1.7, 3.3), squeeze=False)
    axes_1d = axes[0]
    legend_handles: Dict[str, object] = {}

    for p_idx, phi in enumerate(phi_list):
        ax = axes_1d[p_idx]
        ax.grid(True, axis="both", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        subplot_vals: List[float] = []

        for k_idx, k in enumerate(k_values, start=1):
            for base_idx, base in enumerate(base_names, start=1):
                key = (phi, base, k)
                if key not in rec:
                    continue
                y = rec[key]
                # Log-scale is undefined for <=0, so skip those points.
                x_plot: List[int] = []
                y_plot: List[float] = []
                for x_val, y_val in zip(moduli, y):
                    if math.isnan(y_val) or y_val <= 0.0:
                        continue
                    x_plot.append(x_val)
                    y_plot.append(y_val)
                if not x_plot:
                    continue
                subplot_vals.extend(y_plot)
                label = base if k == k_values[0] else None
                (line,) = ax.plot(
                    x_plot,
                    y_plot,
                    wrapped_mark(k_idx, base_idx),
                    linewidth=1.0,
                    label=label,
                )
                if label and label not in legend_handles:
                    legend_handles[label] = line

        ax.set_yscale("log")
        y_bounds = robust_log_bounds(
            subplot_vals,
            mode=args.ylim_mode,
            q_low=args.q_low,
            q_high=args.q_high,
        )
        if y_bounds is None:
            y_min, y_max = cfg.accuracy_ylim
        else:
            y_min, y_max = y_bounds
            y_min = max(cfg.accuracy_ylim[0], y_min)
            y_max = min(cfg.accuracy_ylim[1], max(y_max, y_min * 1.2))
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=8))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(2, 5), numticks=16))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlim(cfg.xlim_min, cfg.moduli_max)
        ticks = list(range(cfg.xlim_min, cfg.moduli_max + 1))
        ax.set_xticks(ticks)
        ax.set_xticklabels(sparse_tick_labels(ticks))
        ax.tick_params(direction="out", labelsize=8)
        ax.set_title(f"phi={phi:g}", fontsize=11)
        if p_idx > 0:
            ax.set_yticklabels([])

    if legend_handles:
        legend_labels = list(legend_handles.keys())
        ncol = legend_ncol(len(legend_labels), max_cols=4)
        fig.legend(
            list(legend_handles.values()),
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
            ncol=ncol,
        )

    fig.suptitle(env_label_from_filename(path), fontsize=11)

    fig.supxlabel("Number of moduli", fontsize=10)
    fig.supylabel("max relative error", fontsize=10)
    fig.text(0.5, 0.95, f"line-style by k: {', '.join(str(k) for k in k_values)}", ha="center", va="top", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = replace_stem_token(path.stem, "_accuracy_", "_accuracy_plot_", "_accuracy_plot")
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
    default_glob = args.glob or f"oz2_results_{args.precision}_accuracy_NVIDIA_*.csv"
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
