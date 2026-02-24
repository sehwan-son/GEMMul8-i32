#!/usr/bin/env python3
"""Plot i32 stage breakdown in oz2-style stacked bars."""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Sequence

from plot_i32_common import metric_value, read_filtered_rows


# Keep the same visual tone as workspace/ozaki_estimated_vs_actual.py
BAR_PAL4 = ["#1a6fb5", "#d48a00", "#2a9d8f", "#cc4f4f"]

STAGE_KEYS = [
    ("gemmul8_encode_ms", "Quantization", BAR_PAL4[0]),
    ("gemmul8_tc_ms", "Low-prec GEMM", BAR_PAL4[1]),
    ("gemmul8_conv32to8_ms", "Requantization", BAR_PAL4[2]),
    ("gemmul8_reconstruct_ms", "Dequantization", BAR_PAL4[3]),
]
LEGEND_HIDDEN_STAGE_KEYS = {"gemmul8_conv32to8_ms"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot i32 stage-wise time breakdown (oz2-style).")
    parser.add_argument("csv", help="Path to i32_bench_speedup_*.csv")
    parser.add_argument("--op", default="all", choices=["all", "NN", "NT", "TN", "TT"])
    parser.add_argument("--i32-scheme", default="all", choices=["all", "oz2", "oz1"])
    parser.add_argument("--use-extra", default="true", choices=["all", "true", "false"])
    parser.add_argument("--num-moduli", default="all")
    parser.add_argument("--x-axis", default="n", choices=["n", "m", "k", "mnk"])
    parser.add_argument(
        "--layout",
        default="n_sweep",
        choices=["n_sweep", "moduli_sweep", "compare"],
        help="n_sweep: fixed moduli and compare x-axis change (recommended), "
             "moduli_sweep: previous oz2-like view, "
             "compare: side-by-side extra vs compact bars per n.",
    )
    parser.add_argument(
        "--target-moduli",
        type=int,
        default=9,
        help="Used in n_sweep when --num-moduli=all. Defaults to 9.",
    )
    parser.add_argument(
        "--mode",
        default="percent",
        choices=["percent", "ms"],
        help="percent: normalized stacked ratio (fp64 figure style), ms: stacked absolute ms.",
    )
    parser.add_argument(
        "--ms-ylim-mode",
        default="robust",
        choices=["robust", "full"],
        help="Y-limit mode for --mode ms.",
    )
    parser.add_argument(
        "--ms-q-high",
        type=float,
        default=0.95,
        help="Upper quantile for robust ms y-limit (used when --ms-ylim-mode=robust).",
    )
    parser.add_argument("--bar-width", type=float, default=0.46)
    parser.add_argument(
        "--label-min-pct",
        type=float,
        default=8.0,
        help="In percent mode, show in-bar labels only for segments >= this value (percent).",
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=7.0,
        help="Font size for in-bar percent labels.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
        return None
    return plt


def _apply_paper_style(plt) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#666666",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "figure.dpi": 150,
    })


def _clean_ax(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#666666")
    ax.spines["left"].set_color("#666666")
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.5, alpha=0.7, color="#cccccc")


def _alternating_labels(values: Sequence[int]) -> List[str]:
    if len(values) <= 6:
        return [str(v) for v in values]
    out: List[str] = []
    for i, v in enumerate(values):
        out.append(str(v) if i % 2 == 0 else "")
    return out


def _sparse_labels(values: Sequence[int]) -> List[str]:
    if len(values) <= 10:
        return [str(v) for v in values]
    return _alternating_labels(values)


def _moduli_labels_fp64_style(values: Sequence[int]) -> List[str]:
    if len(values) <= 7:
        return [str(v) for v in values]
    return [str(v) if (i == 0 or i % 3 == 0) else "" for i, v in enumerate(values)]


def _x_title_name(x_axis: str) -> str:
    if x_axis == "m":
        return "m"
    if x_axis == "k":
        return "k"
    if x_axis == "mnk":
        return "m*n*k"
    return "n"


def _get_row(
    rows: Sequence[dict],
    *,
    x_value: int,
    use_extra: str,
    num_moduli: int,
) -> dict | None:
    for row in rows:
        if int(row["_x"]) != x_value:
            continue
        if str(row["_use_extra"]) != use_extra:
            continue
        if int(row["_mod"]) != num_moduli:
            continue
        return row
    return None


def _stage_values(row: dict | None) -> List[float]:
    if row is None:
        return [0.0] * len(STAGE_KEYS)
    vals: List[float] = []
    for key, _, _ in STAGE_KEYS:
        v = metric_value(row, key)
        vals.append(0.0 if v is None else max(0.0, v))
    return vals


def _active_stage_indices(rows: Sequence[dict]) -> List[int]:
    active: List[int] = []
    for si, (key, _, _) in enumerate(STAGE_KEYS):
        found = False
        for row in rows:
            v = metric_value(row, key)
            if v is not None and v > 0.0:
                found = True
                break
        if found:
            active.append(si)
    if not active:
        return list(range(len(STAGE_KEYS)))
    return active


def _to_percent(values: Sequence[float]) -> List[float]:
    total = sum(values)
    if total <= 0.0:
        return [0.0] * len(values)
    return [v * 100.0 / total for v in values]


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("empty data")
    if q <= 0.0:
        return sorted_vals[0]
    if q >= 1.0:
        return sorted_vals[-1]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _stacked_ms_ylim(stacks: Sequence[Sequence[float]], mode: str, q_high: float) -> float | None:
    totals = sorted(sum(s) for s in stacks if sum(s) > 0.0)
    if not totals:
        return None
    if mode == "full":
        ymax = totals[-1]
    else:
        q = min(max(q_high, 0.0), 1.0)
        ymax = _quantile(totals, q)
        ymax = max(ymax, totals[min(len(totals) - 1, max(0, len(totals) // 2))])
    return ymax * 1.08


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    c = hex_color.strip().lstrip("#")
    if len(c) != 6:
        return (0.2, 0.2, 0.2)
    try:
        r = int(c[0:2], 16) / 255.0
        g = int(c[2:4], 16) / 255.0
        b = int(c[4:6], 16) / 255.0
    except ValueError:
        return (0.2, 0.2, 0.2)
    return (r, g, b)


def _label_text_color_for_fill(hex_color: str) -> str:
    r, g, b = _hex_to_rgb01(hex_color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.55 else "#1D1D1D"


def _annotate_stack_percent_labels(
    ax,
    x_pos: Sequence[float],
    stacks: Sequence[Sequence[float]],
    active_stage_indices: Sequence[int],
    min_pct: float,
    fontsize: float,
) -> None:
    for i, x in enumerate(x_pos):
        if i >= len(stacks):
            continue
        stack = stacks[i]
        bottom = 0.0
        for si in active_stage_indices:
            if si >= len(stack):
                continue
            h = float(stack[si])
            if h < min_pct:
                bottom += h
                continue
            _, _, color = STAGE_KEYS[si]
            y = bottom + 0.5 * h
            ax.text(
                x,
                y,
                f"{h:.0f}%",
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=_label_text_color_for_fill(color),
                zorder=7,
            )
            bottom += h


def _add_top_legend(fig, handles, labels, *, fontsize: int = 8) -> bool:
    if not handles:
        return False
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, len(handles)),
        bbox_to_anchor=(0.5, 0.942),
        fontsize=fontsize,
        frameon=True,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    return True


def main() -> int:
    args = parse_args()
    plt = _import_matplotlib()
    if plt is None:
        return 1
    _apply_paper_style(plt)

    try:
        rows = read_filtered_rows(
            args.csv,
            args.op,
            args.use_extra,
            args.num_moduli,
            i32_scheme=args.i32_scheme,
            x_axis=args.x_axis,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not rows:
        print("No rows matched filters.", file=sys.stderr)
        return 1

    moduli_values = sorted({int(r["_mod"]) for r in rows})
    use_extra_values = sorted({str(r["_use_extra"]) for r in rows}, reverse=True)  # true row first

    if args.use_extra != "all":
        use_extra_values = [args.use_extra]

    if args.layout == "n_sweep":
        if args.num_moduli != "all":
            target_moduli = int(args.num_moduli)
        else:
            if args.target_moduli in moduli_values:
                target_moduli = args.target_moduli
            else:
                target_moduli = moduli_values[0]
                if len(moduli_values) > 1:
                    print(
                        f"[WARN] target moduli {args.target_moduli} not found; using {target_moduli}.",
                        file=sys.stderr,
                    )

        rows = [r for r in rows if int(r["_mod"]) == target_moduli]
        x_values = sorted({int(r["_x"]) for r in rows})
        if not x_values:
            print("No rows left after moduli filtering.", file=sys.stderr)
            return 1
        active_stage_indices = _active_stage_indices(rows)

        nrows = len(use_extra_values)
        fig, axes = plt.subplots(
            nrows,
            1,
            figsize=(0.95 * max(3, len(x_values)) + 0.15, 2.30 * nrows + 1.25),
            squeeze=False,
        )

        x_stride = 0.85
        x_pos = [1.0 + i * x_stride for i in range(len(x_values))]
        xticklabels = _sparse_labels(x_values)

        for row_idx, ue in enumerate(use_extra_values):
            ax = axes[row_idx][0]
            stacks: List[List[float]] = []
            for x_val in x_values:
                r = _get_row(rows, x_value=x_val, use_extra=ue, num_moduli=target_moduli)
                vals = _stage_values(r)
                if args.mode == "percent":
                    vals = _to_percent(vals)
                stacks.append(vals)

            bottom = [0.0] * len(x_values)
            for si in active_stage_indices:
                stage_key, label, color = STAGE_KEYS[si]
                heights = [s[si] for s in stacks]
                legend_label = None if stage_key in LEGEND_HIDDEN_STAGE_KEYS else label
                ax.bar(
                    x_pos,
                    heights,
                    bottom=bottom,
                    color=color,
                    width=args.bar_width,
                    edgecolor="white",
                    linewidth=0.4,
                    label=legend_label if row_idx == 0 else None,
                )
                bottom = [b + h for b, h in zip(bottom, heights)]

            if args.mode == "percent":
                _annotate_stack_percent_labels(
                    ax,
                    x_pos,
                    stacks,
                    active_stage_indices,
                    min_pct=max(0.0, float(args.label_min_pct)),
                    fontsize=max(4.0, float(args.label_fontsize)),
                )

            _clean_ax(ax, "y")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(xticklabels, fontsize=8)
            ax.tick_params(direction="out", labelsize=8)
            ax.margins(x=0.0)
            if args.mode == "percent":
                ax.set_ylim(0, 100)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.set_yticklabels(["0", "", "50", "", "100"])
                ax.set_ylabel("Time share (%)", fontsize=10, labelpad=4)
            else:
                y_max = _stacked_ms_ylim(stacks, args.ms_ylim_mode, args.ms_q_high)
                if y_max is not None:
                    ax.set_ylim(0, y_max)
                ax.set_ylabel("Time (ms)", fontsize=10, labelpad=4)

        handles, labels = axes[0][0].get_legend_handles_labels()
        has_legend = _add_top_legend(fig, handles, labels, fontsize=8)

        axes[-1][0].set_xlabel(_x_title_name(args.x_axis), fontsize=10, labelpad=10)
        fig.suptitle(f"i32 Stage Breakdown (moduli={target_moduli})", fontsize=12, fontweight="bold", y=0.988)
        top = 0.925 if has_legend else 0.955
        fig.tight_layout(rect=(0.0, 0.06, 1.0, top), h_pad=0.30, w_pad=0.20)
    elif args.layout == "moduli_sweep":
        x_values = sorted({int(r["_x"]) for r in rows})
        active_stage_indices = _active_stage_indices(rows)
        nrows = len(use_extra_values)
        ncols = len(x_values)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(1.05 * ncols + 0.20, 2.15 * nrows + 0.20),
            squeeze=False,
        )

        x_stride = 0.85
        x_pos = [1.0 + i * x_stride for i in range(len(moduli_values))]
        xticklabels = _moduli_labels_fp64_style(moduli_values)

        for row_idx, ue in enumerate(use_extra_values):
            for col_idx, x_val in enumerate(x_values):
                ax = axes[row_idx][col_idx]

                stacks: List[List[float]] = []
                for mod in moduli_values:
                    r = _get_row(rows, x_value=x_val, use_extra=ue, num_moduli=mod)
                    vals = _stage_values(r)
                    if args.mode == "percent":
                        vals = _to_percent(vals)
                    stacks.append(vals)

                bottom = [0.0] * len(moduli_values)
                for si in active_stage_indices:
                    stage_key, label, color = STAGE_KEYS[si]
                    heights = [s[si] for s in stacks]
                    legend_label = None if stage_key in LEGEND_HIDDEN_STAGE_KEYS else label
                    ax.bar(
                        x_pos,
                        heights,
                        bottom=bottom,
                        color=color,
                        width=args.bar_width,
                        edgecolor="white",
                        linewidth=0.35,
                        label=legend_label if (row_idx == 0 and col_idx == 0) else None,
                    )
                    bottom = [b + h for b, h in zip(bottom, heights)]

                if args.mode == "percent":
                    _annotate_stack_percent_labels(
                        ax,
                        x_pos,
                        stacks,
                        active_stage_indices,
                        min_pct=max(0.0, float(args.label_min_pct)),
                        fontsize=max(4.0, float(args.label_fontsize)),
                    )

                _clean_ax(ax, "y")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(xticklabels, fontsize=8)
                ax.tick_params(direction="out", labelsize=8)
                ax.margins(x=0.0)

                if args.mode == "percent":
                    ax.set_ylim(0, 100)
                    ax.set_yticks([0, 25, 50, 75, 100])
                    ax.set_yticklabels(["0", "", "50", "", "100"])
                    ylabel = "Time share (%)"
                else:
                    y_max = _stacked_ms_ylim(stacks, args.ms_ylim_mode, args.ms_q_high)
                    if y_max is not None:
                        ax.set_ylim(0, y_max)
                    ylabel = "Time (ms)"

                if col_idx == 0:
                    ax.set_ylabel(ylabel, fontsize=10, labelpad=4)
                else:
                    ax.set_yticklabels([])

                if row_idx == 0:
                    if args.x_axis == "n":
                        ax.set_title(f"$n = {x_val}$", fontsize=10, fontweight="bold", pad=6)
                    else:
                        ax.set_title(f"{_x_title_name(args.x_axis)}={x_val}", fontsize=10, fontweight="bold", pad=6)

        handles, labels = axes[0][0].get_legend_handles_labels()
        has_legend = _add_top_legend(fig, handles, labels, fontsize=8)

        fig.supxlabel("Number of moduli", fontsize=10, y=0.06)
        fig.suptitle("i32 Stage Breakdown", fontsize=12, fontweight="bold", y=0.988)
        top = 0.925 if has_legend else 0.955
        fig.tight_layout(rect=(0.0, 0.07, 1.0, top), h_pad=0.35, w_pad=0.20)

    elif args.layout == "compare":
        if args.num_moduli != "all":
            target_moduli = int(args.num_moduli)
        else:
            if args.target_moduli in moduli_values:
                target_moduli = args.target_moduli
            else:
                target_moduli = moduli_values[0]
                if len(moduli_values) > 1:
                    print(
                        f"[WARN] target moduli {args.target_moduli} not found; using {target_moduli}.",
                        file=sys.stderr,
                    )

        rows = [r for r in rows if int(r["_mod"]) == target_moduli]
        x_values = sorted({int(r["_x"]) for r in rows})
        if not x_values:
            print("No rows left after moduli filtering.", file=sys.stderr)
            return 1
        active_stage_indices = _active_stage_indices(rows)

        # One subplot per n value, grouped bars (extra vs compact) in each
        ncols = len(x_values)
        fig, axes = plt.subplots(
            1,
            ncols,
            figsize=(1.00 * ncols + 0.65, 3.35),
            squeeze=False,
        )
        axes_1d = axes[0]

        group_width = args.bar_width
        mode_keys = list(use_extra_values)
        if not mode_keys:
            mode_keys = ["true"]
        n_modes = len(mode_keys)
        bar_w = group_width / float(max(1, n_modes))
        if n_modes == 1:
            offsets = [0.0]
        else:
            offsets = [(-group_width / 2.0) + (i + 0.5) * bar_w for i in range(n_modes)]
        mode_label = {"true": "extra", "false": "compact"}

        for col_idx, x_val in enumerate(x_values):
            ax = axes_1d[col_idx]
            all_stacks: List[List[List[float]]] = []

            for mi, ue in enumerate(mode_keys):
                r = _get_row(rows, x_value=x_val, use_extra=ue, num_moduli=target_moduli)
                vals = _stage_values(r)
                if args.mode == "percent":
                    vals = _to_percent(vals)
                all_stacks.append([vals])

                bottom = 0.0
                for si in active_stage_indices:
                    stage_key, label, color = STAGE_KEYS[si]
                    h = vals[si]
                    legend_label = None if stage_key in LEGEND_HIDDEN_STAGE_KEYS else label
                    lbl = legend_label if (col_idx == 0 and mi == 0) else None
                    ax.bar(
                        1 + offsets[mi],
                        h,
                        bottom=bottom,
                        color=color,
                        width=bar_w * 0.92,
                        edgecolor="white",
                        linewidth=0.4,
                        label=lbl,
                    )
                    bottom += h

                if args.mode == "percent":
                    _annotate_stack_percent_labels(
                        ax,
                        [1 + offsets[mi]],
                        [vals],
                        active_stage_indices,
                        min_pct=max(0.0, float(args.label_min_pct)),
                        fontsize=max(4.0, float(args.label_fontsize)),
                    )

                # Annotate total on top
                total = sum(vals)
                if args.mode == "ms" and total > 0:
                    ax.text(
                        1 + offsets[mi],
                        total,
                        f"{total:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                    )

            ax.set_xticks([1 + off for off in offsets])
            ax.set_xticklabels([mode_label.get(mk, mk) for mk in mode_keys], fontsize=8)
            _clean_ax(ax, "y")
            ax.tick_params(direction="out", labelsize=8)
            ax.margins(x=0.01)
            if args.x_axis == "n":
                ax.set_title(f"$n = {x_val}$", fontsize=10, fontweight="bold", pad=6)
            else:
                ax.set_title(f"{_x_title_name(args.x_axis)}={x_val}", fontsize=10, fontweight="bold", pad=6)

            if args.mode == "percent":
                ax.set_ylim(0, 100)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.set_yticklabels(["0", "", "50", "", "100"])
            else:
                # Compute shared ylim later
                pass

            if col_idx > 0:
                ax.set_yticklabels([])

        # For ms mode, compute shared ylim
        if args.mode == "ms":
            all_totals: List[float] = []
            for col_idx, x_val in enumerate(x_values):
                for ue in mode_keys:
                    r = _get_row(rows, x_value=x_val, use_extra=ue, num_moduli=target_moduli)
                    vals = _stage_values(r)
                    all_totals.append(sum(vals))
            y_max = max(all_totals) * 1.15 if all_totals else 1.0
            for ax in axes_1d:
                ax.set_ylim(0, y_max)

        if args.mode == "percent":
            axes_1d[0].set_ylabel("Time share (%)", fontsize=10)
        else:
            axes_1d[0].set_ylabel("Time (ms)", fontsize=10)

        handles, labels = axes_1d[0].get_legend_handles_labels()
        has_legend = _add_top_legend(fig, handles, labels, fontsize=8)

        fig.suptitle(f"i32 Stage Breakdown (moduli={target_moduli})", fontsize=12, fontweight="bold", y=0.988)
        top = 0.925 if has_legend else 0.955
        fig.tight_layout(rect=(0.0, 0.07, 1.0, top), h_pad=0.35, w_pad=0.20)

    output = args.output
    if not output:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        suffix = "timebreakdown_pct" if args.mode == "percent" else "timebreakdown_ms"
        if args.layout == "n_sweep":
            suffix += "_nsweep"
        elif args.layout == "compare":
            suffix += "_compare"
        if args.x_axis != "n":
            suffix += f"_{args.x_axis}"
        output = f"{base}_{suffix}.png"
    elif not output.lower().endswith(".png"):
        output += ".png"

    fig.savefig(output, dpi=args.dpi)
    print(f"Saved plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
