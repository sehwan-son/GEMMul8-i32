#!/usr/bin/env python3
"""Plot i32 stage breakdown in oz2-style stacked bars."""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Sequence

from plot_i32_common import metric_value, read_filtered_rows


STAGE_KEYS = [
    ("gemmul8_encode_ms", "encode", "#F6AA00"),
    ("gemmul8_tc_ms", "tc_gemm", "#03AF7A"),
    ("gemmul8_conv32to8_ms", "conv32to8", "#005AFF"),
    ("gemmul8_reconstruct_ms", "reconstruct", "#FF4B00"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot i32 stage-wise time breakdown (oz2-style).")
    parser.add_argument("csv", help="Path to i32_bench_speedup_*.csv")
    parser.add_argument("--op", default="all", choices=["all", "NN", "NT", "TN", "TT"])
    parser.add_argument("--use-extra", default="all", choices=["all", "true", "false"])
    parser.add_argument("--num-moduli", default="all")
    parser.add_argument("--x-axis", default="n", choices=["n", "m", "k", "mnk"])
    parser.add_argument(
        "--layout",
        default="n_sweep",
        choices=["n_sweep", "moduli_sweep"],
        help="n_sweep: fixed moduli and compare x-axis change (recommended), moduli_sweep: previous oz2-like view.",
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
        help="percent: stacked percentage (oz2 style), ms: stacked absolute ms.",
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
    parser.add_argument("--bar-width", type=float, default=0.78)
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


def main() -> int:
    args = parse_args()
    plt = _import_matplotlib()
    if plt is None:
        return 1

    try:
        rows = read_filtered_rows(args.csv, args.op, args.use_extra, args.num_moduli, x_axis=args.x_axis)
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
                print(
                    f"[WARN] target moduli {args.target_moduli} not found; using {target_moduli}.",
                    file=sys.stderr,
                )

        rows = [r for r in rows if int(r["_mod"]) == target_moduli]
        x_values = sorted({int(r["_x"]) for r in rows})
        if not x_values:
            print("No rows left after moduli filtering.", file=sys.stderr)
            return 1

        nrows = len(use_extra_values)
        fig, axes = plt.subplots(
            nrows,
            1,
            figsize=(2.3 * max(3, len(x_values)) + 1.4, 2.5 * nrows + 2.0),
            squeeze=False,
        )

        x_pos = list(range(1, len(x_values) + 1))
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
            for si, (_, label, color) in enumerate(STAGE_KEYS):
                heights = [s[si] for s in stacks]
                ax.bar(
                    x_pos,
                    heights,
                    bottom=bottom,
                    color=color,
                    width=args.bar_width,
                    edgecolor="white",
                    linewidth=0.4,
                    label=label if row_idx == 0 else None,
                )
                bottom = [b + h for b, h in zip(bottom, heights)]

            ax.grid(True, axis="y", alpha=0.3)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(xticklabels)
            ax.tick_params(direction="out", labelsize=8)
            if args.mode == "percent":
                ax.set_ylim(0, 100)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.set_ylabel("%", fontsize=9)
            else:
                y_max = _stacked_ms_ylim(stacks, args.ms_ylim_mode, args.ms_q_high)
                if y_max is not None:
                    ax.set_ylim(0, y_max)
                ax.set_ylabel("ms", fontsize=9)
            ax.set_title(f"use_extra={ue}", fontsize=10)

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

        fig.supxlabel(_x_title_name(args.x_axis), fontsize=10)
        fig.suptitle(f"i32 Stage Breakdown (moduli={target_moduli})", fontsize=11)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    else:
        x_values = sorted({int(r["_x"]) for r in rows})
        nrows = len(use_extra_values)
        ncols = len(x_values)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(2.0 * ncols + 1.8, 2.05 * nrows + 1.9),
            squeeze=False,
        )

        x_pos = list(range(1, len(moduli_values) + 1))
        xticklabels = _sparse_labels(moduli_values)

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
                for si, (_, label, color) in enumerate(STAGE_KEYS):
                    heights = [s[si] for s in stacks]
                    ax.bar(
                        x_pos,
                        heights,
                        bottom=bottom,
                        color=color,
                        width=args.bar_width,
                        edgecolor="white",
                        linewidth=0.35,
                        label=label if (row_idx == 0 and col_idx == 0) else None,
                    )
                    bottom = [b + h for b, h in zip(bottom, heights)]

                ax.grid(True, axis="y", alpha=0.3)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(xticklabels)
                ax.tick_params(direction="out", labelsize=8)

                if args.mode == "percent":
                    ax.set_ylim(0, 100)
                    ax.set_yticks([0, 25, 50, 75, 100])
                    ylabel = "% (use_extra=true)" if ue == "true" else "% (use_extra=false)"
                else:
                    y_max = _stacked_ms_ylim(stacks, args.ms_ylim_mode, args.ms_q_high)
                    if y_max is not None:
                        ax.set_ylim(0, y_max)
                    ylabel = "ms (use_extra=true)" if ue == "true" else "ms (use_extra=false)"

                if col_idx == 0:
                    ax.set_ylabel(ylabel, fontsize=9)
                else:
                    ax.set_yticklabels([])

                if row_idx == 0:
                    ax.set_title(f"{_x_title_name(args.x_axis)}={x_val}", fontsize=10)

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
        fig.suptitle("i32 Stage Breakdown", fontsize=11)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    output = args.output
    if not output:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        suffix = "timebreakdown_pct" if args.mode == "percent" else "timebreakdown_ms"
        if args.layout == "n_sweep":
            suffix += "_nsweep"
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
