#!/usr/bin/env python3
"""Generate publication-oriented i32 benchmark figures.

The script verifies core CSV semantics and produces focused plots that are easier
for paper figures than the generic all-series charts.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


OPS = ("NN", "NT", "TN", "TT")
USE_EXTRA_VALUES = ("true", "false")

SPEEDUP_KEY = "speedup_vs_exact_i32_i32_i64"
EXACT_MS_KEY = "exact_i32_i32_i64_ms"
TOTAL_MS_KEY = "gemmul8_total_ms"

STAGE_MS_KEYS = [
    ("gemmul8_encode_ms", "encode"),
    ("gemmul8_tc_ms", "tc_gemm"),
    ("gemmul8_conv32to8_ms", "conv32to8"),
    ("gemmul8_reconstruct_ms", "reconstruct"),
]

OP_COLORS = {
    "NN": "#1F77B4",
    "NT": "#FF7F0E",
    "TN": "#2CA02C",
    "TT": "#D62728",
}

METHOD_COLORS = {
    "gemmul8_true": "#1565C0",
    "gemmul8_false": "#E65100",
    "exact": "#2E7D32",
}

STAGE_COLORS = {
    "encode": "#4E79A7",
    "tc_gemm": "#59A14F",
    "conv32to8": "#F28E2B",
    "reconstruct": "#E15759",
}


def _to_bool_text(value: str) -> str:
    return "true" if value in ("1", "true", "True") else "false"


def _as_float(row: Dict[str, str], key: str) -> Optional[float]:
    raw = (row.get(key) or "").strip()
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _as_int(row: Dict[str, str], key: str) -> Optional[int]:
    value = _as_float(row, key)
    if value is None:
        return None
    return int(round(value))


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _parse_num_moduli_filter(text: str) -> Optional[int]:
    if text == "all":
        return None
    return int(text)


def _read_rows(
    csv_path: str,
    op_filter: str,
    use_extra_filter: str,
    num_moduli_filter: str,
) -> List[Dict[str, object]]:
    mod_filter = _parse_num_moduli_filter(num_moduli_filter)
    rows: List[Dict[str, object]] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_pair = f"{row['opA']}{row['opB']}"
            use_extra = _to_bool_text(row["use_extra"])
            moduli = _as_int(row, "num_moduli")
            n = _as_int(row, "n")
            m = _as_int(row, "m")
            k = _as_int(row, "k")

            if None in (moduli, n, m, k):
                continue
            if op_filter != "all" and op_pair != op_filter:
                continue
            if use_extra_filter != "all" and use_extra != use_extra_filter:
                continue
            if mod_filter is not None and moduli != mod_filter:
                continue

            rec: Dict[str, object] = dict(row)
            rec["_op"] = op_pair
            rec["_use_extra"] = use_extra
            rec["_mod"] = moduli
            rec["_n"] = n
            rec["_m"] = m
            rec["_k"] = k
            rows.append(rec)

    return rows


def _rows_for(rows: Sequence[Dict[str, object]], *, op: Optional[str] = None, use_extra: Optional[str] = None) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        if op is not None and str(row["_op"]) != op:
            continue
        if use_extra is not None and str(row["_use_extra"]) != use_extra:
            continue
        out.append(row)
    return out


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except ModuleNotFoundError:
        print("matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
        return None


def _apply_paper_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.facecolor": "#FAFBFC",
            "figure.facecolor": "white",
            "axes.edgecolor": "#4A4A4A",
            "axes.linewidth": 0.9,
            "grid.color": "#C8CFD9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.6,
            "lines.linewidth": 2.1,
            "lines.markersize": 5.5,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "#CFD6DF",
            "savefig.bbox": "tight",
        }
    )


def _sorted_xy(rows: Sequence[Dict[str, object]], key: str) -> Tuple[List[int], List[float]]:
    points: List[Tuple[int, float]] = []
    for row in rows:
        n = int(row["_n"])
        v = _as_float(row, key)
        if v is None:
            continue
        points.append((n, v))
    points.sort(key=lambda p: p[0])
    return [p[0] for p in points], [p[1] for p in points]


def _average_by_n(rows: Sequence[Dict[str, object]], key: str) -> Tuple[List[int], List[float]]:
    buckets: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        n = int(row["_n"])
        v = _as_float(row, key)
        if v is None:
            continue
        buckets[n].append(v)

    n_values = sorted(buckets.keys())
    y_values = [sum(buckets[n]) / len(buckets[n]) for n in n_values]
    return n_values, y_values


def _plot_speedup(plt, rows: Sequence[Dict[str, object]], prefix: str, dpi: int) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)

    global_best: Optional[Tuple[float, str, str, int]] = None
    all_speedups: List[float] = []

    for idx, use_extra in enumerate(USE_EXTRA_VALUES):
        ax = axes[idx]
        for op in OPS:
            sub = _rows_for(rows, op=op, use_extra=use_extra)
            xs, ys = _sorted_xy(sub, SPEEDUP_KEY)
            if not xs:
                continue
            all_speedups.extend(ys)
            ax.plot(xs, ys, marker="o", color=OP_COLORS[op], label=op)

            local_best = max(zip(xs, ys), key=lambda p: p[1])
            if global_best is None or local_best[1] > global_best[0]:
                global_best = (local_best[1], op, use_extra, local_best[0])

        ax.axhline(1.0, color="#555555", linestyle=(0, (4, 3)), linewidth=1.15)
        ax.set_title(f"UseExtraWorkspace={use_extra}")
        ax.set_xlabel("n (m=n=k)")
        ax.grid(True)
        if idx == 0:
            ax.set_ylabel("Speedup vs exact i32\u00d7i32\u2192int64")

    if global_best is not None:
        best_val, best_op, best_extra, best_n = global_best
        target_ax = axes[0] if best_extra == "true" else axes[1]
        target_ax.annotate(
            f"max {best_val:.2f}\u00d7 ({best_op}, n={best_n})",
            xy=(best_n, best_val),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#AAAAAA", "alpha": 0.95},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03), title="op")

    median_speedup = sorted(all_speedups)[len(all_speedups) // 2] if all_speedups else float("nan")
    fig.suptitle("GEMMul8 i32 Emulation Speedup", y=1.06, fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        f"Reference: exact i32\u00d7i32\u2192int64 GPU kernel. Median speedup={median_speedup:.2f}\u00d7",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0.01, 0.05, 0.99, 0.94))
    out = f"{prefix}_paper_speedup.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _plot_throughput(plt, rows: Sequence[Dict[str, object]], prefix: str, dpi: int) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharex=True, sharey=True)

    for idx, op in enumerate(OPS):
        ax = axes[idx // 2][idx % 2]

        rows_op = _rows_for(rows, op=op)
        true_rows = _rows_for(rows_op, use_extra="true")
        false_rows = _rows_for(rows_op, use_extra="false")

        x_true, y_true = _sorted_xy(true_rows, "gemmul8_gflops")
        x_false, y_false = _sorted_xy(false_rows, "gemmul8_gflops")
        x_exact, y_exact = _average_by_n(rows_op, "exact_i32_i32_i64_gflops")

        if x_true:
            ax.plot(
                x_true,
                y_true,
                marker="o",
                color=METHOD_COLORS["gemmul8_true"],
                label="GEMMul8 (use_extra=true)",
            )
        if x_false:
            ax.plot(
                x_false,
                y_false,
                marker="s",
                color=METHOD_COLORS["gemmul8_false"],
                label="GEMMul8 (use_extra=false)",
            )
        if x_exact:
            ax.plot(
                x_exact,
                y_exact,
                marker="^",
                color=METHOD_COLORS["exact"],
                linestyle=(0, (5, 2)),
                label="Exact i32\u00d7i32\u2192int64 baseline",
            )

        peak = max(y_true + y_false + y_exact) if (y_true or y_false or y_exact) else 0.0
        ax.set_title(f"op={op}   peak={peak:.0f} GFLOPS")
        ax.grid(True)

        if idx // 2 == 1:
            ax.set_xlabel("n (m=n=k)")
        if idx % 2 == 0:
            ax.set_ylabel("GFLOPS")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))

    fig.suptitle("Throughput by Operation (paper view)", y=1.06, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.95))

    out = f"{prefix}_paper_throughput.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _avg_stage_percent(rows: Sequence[Dict[str, object]], use_extra: str, n: int) -> List[float]:
    subset = [r for r in rows if str(r["_use_extra"]) == use_extra and int(r["_n"]) == n]
    if not subset:
        return [0.0 for _ in STAGE_MS_KEYS]

    stage_totals = [0.0 for _ in STAGE_MS_KEYS]
    total_sum = 0.0

    for row in subset:
        vals: List[float] = []
        for key, _ in STAGE_MS_KEYS:
            v = _as_float(row, key)
            vals.append(0.0 if v is None else max(0.0, v))
        subtotal = sum(vals)
        total_sum += subtotal
        for i, v in enumerate(vals):
            stage_totals[i] += v

    if total_sum <= 0.0:
        return [0.0 for _ in STAGE_MS_KEYS]
    return [v * 100.0 / total_sum for v in stage_totals]


def _plot_time_breakdown(plt, rows: Sequence[Dict[str, object]], prefix: str, dpi: int) -> str:
    n_values = sorted({int(r["_n"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)

    for idx, use_extra in enumerate(USE_EXTRA_VALUES):
        ax = axes[idx]
        x = list(range(len(n_values)))
        bottoms = [0.0] * len(n_values)

        all_stage_values: List[List[float]] = []
        for stage_idx, (_, stage_name) in enumerate(STAGE_MS_KEYS):
            vals = [_avg_stage_percent(rows, use_extra, n)[stage_idx] for n in n_values]
            all_stage_values.append(vals)

            bars = ax.bar(
                x,
                vals,
                bottom=bottoms,
                color=STAGE_COLORS[stage_name],
                edgecolor="white",
                linewidth=0.6,
                width=0.74,
                label=stage_name,
            )

            for bi, bar in enumerate(bars):
                val = vals[bi]
                if val >= 12.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bottoms[bi] + val / 2.0,
                        f"{val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                    )
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in n_values])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, axis="y")
        ax.set_title(f"UseExtraWorkspace={use_extra}")
        ax.set_xlabel("n (m=n=k)")
        if idx == 0:
            ax.set_ylabel("Runtime share (%)")

        conv_vals = all_stage_values[2]
        if use_extra == "true" and conv_vals and max(conv_vals) < 1e-6:
            ax.text(
                0.98,
                0.95,
                "conv32to8 is eliminated",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "fc": "#FFFFFF", "ec": "#BBBBBB", "alpha": 0.9},
            )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))

    fig.suptitle("Stage Time Breakdown (mean over ops)", y=1.06, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.94))

    out = f"{prefix}_paper_timebreakdown.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _plot_correctness_and_guarantee(plt, rows: Sequence[Dict[str, object]], prefix: str, dpi: int) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))

    # Left: required moduli vs n, with used moduli line.
    ax = axes[0]
    used_moduli_values = sorted({int(r["_mod"]) for r in rows})
    used_moduli = used_moduli_values[0] if used_moduli_values else 0

    max_required = 0.0
    for op in OPS:
        rows_op = _rows_for(rows, op=op)
        xs, ys = _average_by_n(rows_op, "required_num_moduli")
        if xs:
            ax.plot(xs, ys, marker="o", color=OP_COLORS[op], label=op)
            max_required = max(max_required, max(ys))

    if used_moduli > 0:
        ax.axhline(
            used_moduli,
            color="#111111",
            linewidth=1.4,
            linestyle=(0, (4, 3)),
            label=f"used moduli = {used_moduli}",
        )

    ax.set_title("Required moduli for exact reconstruction")
    ax.set_xlabel("n (m=n=k)")
    ax.set_ylabel("# of moduli")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    # Right: concise correctness certificate text + small bar.
    ax2 = axes[1]
    ax2.axis("off")

    total_cfg = len(rows)
    gpu_exact = sum(1 for r in rows if (_as_float(r, "gemmul8_mismatch_count") or 0.0) == 0.0 and (_as_float(r, "gemmul8_max_abs_error") or 0.0) == 0.0)
    cpu_exact = sum(1 for r in rows if (_as_float(r, "cpu_mismatch") or 0.0) == 0.0 and (_as_float(r, "cpu_max_abs_error") or 0.0) == 0.0)
    precision_guaranteed = sum(1 for r in rows if (_as_float(r, "precision_guaranteed") or 0.0) >= 1.0)
    max_abs_gpu = max((_as_float(r, "gemmul8_max_abs_error") or 0.0) for r in rows) if rows else 0.0
    max_abs_cpu = max((_as_float(r, "cpu_max_abs_error") or 0.0) for r in rows) if rows else 0.0

    margin = used_moduli - max_required
    lines = [
        f"Configurations tested: {total_cfg}",
        f"GPU exact match: {gpu_exact}/{total_cfg}",
        f"CPU sampled match: {cpu_exact}/{total_cfg}",
        f"max abs error (GPU): {max_abs_gpu:.3g}",
        f"max abs error (CPU): {max_abs_cpu:.3g}",
        f"precision_guaranteed=1: {precision_guaranteed}/{total_cfg}",
        f"moduli margin: used {used_moduli} - required max {max_required:.1f} = {margin:.1f}",
    ]

    ax2.text(0.03, 0.95, "Correctness Certificate", fontsize=13, fontweight="bold", va="top")
    ax2.text(
        0.03,
        0.86,
        "\n".join(lines),
        fontsize=10,
        va="top",
        family="DejaVu Sans Mono",
        linespacing=1.5,
    )

    inset = ax2.inset_axes([0.08, 0.08, 0.84, 0.28])
    y_pos = [0, 1]
    labels = ["required (max)", "used"]
    inset.barh(y_pos, [max_required, float(used_moduli)], color=["#8AB17D", "#355070"])
    inset.set_yticks(y_pos)
    inset.set_yticklabels(labels)
    inset.set_xlabel("# of moduli", fontsize=9)
    inset.tick_params(labelsize=8)
    inset.grid(True, axis="x", alpha=0.4)

    fig.suptitle("Accuracy and Guarantee Summary", y=1.04, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.96))

    out = f"{prefix}_paper_correctness.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _validate_semantics(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    speedup_abs_diff_max = 0.0
    speedup_checked = 0

    pct_sum_abs_diff_max = 0.0
    pct_sum_checked = 0

    for row in rows:
        exact_ms = _as_float(row, EXACT_MS_KEY)
        total_ms = _as_float(row, TOTAL_MS_KEY)
        speedup = _as_float(row, SPEEDUP_KEY)
        if exact_ms is not None and total_ms is not None and speedup is not None and total_ms > 0.0:
            calc = exact_ms / total_ms
            diff = abs(calc - speedup)
            speedup_abs_diff_max = max(speedup_abs_diff_max, diff)
            speedup_checked += 1

        pcts = []
        for key in ("encode_pct", "tc_pct", "conv32to8_pct", "reconstruct_pct"):
            val = _as_float(row, key)
            if val is not None:
                pcts.append(val)
        if len(pcts) == 4:
            pct_sum_abs_diff_max = max(pct_sum_abs_diff_max, abs(sum(pcts) - 100.0))
            pct_sum_checked += 1

    return {
        "speedup_abs_diff_max": speedup_abs_diff_max,
        "speedup_rows_checked": float(speedup_checked),
        "pct_sum_abs_diff_max": pct_sum_abs_diff_max,
        "pct_rows_checked": float(pct_sum_checked),
    }


def _write_summary(rows: Sequence[Dict[str, object]], validation: Dict[str, float], out_path: str) -> None:
    total_cfg = len(rows)
    if total_cfg == 0:
        text = "No rows matched filters."
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        return

    speedups = [_as_float(r, SPEEDUP_KEY) or 0.0 for r in rows]
    best = max(rows, key=lambda r: _as_float(r, SPEEDUP_KEY) or -1.0)

    gpu_exact = sum(1 for r in rows if (_as_float(r, "gemmul8_mismatch_count") or 0.0) == 0.0 and (_as_float(r, "gemmul8_max_abs_error") or 0.0) == 0.0)
    cpu_exact = sum(1 for r in rows if (_as_float(r, "cpu_mismatch") or 0.0) == 0.0 and (_as_float(r, "cpu_max_abs_error") or 0.0) == 0.0)

    req_vals = [(_as_float(r, "required_num_moduli") or 0.0) for r in rows]
    used_vals = sorted({int(r["_mod"]) for r in rows})
    used_mod = used_vals[0] if used_vals else 0

    lines = [
        "i32 plot semantic check summary",
        f"rows_matched={total_cfg}",
        f"speedup_check_rows={int(validation['speedup_rows_checked'])}",
        f"speedup_abs_diff_max={validation['speedup_abs_diff_max']:.6e}",
        f"pct_check_rows={int(validation['pct_rows_checked'])}",
        f"pct_sum_abs_diff_max={validation['pct_sum_abs_diff_max']:.6e}",
        f"speedup_min={min(speedups):.6f}",
        f"speedup_mean={sum(speedups)/len(speedups):.6f}",
        f"speedup_max={max(speedups):.6f}",
        f"best_speedup_cfg=op={best['_op']},use_extra={best['_use_extra']},n={best['_n']},value={(_as_float(best, SPEEDUP_KEY) or 0.0):.6f}",
        f"gpu_exact_match={gpu_exact}/{total_cfg}",
        f"cpu_sample_exact_match={cpu_exact}/{total_cfg}",
        f"required_moduli_min={min(req_vals):.2f}",
        f"required_moduli_max={max(req_vals):.2f}",
        f"used_moduli={used_mod}",
        f"moduli_margin={used_mod - max(req_vals):.2f}",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-style i32 benchmark plots.")
    parser.add_argument("csv", help="Path to i32_bench_speedup_*.csv")
    parser.add_argument("--op", default="all", choices=["all", "NN", "NT", "TN", "TT"])
    parser.add_argument("--use-extra", default="all", choices=["all", "true", "false"])
    parser.add_argument("--num-moduli", default="all")
    parser.add_argument(
        "--x-axis",
        default="n",
        choices=["n", "m", "k", "mnk"],
        help="Compatibility option for plot_all.py (paper plots currently use n-axis view).",
    )
    parser.add_argument("--dpi", type=int, default=380)
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Output path prefix (default: <csv_stem> in current directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plt = _ensure_matplotlib()
    if plt is None:
        return 1

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        rows = _read_rows(str(csv_path), args.op, args.use_extra, args.num_moduli)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not rows:
        print("No rows matched the filters.", file=sys.stderr)
        return 1

    _apply_paper_style(plt)

    prefix = args.output_prefix.strip()
    if not prefix:
        prefix = csv_path.stem

    validation = _validate_semantics(rows)

    outputs = [
        _plot_speedup(plt, rows, prefix, args.dpi),
        _plot_throughput(plt, rows, prefix, args.dpi),
        _plot_time_breakdown(plt, rows, prefix, args.dpi),
        _plot_correctness_and_guarantee(plt, rows, prefix, args.dpi),
    ]

    summary_path = f"{prefix}_paper_summary.txt"
    _write_summary(rows, validation, summary_path)

    print(
        "Semantic checks: "
        f"speedup_abs_diff_max={validation['speedup_abs_diff_max']:.3e}, "
        f"pct_sum_abs_diff_max={validation['pct_sum_abs_diff_max']:.3e}"
    )
    for out in outputs:
        print(f"Saved plot: {out}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
