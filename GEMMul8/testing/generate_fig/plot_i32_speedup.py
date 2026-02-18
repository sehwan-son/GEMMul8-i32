#!/usr/bin/env python3
"""Plot GEMMul8-i32 benchmark CSV.

Supports:
- single-metric plotting (speedup / ms / GFLOPS)
- stage breakdown plotting (encode/tc/conv32to8/reconstruct[/baseline])
"""

import argparse
import csv
import html
import os
import sys
from collections import defaultdict


METRIC_LABELS = {
    "speedup_vs_exact_i32_i32_i64": "Speedup vs exact int32*int32->int64 GPU baseline",
    "speedup_vs_cublas_i8_single": "Speedup vs cuBLAS int8 (single GEMM)",
    "speedup_vs_cublas_i8_x_moduli": "Speedup vs cuBLAS int8 x num_moduli",
    "gemmul8_total_ms": "GEMMul8 total time (ms)",
    "gemmul8_encode_ms": "GEMMul8 encode time (ms)",
    "gemmul8_tc_ms": "GEMMul8 tensor-core GEMM time (ms)",
    "gemmul8_conv32to8_ms": "GEMMul8 conv32->8 time (ms)",
    "gemmul8_reconstruct_ms": "GEMMul8 reconstruct time (ms)",
    "exact_i32_i32_i64_ms": "Exact int32*int32->int64 time (ms)",
    "cublas_i8_single_ms": "cuBLAS int8 single GEMM time (ms)",
    "cublas_i8_x_moduli_ms": "cuBLAS int8 x num_moduli time (ms)",
    "gemmul8_gflops": "GEMMul8 throughput (GFLOPS)",
    "exact_i32_i32_i64_gflops": "Exact int32*int32->int64 throughput (GFLOPS)",
    "cublas_i8_single_gflops": "cuBLAS int8 single GEMM throughput (GFLOPS)",
    "cublas_i8_x_moduli_gflops": "cuBLAS int8 x num_moduli throughput (GFLOPS)",
}

BREAKDOWN_SERIES = [
    ("gemmul8_encode_ms", "encode"),
    ("gemmul8_tc_ms", "tc_gemm"),
    ("gemmul8_conv32to8_ms", "conv32to8"),
    ("gemmul8_reconstruct_ms", "reconstruct"),
    ("gemmul8_total_ms", "total"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot i32 benchmark metrics from CSV.")
    parser.add_argument("csv", help="Path to i32_bench_speedup_*.csv")
    parser.add_argument(
        "--plot-type",
        default="single",
        choices=["single", "breakdown"],
        help="single: one metric per line / breakdown: stage-wise ms plot",
    )
    parser.add_argument(
        "--metric",
        default="speedup_vs_exact_i32_i32_i64",
        choices=sorted(METRIC_LABELS.keys()),
        help="Y-axis metric for --plot-type single",
    )
    parser.add_argument(
        "--op",
        default="all",
        choices=["all", "NN", "NT", "TN", "TT"],
        help="Operation pair filter",
    )
    parser.add_argument(
        "--use-extra",
        default="all",
        choices=["all", "true", "false"],
        help="Filter by UseExtraWorkspace",
    )
    parser.add_argument(
        "--num-moduli",
        default="all",
        help="Filter by num_moduli (e.g., 9, 20, or all)",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include exact_i32_i32_i64_ms in --plot-type breakdown",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output image path (default: inferred from CSV + plot mode/metric)",
    )
    return parser.parse_args()


def to_bool_text(value: str) -> str:
    return "true" if value in ("1", "true", "True") else "false"


def ensure_extension(path: str, ext: str) -> str:
    root, cur_ext = os.path.splitext(path)
    if cur_ext.lower() == ext.lower():
        return path
    return root + ext


def parse_num_moduli_filter(text: str):
    if text == "all":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid --num-moduli value: {text}") from exc


def metric_value(row: dict, key: str):
    raw = row.get(key, "")
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def save_svg_plot(grouped, title: str, xlabel: str, ylabel: str, output: str) -> None:
    width = 1240
    height = 780
    margin_left = 100
    margin_right = 380
    margin_top = 75
    margin_bottom = 100
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_x = sorted({x for points in grouped.values() for x, _ in points})
    all_y = [y for points in grouped.values() for _, y in points]

    if not all_x or not all_y:
        raise RuntimeError("No points to plot.")

    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)
    y_min = min(0.0, y_min)
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    def x_to_px(x_val: float) -> float:
        if x_max == x_min:
            return margin_left + plot_w * 0.5
        return margin_left + (x_val - x_min) * plot_w / (x_max - x_min)

    def y_to_px(y_val: float) -> float:
        return margin_top + (y_max - y_val) * plot_h / (y_max - y_min)

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#17becf",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]

    y_ticks = 6
    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(
        f'<text x="{width/2:.1f}" y="35" text-anchor="middle" font-size="24" font-family="sans-serif">{html.escape(title)}</text>'
    )
    lines.append(
        f'<text x="{margin_left + plot_w/2:.1f}" y="{height - 25}" text-anchor="middle" font-size="18" font-family="sans-serif">{html.escape(xlabel)}</text>'
    )
    lines.append(
        f'<text x="32" y="{margin_top + plot_h/2:.1f}" transform="rotate(-90 32 {margin_top + plot_h/2:.1f})" text-anchor="middle" font-size="18" font-family="sans-serif">{html.escape(ylabel)}</text>'
    )

    for i in range(y_ticks + 1):
        y_val = y_min + (y_max - y_min) * i / y_ticks
        py = y_to_px(y_val)
        lines.append(
            f'<line x1="{margin_left}" y1="{py:.2f}" x2="{margin_left + plot_w}" y2="{py:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{py + 5:.2f}" text-anchor="end" font-size="13" font-family="monospace">{y_val:.3f}</text>'
        )

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="black" stroke-width="2"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="black" stroke-width="2"/>'
    )

    for x_val in all_x:
        px = x_to_px(x_val)
        lines.append(
            f'<line x1="{px:.2f}" y1="{margin_top + plot_h}" x2="{px:.2f}" y2="{margin_top + plot_h + 7}" stroke="black" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{px:.2f}" y="{margin_top + plot_h + 28}" text-anchor="middle" font-size="13" font-family="monospace">{x_val}</text>'
        )

    legend_x = margin_left + plot_w + 20
    legend_y = margin_top + 10
    for idx, (label, points) in enumerate(sorted(grouped.items())):
        color = colors[idx % len(colors)]
        points_sorted = sorted(points, key=lambda x: x[0])
        path = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in points_sorted)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{path}"/>')
        for x_val, y_val in points_sorted:
            lines.append(
                f'<circle cx="{x_to_px(x_val):.2f}" cy="{y_to_px(y_val):.2f}" r="3.8" fill="{color}" stroke="white" stroke-width="1"/>'
            )

        ly = legend_y + idx * 24
        lines.append(
            f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 24}" y2="{ly}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{legend_x + 30}" y="{ly + 5}" font-size="12" font-family="sans-serif">{html.escape(label)}</text>'
        )

    lines.append("</svg>")
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    args = parse_args()

    try:
        mod_filter = parse_num_moduli_filter(args.num_moduli)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    plt = None
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None

    if not os.path.isfile(args.csv):
        print(f"CSV file not found: {args.csv}", file=sys.stderr)
        return 1

    grouped = defaultdict(list)
    missing_metric_rows = 0

    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_pair = f"{row['opA']}{row['opB']}"
            use_extra = to_bool_text(row["use_extra"])
            mod = int(float(row["num_moduli"]))

            if args.op != "all" and op_pair != args.op:
                continue
            if args.use_extra != "all" and use_extra != args.use_extra:
                continue
            if mod_filter is not None and mod != mod_filter:
                continue

            n = int(float(row["n"]))
            config = f"op={op_pair}, mod={mod}, use_extra={use_extra}"

            if args.plot_type == "single":
                value = metric_value(row, args.metric)
                if value is None:
                    missing_metric_rows += 1
                    continue
                grouped[config].append((n, value))
            else:
                is_fixed_config = (args.op != "all" and args.use_extra != "all" and mod_filter is not None)
                stage_items = list(BREAKDOWN_SERIES)
                if args.include_baseline:
                    stage_items.append(("exact_i32_i32_i64_ms", "baseline_exact_i32_i64"))

                for key, stage_name in stage_items:
                    value = metric_value(row, key)
                    if value is None:
                        continue
                    label = stage_name if is_fixed_config else f"{stage_name}, {config}"
                    grouped[label].append((n, value))

    if not grouped:
        print("No rows matched filters (or selected metric missing).", file=sys.stderr)
        return 1

    if args.plot_type == "single" and missing_metric_rows > 0:
        print(f"[WARN] skipped {missing_metric_rows} rows with missing metric '{args.metric}'.", file=sys.stderr)

    if args.plot_type == "single":
        title = f"GEMMul8 i32 Metric: {args.metric}"
        ylabel = METRIC_LABELS[args.metric]
        suffix = args.metric
    else:
        title = "GEMMul8 i32 Stage Breakdown"
        ylabel = "Time (ms)"
        suffix = "breakdown_ms" if not args.include_baseline else "breakdown_ms_with_baseline"

    xlabel = "Matrix size n (m=n=k=n)"

    output = args.output
    if not output:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        output = f"{base}_{suffix}.png" if plt else f"{base}_{suffix}.svg"

    if plt:
        plt.figure(figsize=(9.8, 5.8))
        for label, points in sorted(grouped.items()):
            pts = sorted(points, key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", linewidth=1.6, label=label)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.legend(fontsize=8)
        plt.tight_layout()
        output = ensure_extension(output, ".png")
        plt.savefig(output, dpi=160)
    else:
        output = ensure_extension(output, ".svg")
        save_svg_plot(grouped, title, xlabel, ylabel, output)
        print("matplotlib not found: generated SVG plot instead.")

    print(f"Saved plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
