#!/usr/bin/env python3
"""Plot i32 benchmark GFLOPS for 4096x4096x4096 only."""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _to_bool_text(value: str) -> str:
    return "true" if value in ("1", "true", "True") else "false"


def _as_int(row: Dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _as_float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


@dataclass
class SeriesPoint:
    op: str
    i32_scheme: str
    use_extra: str
    num_moduli: int
    gflops: float
    total_ms: float
    speedup_vs_exact: float
    exact_gflops: float
    mismatch_count: int
    exact_match: int

    @property
    def label(self) -> str:
        extra_tag = "extra" if self.use_extra == "true" else "compact"
        return f"{self.op}\n{self.i32_scheme}, mod={self.num_moduli}\n{extra_tag}"


def ensure_extension(path: str, ext: str) -> str:
    root, cur_ext = os.path.splitext(path)
    if cur_ext.lower() == ext.lower():
        return path
    return root + ext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot i32 benchmark GFLOPS for 4096x4096x4096.")
    parser.add_argument("csv", help="Path to i32_bench_speedup_*.csv")
    parser.add_argument("--op", default="NN", choices=["all", "NN", "NT", "TN", "TT"])
    parser.add_argument("--i32-scheme", default="all", choices=["all", "oz2", "oz1"])
    parser.add_argument("--size", type=int, default=4096, help="Matrix size filter for M=N=K=size (default: 4096).")
    parser.add_argument("--output", default="", help="Output image path (.png).")
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    return parser.parse_args()


def load_points(csv_path: str, target_size: int, op_filter: str, i32_scheme_filter: str) -> List[SeriesPoint]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    grouped: Dict[Tuple[str, str, int], SeriesPoint] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = _as_int(row, "m")
            n = _as_int(row, "n")
            k = _as_int(row, "k")
            if not (m == target_size and n == target_size and k == target_size):
                continue

            op = f"{row['opA']}{row['opB']}"
            if op_filter != "all" and op != op_filter:
                continue
            i32_scheme = (row.get("i32_scheme", "oz2") or "oz2").strip().lower()
            if i32_scheme not in ("oz2", "oz1"):
                i32_scheme = "oz2"
            if i32_scheme_filter != "all" and i32_scheme != i32_scheme_filter:
                continue

            use_extra = _to_bool_text(row["use_extra"])
            num_moduli = _as_int(row, "num_moduli")
            gflops = _as_float(row, "gemmul8_gflops")
            total_ms = _as_float(row, "gemmul8_total_ms")
            speedup_vs_exact = _as_float(row, "speedup_vs_exact_i32_i32_i64")
            exact_gflops = _as_float(row, "exact_i32_i32_i64_gflops")
            mismatch_count = _as_int(row, "gemmul8_mismatch_count")
            exact_match = _as_int(row, "gemmul8_exact_match")

            if not (math.isfinite(gflops) and math.isfinite(total_ms) and total_ms > 0.0):
                continue

            point = SeriesPoint(
                op=op,
                i32_scheme=i32_scheme,
                use_extra=use_extra,
                num_moduli=num_moduli,
                gflops=gflops,
                total_ms=total_ms,
                speedup_vs_exact=speedup_vs_exact,
                exact_gflops=exact_gflops,
                mismatch_count=mismatch_count,
                exact_match=exact_match,
            )
            key = (op, i32_scheme, use_extra, num_moduli)
            prev = grouped.get(key)
            if prev is None or point.total_ms < prev.total_ms:
                grouped[key] = point

    points = list(grouped.values())
    op_order = {"NN": 0, "NT": 1, "TN": 2, "TT": 3}
    scheme_order = {"oz2": 0, "oz1": 1}
    points.sort(key=lambda p: (op_order.get(p.op, 99), scheme_order.get(p.i32_scheme, 99), p.num_moduli, 0 if p.use_extra == "true" else 1))
    return points


def _plot_svg(points: List[SeriesPoint], output_path: str, title: str) -> str:
    output_svg = ensure_extension(output_path, ".svg")

    width = 1420
    height = 820
    margin_left = 90
    margin_right = 50
    margin_top = 80
    margin_bottom = 230
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    exact_values = [p.exact_gflops for p in points if math.isfinite(p.exact_gflops) and p.exact_gflops > 0.0]
    exact_baseline = exact_values[0] if exact_values else 0.0
    ymax = max([p.gflops for p in points] + ([exact_baseline] if exact_baseline > 0.0 else [0.0]))
    ymax = max(ymax * 1.22, 1.0)

    bar_count = len(points)
    slot_w = plot_w / max(bar_count, 1)
    bar_w = min(64.0, slot_w * 0.62)

    def y_to_px(val: float) -> float:
        return margin_top + (1.0 - (val / ymax)) * plot_h

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    svg.append(
        f'<text x="{width/2:.1f}" y="36" text-anchor="middle" font-size="24" font-family="sans-serif">{html.escape(title)}</text>'
    )

    # Y grid/ticks.
    ticks = 6
    for i in range(ticks + 1):
        v = ymax * i / ticks
        py = y_to_px(v)
        svg.append(
            f'<line x1="{margin_left}" y1="{py:.2f}" x2="{margin_left + plot_w}" y2="{py:.2f}" stroke="#d8d8d8" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{margin_left - 10}" y="{py + 5:.2f}" text-anchor="end" font-size="12" font-family="monospace">{v:.0f}</text>'
        )

    # Axes.
    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="black" stroke-width="2"/>'
    )
    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="black" stroke-width="2"/>'
    )
    svg.append(
        f'<text x="{34}" y="{margin_top + plot_h/2:.1f}" transform="rotate(-90 34 {margin_top + plot_h/2:.1f})" text-anchor="middle" font-size="16" font-family="sans-serif">GFLOPS</text>'
    )
    svg.append(
        f'<text x="{margin_left + plot_w/2:.1f}" y="{height - 24}" text-anchor="middle" font-size="16" font-family="sans-serif">Configuration</text>'
    )

    # Baseline line.
    if exact_baseline > 0.0:
        py = y_to_px(exact_baseline)
        svg.append(
            f'<line x1="{margin_left}" y1="{py:.2f}" x2="{margin_left + plot_w}" y2="{py:.2f}" stroke="#B71C1C" stroke-width="2" stroke-dasharray="8,5"/>'
        )
        svg.append(
            f'<text x="{margin_left + 8}" y="{py - 8:.2f}" text-anchor="start" font-size="12" font-family="sans-serif" fill="#B71C1C">exact baseline {exact_baseline:.1f} GFLOPS</text>'
        )

    # Bars.
    for idx, p in enumerate(points):
        cx = margin_left + (idx + 0.5) * slot_w
        x0 = cx - bar_w / 2
        y0 = y_to_px(p.gflops)
        h = margin_top + plot_h - y0
        color = "#4CAF50" if (p.exact_match == 1 and p.mismatch_count == 0) else "#F39C12"
        svg.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" stroke="black" stroke-width="1"/>'
        )

        svg.append(
            f'<text x="{cx:.2f}" y="{max(y0 - 6, margin_top + 14):.2f}" text-anchor="middle" font-size="11" font-family="sans-serif" font-weight="bold">{p.gflops:.0f}</text>'
        )
        svg.append(
            f'<text x="{cx:.2f}" y="{y0 + max(h * 0.45, 16):.2f}" text-anchor="middle" font-size="10" font-family="sans-serif">{p.speedup_vs_exact:.2f}x</text>'
        )

        label_lines = p.label.split("\n")
        for j, line in enumerate(label_lines):
            ly = margin_top + plot_h + 22 + j * 14
            svg.append(
                f'<text x="{cx:.2f}" y="{ly:.2f}" text-anchor="middle" font-size="11" font-family="sans-serif">{html.escape(line)}</text>'
            )

    svg.append("</svg>")
    with open(output_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    return output_svg


def plot(points: List[SeriesPoint], output_path: str, title: str, show: bool) -> Tuple[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _plot_svg(points, output_path, title), "svg"

    x = list(range(len(points)))
    y = [p.gflops for p in points]
    labels = [p.label for p in points]

    # Green if exact, amber if not exact.
    colors = []
    for p in points:
        if p.exact_match == 1 and p.mismatch_count == 0:
            colors.append("#4CAF50")
        else:
            colors.append("#F39C12")

    exact_values = [p.exact_gflops for p in points if math.isfinite(p.exact_gflops) and p.exact_gflops > 0.0]
    exact_baseline = exact_values[0] if exact_values else 0.0

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
        }
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)

    if exact_baseline > 0.0:
        ax.axhline(
            y=exact_baseline,
            color="#B71C1C",
            linestyle="--",
            linewidth=2.0,
            label=f"exact_i32_i64 baseline ({exact_baseline:.1f} GFLOPS)",
            zorder=3,
        )

    ymax = max(y + ([exact_baseline] if exact_baseline > 0.0 else []))
    ypad = max(1.0, ymax * 0.08)
    ax.set_ylim(0.0, ymax + ypad * 2.0)

    for bar, p in zip(bars, points):
        bx = bar.get_x() + bar.get_width() * 0.5
        by = bar.get_height()
        ax.text(
            bx,
            by + ypad * 0.2,
            f"{p.gflops:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            bx,
            max(by * 0.40, ypad * 0.25),
            f"{p.speedup_vs_exact:.2f}x",
            ha="center",
            va="center",
            fontsize=8,
            color="#1f1f1f",
        )

    ax.set_title(title, pad=12)
    ax.set_ylabel("GFLOPS")
    ax.set_xlabel("Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.grid(axis="y", linestyle="-", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)

    if exact_baseline > 0.0:
        ax.legend(loc="upper left", frameon=True)

    output_png = ensure_extension(output_path, ".png")
    fig.tight_layout()
    fig.savefig(output_png, dpi=280, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_png, "matplotlib"


def main() -> int:
    args = parse_args()
    try:
        points = load_points(args.csv, args.size, args.op, args.i32_scheme)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not points:
        print("No matching rows found for requested filter.", file=sys.stderr)
        return 1

    if args.output:
        output = args.output
    else:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        op_tag = args.op if args.op != "all" else "all_ops"
        scheme_tag = args.i32_scheme if args.i32_scheme != "all" else "all_schemes"
        output = f"{base}_{args.size}_{op_tag}_{scheme_tag}_gflops.png"

    title = f"GEMMul8 i32 GFLOPS @ {args.size}x{args.size}x{args.size}"
    if args.op != "all":
        title += f" ({args.op})"
    if args.i32_scheme != "all":
        title += f" [{args.i32_scheme}]"

    saved_path, backend = plot(points, output, title, args.show)

    best = max(points, key=lambda p: p.gflops)
    if backend == "svg":
        print("matplotlib not found: generated SVG plot instead.")
    print(f"Saved plot: {saved_path}")
    print(
        "Best config: "
        f"op={best.op}, scheme={best.i32_scheme}, use_extra={best.use_extra}, mod={best.num_moduli}, "
        f"gflops={best.gflops:.2f}, speedup_vs_exact={best.speedup_vs_exact:.3f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
