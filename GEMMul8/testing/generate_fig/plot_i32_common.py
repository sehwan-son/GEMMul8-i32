#!/usr/bin/env python3
"""Common helpers for i32 plotting scripts."""

from __future__ import annotations

import csv
import html
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def to_bool_text(value: str) -> str:
    return "true" if value in ("1", "true", "True") else "false"


def ensure_extension(path: str, ext: str) -> str:
    root, cur_ext = os.path.splitext(path)
    if cur_ext.lower() == ext.lower():
        return path
    return root + ext


def parse_num_moduli_filter(text: str) -> Optional[int]:
    if text == "all":
        return None
    return int(text)


def metric_value(row: Dict[str, str], key: str) -> Optional[float]:
    raw = row.get(key, "")
    if raw == "":
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def read_fieldnames(csv_path: str) -> List[str]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def read_filtered_rows(
    csv_path: str,
    op: str = "all",
    use_extra: str = "all",
    num_moduli: str = "all",
    x_axis: str = "n",
) -> List[Dict[str, object]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    mod_filter = parse_num_moduli_filter(num_moduli)
    out: List[Dict[str, object]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_pair = f"{row['opA']}{row['opB']}"
            use_extra_text = to_bool_text(row["use_extra"])
            mod = int(float(row["num_moduli"]))
            m = int(float(row["m"]))
            n = int(float(row["n"]))
            k = int(float(row["k"]))

            if op != "all" and op_pair != op:
                continue
            if use_extra != "all" and use_extra_text != use_extra:
                continue
            if mod_filter is not None and mod != mod_filter:
                continue

            if x_axis == "n":
                x_value = n
            elif x_axis == "m":
                x_value = m
            elif x_axis == "k":
                x_value = k
            elif x_axis == "mnk":
                x_value = m * n * k
            else:
                raise ValueError(f"Invalid x_axis: {x_axis}")

            rec: Dict[str, object] = dict(row)
            rec["_op_pair"] = op_pair
            rec["_use_extra"] = use_extra_text
            rec["_mod"] = mod
            rec["_m"] = m
            rec["_n"] = n
            rec["_k"] = k
            rec["_shape"] = f"{m}x{n}x{k}"
            rec["_x"] = int(x_value)
            rec["_config"] = f"op={op_pair}, mod={mod}, use_extra={use_extra_text}"
            out.append(rec)
    return out


@dataclass(frozen=True)
class LinePlotOptions:
    yscale: str = "auto"  # auto|linear|log
    ylim_mode: str = "robust"  # robust|full
    q_low: float = 0.02
    q_high: float = 0.98
    include_zero: bool = True
    dpi: int = 260


def _save_svg_plot(
    grouped: Dict[str, List[Tuple[int, float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    output: str,
) -> None:
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


def _quantile(sorted_vals: List[float], q: float) -> float:
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


def _resolve_yscale(all_y: List[float], yscale: str) -> str:
    if yscale in ("linear", "log"):
        return yscale
    positive = sorted(v for v in all_y if v > 0.0 and math.isfinite(v))
    if len(positive) >= 2:
        spread = positive[-1] / max(positive[0], 1e-30)
        if spread >= 300.0:
            return "log"
    return "linear"


def _resolve_ylim(
    all_y: List[float],
    yscale: str,
    ylabel: str,
    options: LinePlotOptions,
) -> Optional[Tuple[float, float]]:
    if yscale == "log":
        vals = sorted(v for v in all_y if v > 0.0 and math.isfinite(v))
    else:
        vals = sorted(v for v in all_y if math.isfinite(v))
    if not vals:
        return None

    if options.ylim_mode == "full":
        y0 = vals[0]
        y1 = vals[-1]
    else:
        q_low = min(max(options.q_low, 0.0), 1.0)
        q_high = min(max(options.q_high, 0.0), 1.0)
        if q_high < q_low:
            q_low, q_high = q_high, q_low
        y0 = _quantile(vals, q_low)
        y1 = _quantile(vals, q_high)
        if not (math.isfinite(y0) and math.isfinite(y1)) or y1 <= y0:
            y0 = vals[0]
            y1 = vals[-1]

    if yscale == "log":
        y0 = max(y0, vals[0], 1e-30)
        y1 = max(y1, y0 * 1.0001)
        # Small margin in log domain.
        return (y0 * 0.92, y1 * 1.08)

    if y1 <= y0:
        y1 = y0 + max(abs(y0) * 0.1, 1.0)
    pad = (y1 - y0) * 0.08
    y0 -= pad
    y1 += pad

    low_label = ylabel.lower()
    if options.include_zero and vals[0] >= 0.0:
        y0 = 0.0
    if "speedup" in low_label and vals[0] >= 0.0:
        y0 = 0.0
    if "exact_match" in low_label:
        y0 = min(y0, -0.05)
        y1 = max(y1, 1.05)
    return (y0, y1)


def _apply_paper_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.9,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "lines.linewidth": 1.8,
        }
    )


def plot_lines(
    grouped: Dict[str, List[Tuple[int, float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    output: str,
    options: Optional[LinePlotOptions] = None,
) -> Tuple[str, str]:
    if not grouped:
        raise RuntimeError("No points to plot.")
    opts = options or LinePlotOptions()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None

    if plt is not None:
        _apply_paper_style(plt)
        fig, ax = plt.subplots(figsize=(11.8, 6.6))
        all_x = sorted({x for points in grouped.values() for x, _ in points})
        all_y = [y for points in grouped.values() for _, y in points if math.isfinite(y)]
        yscale = _resolve_yscale(all_y, opts.yscale)

        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"]
        linestyles = ["-", "--", "-.", ":"]

        for idx, (label, points) in enumerate(sorted(grouped.items())):
            pts = sorted(points, key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(
                xs,
                ys,
                marker=markers[idx % len(markers)],
                linestyle=linestyles[idx % len(linestyles)],
                markersize=4.5,
                label=label,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        ax.grid(True, axis="both", linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if all_x:
            if len(all_x) <= 14:
                ax.set_xticks(all_x)
            else:
                step = max(1, len(all_x) // 10)
                ax.set_xticks(all_x[::step])

        ax.set_yscale(yscale)
        ylim = _resolve_ylim(all_y, yscale, ylabel, opts)
        if ylim is not None:
            ax.set_ylim(*ylim)

        series_count = len(grouped)
        if series_count <= 7:
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
        else:
            if series_count <= 14:
                legend_ncol = 2
            elif series_count <= 28:
                legend_ncol = 3
            else:
                legend_ncol = 4
            ax.legend(
                fontsize=8,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.22),
                ncol=legend_ncol,
                frameon=False,
            )
            fig.subplots_adjust(top=0.73)

        output = ensure_extension(output, ".png")
        fig.savefig(output, dpi=opts.dpi)
        plt.close(fig)
        return output, "matplotlib"

    output = ensure_extension(output, ".svg")
    _save_svg_plot(grouped, title, xlabel, ylabel, output)
    return output, "svg"
