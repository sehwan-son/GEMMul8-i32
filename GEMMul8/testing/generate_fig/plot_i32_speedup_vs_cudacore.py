#!/usr/bin/env python3
"""Paper-style speedup comparison: CUDA-core vs Ozaki (oz1/oz2) across sizes.

Visual style follows sgemm-opt-intern/plot_4096_ozaki_vs_cuda_core.py.
Order per size: CUDA-core, oz1, oz2.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot speedup over best CUDA-core INT32 kernel by size using "
            "i32_bench_speedup CSV (use_extra=true by default)."
        )
    )
    parser.add_argument(
        "--i32-csv",
        default="/home/sehwan/scale-internship/GEMMul8-i32/GEMMul8/testing/"
        "i32_bench_speedup_NVIDIA_GB10_20260224_040632.csv",
        help="Path to i32_bench_speedup_*.csv",
    )
    parser.add_argument(
        "--cuda-csv",
        default="/home/sehwan/scale-internship/sgemm-opt-intern/int32_core_results.csv",
        help="Path to int32_core_results.csv",
    )
    parser.add_argument("--op", default="NN", choices=["NN", "NT", "TN", "TT"])
    parser.add_argument("--num-moduli", type=int, default=5)
    parser.add_argument("--use-extra", default="true", choices=["true", "false"])
    parser.add_argument("--output", default="", help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Show interactive window")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def to_int(text: str) -> int:
    return int(float(text))


def to_float(text: str) -> float:
    return float(text)


def truthy(value: str) -> bool:
    return value in ("1", "true", "True")


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.facecolor": "#FBFCFE",
            "figure.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "grid.color": "#C7CFDA",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.55,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, output: str, default_base: str, dpi: int) -> List[str]:
    saved: List[str] = []
    if output:
        root, ext = os.path.splitext(output)
        if ext.lower() in (".png", ".pdf", ".svg"):
            kwargs = {"dpi": dpi} if ext.lower() == ".png" else {}
            fig.savefig(output, facecolor="white", **kwargs)
            return [output]
        base = output
    else:
        base = default_base

    png_path = f"{base}.png"
    pdf_path = f"{base}.pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    saved.extend([png_path, pdf_path])
    return saved


def read_cuda_best_by_size(cuda_csv: Path, op: str) -> dict[int, tuple[float, str]]:
    op_a, op_b = op[0], op[1]
    best: dict[int, tuple[float, str]] = {}
    with cuda_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                m = to_int(row["M"])
                n = to_int(row["N"])
                k = to_int(row["K"])
                gflops = to_float(row["gflops"])
            except (KeyError, ValueError):
                continue
            if m != n or n != k:
                continue
            if row.get("opA", "") != op_a or row.get("opB", "") != op_b:
                continue
            if gflops <= 0.0:
                continue
            if row.get("mismatch_count", "") not in ("", "0", "0.0"):
                continue
            if row.get("overflow_safe", "") not in ("", "1", "1.0"):
                continue

            prev = best.get(m)
            if prev is None or gflops > prev[0]:
                best[m] = (gflops, row.get("kernel_name", "unknown"))
    return best


def read_ozaki_best_by_scheme(
    i32_csv: Path,
    op: str,
    num_moduli: int,
    use_extra: bool,
) -> tuple[dict[int, float], dict[int, float]]:
    op_a, op_b = op[0], op[1]
    oz1_best: dict[int, float] = {}
    oz2_best: dict[int, float] = {}

    with i32_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                m = to_int(row["m"])
                n = to_int(row["n"])
                k = to_int(row["k"])
                mod = to_int(row["num_moduli"])
                gflops = to_float(row["gemmul8_gflops"])
            except (KeyError, ValueError):
                continue
            if m != n or n != k:
                continue
            if row.get("opA", "") != op_a or row.get("opB", "") != op_b:
                continue
            if mod != num_moduli:
                continue
            if truthy(row.get("use_extra", "")) != use_extra:
                continue
            if row.get("gemmul8_exact_match", "1") in ("0", "false", "False"):
                continue
            if gflops <= 0.0:
                continue

            scheme = (row.get("i32_scheme", "oz2") or "oz2").strip().lower()
            if scheme == "oz1":
                oz1_best[m] = max(gflops, oz1_best.get(m, 0.0))
            elif scheme == "oz2":
                oz2_best[m] = max(gflops, oz2_best.get(m, 0.0))
    return oz1_best, oz2_best


def main() -> int:
    args = parse_args()
    i32_csv = Path(args.i32_csv).resolve()
    cuda_csv = Path(args.cuda_csv).resolve()
    if not i32_csv.exists():
        raise FileNotFoundError(f"i32 csv not found: {i32_csv}")
    if not cuda_csv.exists():
        raise FileNotFoundError(f"cuda csv not found: {cuda_csv}")

    use_extra_flag = args.use_extra == "true"
    cuda_best = read_cuda_best_by_size(cuda_csv, args.op)
    oz1_best, oz2_best = read_ozaki_best_by_scheme(
        i32_csv=i32_csv,
        op=args.op,
        num_moduli=args.num_moduli,
        use_extra=use_extra_flag,
    )

    sizes = sorted(set(cuda_best.keys()) & set(oz1_best.keys()) & set(oz2_best.keys()))
    if not sizes:
        raise RuntimeError(
            "No overlapping sizes for CUDA-core, oz1, oz2 with current filters "
            f"(op={args.op}, num_moduli={args.num_moduli}, use_extra={args.use_extra})."
        )

    cuda_speedup = np.ones(len(sizes), dtype=float)  # baseline
    oz1_speedup = np.array([oz1_best[s] / cuda_best[s][0] for s in sizes], dtype=float)
    oz2_speedup = np.array([oz2_best[s] / cuda_best[s][0] for s in sizes], dtype=float)

    apply_paper_style()
    fig_w = max(8.5, 1.45 * len(sizes) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.3))

    x = np.arange(len(sizes), dtype=float)
    w = 0.24
    bars_cuda = ax.bar(
        x - w,
        cuda_speedup,
        w,
        label="CUDA-core",
        color="#355F8C",
        edgecolor="#1E1E1E",
        linewidth=0.8,
        zorder=3,
    )
    bars_oz1 = ax.bar(
        x,
        oz1_speedup,
        w,
        label="oz1",
        color="#60A6D5",
        edgecolor="#1E1E1E",
        linewidth=0.8,
        zorder=3,
    )
    bars_oz2 = ax.bar(
        x + w,
        oz2_speedup,
        w,
        label="oz2",
        color="#D88A2D",
        edgecolor="#1E1E1E",
        linewidth=0.8,
        zorder=3,
    )

    ymax = max(1.0, float(np.max(oz1_speedup)), float(np.max(oz2_speedup)))
    pad = ymax * 0.022

    def annotate(bars, color: str, bold: bool = False) -> None:
        for bar in bars:
            h = float(bar.get_height())
            ax.text(
                float(bar.get_x() + bar.get_width() / 2.0),
                h + pad,
                f"{h:.2f}x",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight=("bold" if bold else "normal"),
                color=color,
                zorder=6,
            )

    annotate(bars_cuda, "#1D1D1D")
    annotate(bars_oz1, "#1D1D1D")
    annotate(bars_oz2, "#1D1D1D", bold=True)

    ax.set_title(
        f"INT32 GEMM speedup vs CUDA-core per size (op={args.op})",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Speedup over CUDA-core (x)", fontweight="bold")
    ax.set_xlabel("Matrix size (M=N=K)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in sizes], fontsize=13)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, ymax * 1.28)
    ax.legend(loc="upper left", frameon=True, fontsize=13)
    plt.tight_layout()

    if args.output:
        out_base_or_path = str(Path(args.output).resolve())
    else:
        out_base_or_path = str(i32_csv.with_name(i32_csv.stem + "_speedup_vs_cudacore_oz1_oz2_test"))
    saved = save_figure(fig, out_base_or_path, out_base_or_path, args.dpi)
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print("saved:")
    for p in saved:
        print(f"  - {p}")
    for s in sizes:
        base, kernel = cuda_best[s]
        print(
            f"[INFO] n={s:5d} CUDA={base:.2f} ({kernel}) "
            f"oz1={oz1_best[s]:.2f} ({oz1_best[s]/base:.3f}x) "
            f"oz2={oz2_best[s]:.2f} ({oz2_best[s]/base:.3f}x)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
