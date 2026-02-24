#!/usr/bin/env python3
"""Plot i32 input-value distributions for multiple phi values.

The sampler exactly follows `fill_matrix_exponent_i32()` in test_int32_bench.cu:
  x = U(-0.5, 0.5) * exp(N(0,1) * phi) * 2^scale_exp
  clamped = clip(x, -input_bound, input_bound)
  out = llround(clamped)  # halfway away from zero
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PhiStats:
    phi: float
    mean: float
    std: float
    zero_pct: float
    sat_pct: float
    p95_abs: float
    p99_abs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot i32 value distributions for several phi values.")
    parser.add_argument("--phis", default="0.5,1,2,3,4", help="Comma-separated phi values.")
    parser.add_argument("--samples", type=int, default=400000, help="Samples per phi (default: 400000).")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument("--scale-exp", type=int, default=10, help="Scale exponent (2^scale_exp).")
    parser.add_argument("--input-bound", type=int, default=1024, help="Clamp bound.")
    parser.add_argument(
        "--bins",
        type=int,
        default=160,
        help="Histogram bins over [-input_bound, input_bound].",
    )
    parser.add_argument(
        "--output",
        default="figures/i32_phi_distribution.png",
        help="Output figure path (.png/.pdf/.svg).",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive window.")
    return parser.parse_args()


def llround_away_from_zero(x: np.ndarray) -> np.ndarray:
    # C++ std::llround semantics: halfway cases round away from zero.
    return np.where(x >= 0.0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int32)


def sample_i32_values(rng: np.random.Generator, phi: float, samples: int, scale_exp: int, input_bound: int) -> np.ndarray:
    u = rng.uniform(-0.5, 0.5, size=samples)
    g = rng.normal(0.0, 1.0, size=samples)
    scale = float(np.ldexp(1.0, scale_exp))
    x = u * np.exp(g * phi) * scale
    x = np.clip(x, -float(input_bound), float(input_bound))
    return llround_away_from_zero(x)


def compute_stats(phi: float, vals: np.ndarray, input_bound: int) -> PhiStats:
    abs_vals = np.abs(vals)
    return PhiStats(
        phi=phi,
        mean=float(np.mean(vals)),
        std=float(np.std(vals)),
        zero_pct=float(np.mean(vals == 0) * 100.0),
        sat_pct=float(np.mean(abs_vals >= input_bound) * 100.0),
        p95_abs=float(np.percentile(abs_vals, 95)),
        p99_abs=float(np.percentile(abs_vals, 99)),
    )


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_plot(
    values_by_phi: List[np.ndarray],
    stats: List[PhiStats],
    phis: List[float],
    input_bound: int,
    bins: int,
    output: str,
    show: bool,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.facecolor": "#FBFCFE",
            "figure.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.9,
            "grid.color": "#D0D6DF",
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.55,
        }
    )

    ncols = min(3, len(phis))
    nrows = int(math.ceil(len(phis) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(phis)))
    edges = np.linspace(-input_bound, input_bound, bins + 1)
    ymax = 0.0

    # Draw per-phi histograms.
    for i, (phi, vals, stat) in enumerate(zip(phis, values_by_phi, stats)):
        ax = axes_flat[i]
        color = cmap[i]
        hist, _ = np.histogram(vals, bins=edges, density=True)
        ymax = max(ymax, float(np.max(hist)))
        ax.hist(vals, bins=edges, density=True, color=color, alpha=0.88, edgecolor="none")
        ax.set_title(
            f"$\\phi={phi:g}$\n"
            f"sat={stat.sat_pct:.1f}%, zero={stat.zero_pct:.1f}%",
            fontweight="bold",
        )
        ax.grid(True)
        ax.set_axisbelow(True)

    # Hide any unused subplot slots.
    for i in range(len(phis), len(axes_flat)):
        axes_flat[i].axis("off")

    # Shared axis labels and limits for histogram panes.
    bottom_row_start = (nrows - 1) * ncols
    for i in range(len(phis)):
        ax = axes_flat[i]
        ax.set_xlim(-input_bound, input_bound)
        ax.set_ylim(0.0, ymax * 1.10 if ymax > 0 else 1.0)
        if i % ncols == 0:
            ax.set_ylabel("Density")
        if i >= bottom_row_start:
            ax.set_xlabel("Sampled int32 value")

    fig.suptitle(
        "Input Value Distribution vs $\\phi$ (i32 benchmark generator)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Sampling rule: U(-0.5,0.5) * exp(N(0,1)*phi) * 2^scale_exp, then clamp + llround",
        ha="center",
        fontsize=9.5,
        color="#333333",
    )
    fig.tight_layout(rect=(0.015, 0.04, 0.995, 0.95))

    ensure_parent(output)
    fig.savefig(output, dpi=300, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    args = parse_args()
    phis = [float(x.strip()) for x in args.phis.split(",") if x.strip()]
    if not phis:
        raise ValueError("empty --phis")
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.input_bound <= 0:
        raise ValueError("--input-bound must be > 0")

    rng = np.random.default_rng(args.seed)
    values_by_phi: List[np.ndarray] = []
    stats: List[PhiStats] = []

    for phi in phis:
        vals = sample_i32_values(
            rng=rng,
            phi=phi,
            samples=args.samples,
            scale_exp=args.scale_exp,
            input_bound=args.input_bound,
        )
        values_by_phi.append(vals)
        stats.append(compute_stats(phi, vals, args.input_bound))

    make_plot(
        values_by_phi=values_by_phi,
        stats=stats,
        phis=phis,
        input_bound=args.input_bound,
        bins=args.bins,
        output=args.output,
        show=args.show,
    )

    print(f"saved: {args.output}")
    print("stats:")
    for st in stats:
        print(
            f"  phi={st.phi:g}: mean={st.mean:.2f}, std={st.std:.2f}, "
            f"zero={st.zero_pct:.2f}%, sat={st.sat_pct:.2f}%, "
            f"|x|p95={st.p95_abs:.1f}, |x|p99={st.p99_abs:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
