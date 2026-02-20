#!/usr/bin/env python3
"""
Generate publication-quality figures for the Ozaki Scheme 2 paper.

Usage:
    python plot_paper_figures.py [--output-dir ./figures] [--show] [--dpi 300] [--show-int8-aux]
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Palette & style
# ---------------------------------------------------------------------------

# Refined palette — muted, print-friendly, distinguishable in greyscale
C = {
    "blue":    "#2D6A9F",
    "orange":  "#E8833A",
    "teal":    "#2A9D8F",
    "red":     "#C44E52",
    "purple":  "#8172B3",
    "gold":    "#CCB974",
    "pink":    "#DA8BC3",
    "grey":    "#8C8C8C",
    "navy":    "#1B3A5C",
    "lime":    "#7CBE42",
}

# Stacked-bar palettes (soft, high-contrast neighbours)
BAR_PAL4 = ["#4C9BE8", "#F4A261", "#2A9D8F", "#E76F51"]
BAR_PAL5 = ["#4C9BE8", "#F4A261", "#2A9D8F", "#E76F51", "#8172B3"]

# Sequential tints for moduli sweep (blue → warm)
SEQ6 = ["#2D6A9F", "#2A9D8F", "#7CBE42", "#E8833A", "#C44E52", "#8172B3"]


def apply_style(plt) -> None:
    plt.rcParams.update({
        # Font
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset":  "dejavusans",
        # Sizes
        "font.size":         9.5,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   8.5,
        "ytick.labelsize":   8.5,
        "legend.fontsize":   8,
        # Frame
        "axes.linewidth":    0.7,
        "axes.edgecolor":    "#333333",
        "axes.facecolor":    "#FAFAFA",
        "figure.facecolor":  "white",
        # Grid
        "axes.grid":         False,
        "grid.color":        "#CCCCCC",
        "grid.linewidth":    0.4,
        "grid.alpha":        0.6,
        # Lines / markers
        "lines.linewidth":   1.8,
        "lines.markersize":  5.5,
        # Legend
        "legend.framealpha": 0.92,
        "legend.edgecolor":  "#CCCCCC",
        "legend.fancybox":   True,
        "legend.borderpad":  0.4,
        # Save
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.08,
        "figure.dpi":        150,
    })


def _clean_ax(ax, grid_axis="y"):
    """Remove top/right spines, add subtle grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linewidth=0.4, alpha=0.55, color="#CCCCCC")


def _fmt_n(n: int) -> str:
    """Pretty-print dimension: 16384 → '16K'."""
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        hdr_raw = next(reader, None)
        if hdr_raw is None:
            return [], []
        header = [h.strip() for h in hdr_raw if h.strip()]
        rows: List[Dict[str, str]] = []
        for raw in reader:
            if not any(c.strip() for c in raw):
                continue
            if len(raw) < len(header):
                raw = raw + [""] * (len(header) - len(raw))
            rows.append({header[i]: raw[i].strip() for i in range(len(header))})
    return header, rows


def pf(text: str) -> Optional[float]:
    if not text or not text.strip():
        return None
    try:
        return float(text.strip())
    except ValueError:
        return None


def pi(text: str) -> Optional[int]:
    v = pf(text)
    return None if v is None else int(v)


def find_csvs(parent: Path, pattern: str) -> List[Path]:
    return sorted(parent.glob(pattern))


# ===================================================================
#  FP64 — Figure 1: Time breakdown (stacked bars)
# ===================================================================

FP64_STAGES = ["conv_64f_2_8i", "cublasGemmEx", "conv_32i_2_8u", "inverse_scaling"]
FP64_SLABELS = ["Quantization", "INT8 GEMM (TC)", "Requantization", "Inverse scaling"]


def plot_fp64_timebreakdown(plt, csv_path, out_dir, dpi, show):
    _, rows = load_csv(csv_path)
    n_vals = list(dict.fromkeys(pi(r.get("n", "")) for r in rows if pi(r.get("n", "")) is not None))
    moduli = list(range(2, 21))

    ncols = len(n_vals)
    fig, axes = plt.subplots(2, ncols, figsize=(2.4 * ncols + 0.6, 4.4), squeeze=False)

    for ci, n in enumerate(n_vals):
        for ri, pfx in enumerate(["OS2-fast", "OS2-accu"]):
            ax = axes[ri][ci]
            pcts, vm = [], []
            for m in moduli:
                row = next((r for r in rows if pi(r.get("n", "")) == n
                            and r.get("function", "") == f"{pfx}-{m}"), None)
                if row is None:
                    continue
                vals = [pf(row.get(c, "")) or 0.0 for c in FP64_STAGES]
                s = sum(vals)
                if s > 0:
                    pcts.append([100 * v / s for v in vals])
                    vm.append(m)
            if not pcts:
                continue

            x = range(len(vm))
            bot = [0.0] * len(vm)
            for si in range(4):
                h = [pcts[i][si] for i in range(len(pcts))]
                ax.bar(x, h, bottom=bot, color=BAR_PAL4[si], width=0.78,
                       edgecolor="white", linewidth=0.35,
                       label=FP64_SLABELS[si] if (ci == 0 and ri == 0) else None)
                bot = [b + hi for b, hi in zip(bot, h)]

            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_yticklabels(["0", "", "50", "", "100"])
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(m) if m % 3 == 2 or m == 2 else "" for m in vm], fontsize=7)
            _clean_ax(ax, "y")
            if ri == 0:
                ax.set_title(f"$n = {n}$", fontweight="bold", pad=6)
            if ci == 0:
                ax.set_ylabel(f"{pfx}\nTime share (%)", fontsize=8.5, labelpad=4)
            else:
                ax.set_yticklabels([])

    # Legend
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.015),
               fontsize=8, handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
    fig.supxlabel("Number of moduli", fontsize=10, y=-0.01)
    fig.tight_layout(rect=(0, 0.01, 1, 0.94), h_pad=1.0, w_pad=0.6)

    out = out_dir / "fig_fp64_timebreakdown.png"
    fig.savefig(out, dpi=dpi)
    if not show:
        plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ===================================================================
#  FP64 — Figure 2: Accuracy vs moduli
# ===================================================================

def plot_fp64_accuracy(plt, csv_path, out_dir, dpi, show):
    import matplotlib.ticker as mt
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    header, rows = load_csv(csv_path)
    mcols = [int(h) for h in header if h.isdigit()]
    if not mcols:
        raise ValueError("No moduli columns")

    # ── Parse data ──
    phi_vals: List[float] = []
    data: Dict[Tuple[float, str, int], List[float]] = {}
    for row in rows:
        phi = pf(row.get("phi", ""))
        if phi is None:
            continue
        if phi not in phi_vals:
            phi_vals.append(phi)
        func = row.get("function", "")
        m = re.search(r"\(k\s*=\s*(\d+)\)", func)
        if not m:
            continue
        k = int(m.group(1))
        base = re.sub(r"\s*\(k\s*=\s*\d+\)", "", func).strip()
        data[(phi, base, k)] = [math.nan if (v := pf(row.get(str(c), ""))) is None else v
                                 for c in mcols]

    all_k = sorted(set(k for _, _, k in data))
    k_plot = [all_k[0], all_k[-1]] if len(all_k) > 1 else all_k

    # Separate: OS1/DGEMM are flat reference lines, OS2 are the main curves
    ref_bases = [b for b in ["DGEMM", "OS1-7", "OS1-8", "OS1-9", "OS1-10"]
                 if any(bb == b for _, bb, _ in data)]
    os2_bases = [b for b in ["OS2-fast", "OS2-accu"]
                 if any(bb == b for _, bb, _ in data)]

    # Colors: references get muted grays/pastels, OS2 gets vivid colors
    ref_colors = {"DGEMM": "#AAAAAA", "OS1-7": "#78C2A4", "OS1-8": "#E8C170",
                  "OS1-9": "#E8936A", "OS1-10": "#C98DBF"}
    os2_colors = {"OS2-fast": C["blue"], "OS2-accu": C["red"]}
    os2_markers = {"OS2-fast": "o", "OS2-accu": "s"}

    # ── Layout: 1 row, up to 5 cols ──
    nphi = len(phi_vals)
    fig, axes = plt.subplots(1, nphi, figsize=(3.0 * nphi + 0.8, 4.2), squeeze=False)
    axes_1d = axes[0]

    for idx, phi in enumerate(phi_vals):
        ax = axes_1d[idx]
        _clean_ax(ax, "both")

        # --- Draw reference baselines as horizontal bands (they're constant) ---
        for b in ref_bases:
            # Collect all y-values for this base at this phi (across k)
            # They're constant across moduli, so just get the representative value
            for ki, k in enumerate(k_plot):
                key = (phi, b, k)
                if key not in data:
                    continue
                y = data[key]
                yvals = [yi for yi in y if not math.isnan(yi) and yi > 0]
                if not yvals:
                    continue
                yval = yvals[0]  # constant across moduli
                ls = "-" if ki == 0 else ":"
                lw = 1.0 if ki == 0 else 0.7
                ax.axhline(yval, color=ref_colors.get(b, "#CCCCCC"), ls=ls, lw=lw,
                           alpha=0.65, zorder=1)
                # Label on right edge (first k only)
                if ki == 0 and idx == nphi - 1:
                    ax.annotate(b, xy=(mcols[-1] + 0.3, yval),
                                fontsize=5.5, color=ref_colors.get(b, "#999999"),
                                va="center", ha="left", fontweight="bold", alpha=0.8,
                                annotation_clip=False)

        # --- Draw OS2 curves prominently ---
        for b in os2_bases:
            col = os2_colors[b]
            mk = os2_markers[b]
            for ki, k in enumerate(k_plot):
                key = (phi, b, k)
                if key not in data:
                    continue
                y = data[key]
                xp = [xi for xi, yi in zip(mcols, y) if not math.isnan(yi) and yi > 0]
                yp = [yi for yi in y if not math.isnan(yi) and yi > 0]
                if not xp:
                    continue
                ls = "-" if ki == 0 else "--"
                lw = 2.0 if ki == 0 else 1.3
                alpha = 0.95 if ki == 0 else 0.6
                msz = 5 if ki == 0 else 3.5
                ax.plot(xp, yp, linestyle=ls, color=col, marker=mk,
                        markersize=msz, markeredgecolor="white", markeredgewidth=0.4,
                        linewidth=lw, alpha=alpha, zorder=3)

        ax.set_yscale("log")
        ax.set_title(f"$\\phi = {phi:g}$", fontweight="bold", fontsize=11, pad=8)
        ax.set_xlim(mcols[0] - 0.5, mcols[-1] + 0.5)
        ax.set_xticks([2, 5, 8, 11, 14, 17, 20])
        ax.yaxis.set_major_locator(mt.LogLocator(base=10, numticks=10))
        ax.yaxis.set_minor_locator(mt.LogLocator(base=10, subs=(2, 5), numticks=20))
        ax.yaxis.set_minor_formatter(mt.NullFormatter())
        ax.set_xlabel("Number of moduli", fontsize=9)

        if idx == 0:
            ax.set_ylabel("Max relative error", fontsize=10)

    # ── Custom legend ──
    legend_elements = []
    # OS2 entries (prominent)
    for b in os2_bases:
        legend_elements.append(
            Line2D([0], [0], color=os2_colors[b], marker=os2_markers[b],
                   markersize=6, markeredgecolor="white", markeredgewidth=0.4,
                   linewidth=2.0, label=b))
    # Reference entries (subtle)
    for b in ref_bases:
        legend_elements.append(
            Line2D([0], [0], color=ref_colors.get(b, "#CCC"), linewidth=1.2,
                   alpha=0.7, label=b))
    # Line style legend
    legend_elements.append(
        Line2D([0], [0], color="#666666", linewidth=1.5, linestyle="-",
               label=f"$k = {k_plot[0]}$"))
    if len(k_plot) > 1:
        legend_elements.append(
            Line2D([0], [0], color="#666666", linewidth=1.0, linestyle="--",
                   label=f"$k = {k_plot[-1]}$"))

    fig.legend(handles=legend_elements, loc="upper center",
               ncol=min(len(legend_elements), 5),
               bbox_to_anchor=(0.5, 1.02),
               fontsize=8.5, handlelength=2.0, handletextpad=0.5,
               columnspacing=1.8, framealpha=0.95, edgecolor="#CCCCCC")

    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.0)

    out = out_dir / "fig_fp64_accuracy.png"
    fig.savefig(out, dpi=dpi)
    if not show:
        plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ===================================================================
#  FP64 — Figure 3: Speedup vs cuBLAS DGEMM
# ===================================================================

def plot_fp64_speedup(plt, csv_path, out_dir, dpi, show):
    _, rows = load_csv(csv_path)
    n_vals = list(dict.fromkeys(pi(r.get("n", "")) for r in rows if pi(r.get("n", "")) is not None))
    perf: Dict[int, Dict[str, float]] = {}
    for r in rows:
        n, func, t = pi(r.get("n", "")), r.get("function", ""), pf(r.get("TFLOPS", ""))
        if n is not None and t is not None:
            perf.setdefault(n, {})[func] = t

    msel = [8, 10, 12, 14, 16, 18]
    x = list(range(len(n_vals)))
    xl = [_fmt_n(n) for n in n_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={"wspace": 0.32})

    # ── Left: TFLOPS ──
    dgemm = [perf.get(n, {}).get("DGEMM", 0) for n in n_vals]
    int8  = [perf.get(n, {}).get("INT8-GEMM", 0) for n in n_vals]
    ax1.plot(x, int8, color=C["grey"], ls="--", marker="^", markersize=4, lw=1.0,
             label="INT8-GEMM (peak)", zorder=2)
    ax1.plot(x, dgemm, color=C["navy"], ls="-", marker="s", markersize=5, lw=2.0,
             label="cuBLAS DGEMM", zorder=3)
    for mi, m in enumerate(msel):
        vals = [perf.get(n, {}).get(f"OS2-fast-{m}", 0) for n in n_vals]
        ax1.plot(x, vals, color=SEQ6[mi], marker="o", markersize=4, lw=1.4,
                 label=f"OS2-fast-{m}", zorder=2)

    ax1.set_xticks(x); ax1.set_xticklabels(xl)
    ax1.set_xlabel("Matrix dimension ($n$)")
    ax1.set_ylabel("TFLOPS")
    ax1.set_yscale("log")
    _clean_ax(ax1, "both")
    ax1.legend(fontsize=7, loc="lower right", ncol=1, framealpha=0.9)
    ax1.set_title("Throughput", fontweight="bold", pad=8)

    # ── Right: Speedup ──
    for mi, m in enumerate(msel):
        sp = []
        for n in n_vals:
            d = perf.get(n, {}).get("DGEMM")
            f = perf.get(n, {}).get(f"OS2-fast-{m}")
            sp.append(f / d if d and f and d > 0 else 0)
        line, = ax2.plot(x, sp, color=SEQ6[mi], marker="o", markersize=4, lw=1.4,
                         label=f"OS2-fast-{m}")
        # annotate last point
        if sp[-1] > 0:
            ax2.annotate(f"{sp[-1]:.1f}x", xy=(x[-1], sp[-1]),
                         xytext=(6, 0), textcoords="offset points",
                         fontsize=6.5, color=SEQ6[mi], fontweight="bold", va="center")

    ax2.axhline(1.0, color="#999999", ls=":", lw=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels(xl)
    ax2.set_xlabel("Matrix dimension ($n$)")
    ax2.set_ylabel("Speedup vs cuBLAS DGEMM")
    _clean_ax(ax2, "both")
    ax2.legend(fontsize=7, loc="upper left", ncol=1)
    ax2.set_title("Speedup over DGEMM", fontweight="bold", pad=8)

    fig.tight_layout()
    out = out_dir / "fig_fp64_speedup.png"
    fig.savefig(out, dpi=dpi)
    if not show:
        plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ===================================================================
#  FP64 — Figure 4: Accuracy-Performance Pareto
# ===================================================================

def plot_fp64_accuracy_vs_tflops(plt, csv_path, out_dir, dpi, show):
    _, rows = load_csv(csv_path)
    n_vals = list(dict.fromkeys(pi(r.get("n", "")) for r in rows if pi(r.get("n", "")) is not None))
    n_t = n_vals[-1] if n_vals else 8192

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    for pfx, mk, col, fc in [("OS2-fast", "o", C["blue"], "#D0E4F5"),
                               ("OS2-accu", "s", C["red"],  "#F5D0D3")]:
        xs, ys, lbls = [], [], []
        for r in rows:
            if pi(r.get("n", "")) != n_t or not r.get("function", "").startswith(pfx + "-"):
                continue
            t, e = pf(r.get("TFLOPS", "")), pf(r.get("relerr_max", ""))
            if t is None or e is None or e <= 0:
                continue
            xs.append(t); ys.append(e); lbls.append(r["function"].split("-")[-1])

        ax.plot(xs, ys, color=col, lw=1.2, alpha=0.4, zorder=1)
        ax.scatter(xs, ys, marker=mk, c=col, s=32, edgecolors="white", linewidths=0.5,
                   label=pfx, zorder=3)
        for xi, yi, lb in zip(xs, ys, lbls):
            m = int(lb)
            if m % 4 == 0 or m in [2, 3, 18, 20]:
                ax.annotate(lb, (xi, yi), fontsize=6, fontweight="bold", color=col,
                            xytext=(4, 4), textcoords="offset points", zorder=4)

    # DGEMM star
    dr = next((r for r in rows if pi(r.get("n", "")) == n_t
               and r.get("function", "") == "DGEMM"), None)
    if dr:
        dt, de = pf(dr.get("TFLOPS", "")), pf(dr.get("relerr_max", ""))
        if dt and de and de > 0:
            ax.scatter([dt], [de], marker="*", s=120, c=C["navy"], edgecolors="white",
                       linewidths=0.5, zorder=5, label="cuBLAS DGEMM")

    ax.set_yscale("log")
    ax.set_xlabel("TFLOPS"); ax.set_ylabel("Max relative error")
    ax.set_title(f"Accuracy vs Throughput  ($n = {_fmt_n(n_t)}$)", fontweight="bold", pad=8)
    _clean_ax(ax, "both")
    ax.legend(fontsize=8, loc="upper right", markerscale=1.2)
    fig.tight_layout()

    out = out_dir / "fig_fp64_accuracy_vs_tflops.png"
    fig.savefig(out, dpi=dpi)
    if not show:
        plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ===================================================================
#  INT32 — Figure 1: Speedup vs baselines
# ===================================================================

def plot_i32_speedup(plt, csv_path, out_dir, dpi, show, show_int8_aux=False):
    from matplotlib.patches import FancyBboxPatch
    _, rows = load_csv(csv_path)
    filt = [r for r in rows if r.get("opA") == "N" and r.get("opB") == "N"]
    extra   = [r for r in filt if r.get("use_extra") == "1"]
    compact = [r for r in filt if r.get("use_extra") == "0"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8), gridspec_kw={"wspace": 0.30})

    for ax, drows, title in [(ax1, extra, "Extra workspace"),
                              (ax2, compact, "Compact (in-place)")]:
        nv = sorted(set(pi(r.get("n", "")) for r in drows if pi(r.get("n", "")) is not None))
        su_i32, su_cub = [], []
        for n in nv:
            r = next((r for r in drows if pi(r.get("n", "")) == n), None)
            su_i32.append(pf(r.get("speedup_vs_exact_i32_i32_i64", "")) or 0 if r else 0)
            if show_int8_aux:
                su_cub.append(pf(r.get("speedup_vs_cublas_i8_single", "")) or 0 if r else 0)

        x = range(len(nv))
        if show_int8_aux:
            w = 0.32
            b1 = ax.bar([xi - w / 2 for xi in x], su_i32, w, color=C["blue"], label="vs Exact i32",
                        edgecolor="white", linewidth=0.4, zorder=2)
            b2 = ax.bar([xi + w / 2 for xi in x], su_cub, w, color=C["orange"], label="vs cuBLAS i8",
                        edgecolor="white", linewidth=0.4, zorder=2)
            bar_sets = [b1, b2]
        else:
            b1 = ax.bar(list(x), su_i32, 0.56, color=C["blue"], label="vs Exact i32",
                        edgecolor="white", linewidth=0.4, zorder=2)
            bar_sets = [b1]
        # Value labels on top
        for bar_set in bar_sets:
            for bar in bar_set:
                h = bar.get_height()
                if h > 0.15:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.12,
                            f"{h:.1f}", ha="center", va="bottom", fontsize=6, fontweight="bold",
                            color="#555555")

        ax.axhline(1.0, color="#AAAAAA", ls="--", lw=0.7, zorder=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels([_fmt_n(n) for n in nv])
        ax.set_xlabel("Matrix dimension ($n$)")
        ax.set_ylabel("Speedup")
        ax.set_title(title, fontweight="bold", pad=8)
        _clean_ax(ax, "y")
        ax.legend(fontsize=7.5, loc="upper left")

    fig.tight_layout()
    out = out_dir / "fig_i32_speedup.png"
    fig.savefig(out, dpi=dpi)
    if not show:
        plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ===================================================================
#  INT32 — Figure 2: Time breakdown (stacked)
# ===================================================================

EXTRA_COLS   = ["gemmul8_encode_ms", "gemmul8_tc_ms", "gemmul8_reconstruct_ms"]
EXTRA_LBLS   = ["Encode", "TC GEMM", "Reconstruct"]
COMPACT_COLS = ["gemmul8_encode_ms", "gemmul8_tc_ms", "gemmul8_conv32to8_ms", "gemmul8_reconstruct_ms"]
COMPACT_LBLS = ["Encode", "TC GEMM", u"Conv 32\u21928", "Reconstruct"]


def plot_i32_timebreakdown(plt, csv_path, out_dir, dpi, show):
    _, rows = load_csv(csv_path)
    filt = [r for r in rows if r.get("opA") == "N" and r.get("opB") == "N"]
    extra   = [r for r in filt if r.get("use_extra") == "1"]
    compact = [r for r in filt if r.get("use_extra") == "0"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), gridspec_kw={"wspace": 0.22})

    for ax, drows, scols, slbls, title in [
        (axes[0], extra,   EXTRA_COLS,   EXTRA_LBLS,   "Extra workspace"),
        (axes[1], compact, COMPACT_COLS, COMPACT_LBLS, "Compact (in-place)"),
    ]:
        nv = sorted(set(pi(r.get("n", "")) for r in drows if pi(r.get("n", "")) is not None))
        x = range(len(nv))
        bot = [0.0] * len(nv)
        for si, col in enumerate(scols):
            h = []
            for n in nv:
                r = next((r for r in drows if pi(r.get("n", "")) == n), None)
                tot = pf(r.get("gemmul8_total_ms", "")) if r else None
                val = pf(r.get(col, "")) if r else None
                h.append(100 * val / tot if tot and val and tot > 0 else 0)
            ax.bar(x, h, bottom=bot, color=BAR_PAL4[si], width=0.65,
                   edgecolor="white", linewidth=0.4,
                   label=slbls[si] if ax is axes[0] else None)
            # Percentage labels inside bars
            for xi, hi, bi in zip(x, h, bot):
                if hi > 8:
                    ax.text(xi, bi + hi / 2, f"{hi:.0f}%", ha="center", va="center",
                            fontsize=6, color="white", fontweight="bold")
            bot = [b + hi for b, hi in zip(bot, h)]

        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(list(x))
        ax.set_xticklabels([_fmt_n(n) for n in nv])
        ax.set_xlabel("Matrix dimension ($n$)")
        ax.set_title(title, fontweight="bold", pad=8)
        _clean_ax(ax, "y")

    axes[0].set_ylabel("Time breakdown (%)")
    axes[1].set_yticklabels([])

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01),
               fontsize=8, handlelength=1.2, handletextpad=0.4, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = out_dir / "fig_i32_timebreakdown.png"
    fig.savefig(out, dpi=dpi)
    if not show:
        plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ===================================================================
#  INT32 — Figure 3: GFLOPS comparison
# ===================================================================

def plot_i32_gflops(plt, csv_path, out_dir, dpi, show, show_int8_aux=False):
    _, rows = load_csv(csv_path)
    filt = [r for r in rows if r.get("opA") == "N" and r.get("opB") == "N"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={"wspace": 0.28})

    configs = [(ax1, "1", "Extra workspace"), (ax2, "0", "Compact (in-place)")]
    for ax, ue, title in configs:
        dr = [r for r in filt if r.get("use_extra") == ue]
        nv = sorted(set(pi(r.get("n", "")) for r in dr if pi(r.get("n", "")) is not None))
        gf_g, gf_e, gf_c = [], [], []
        for n in nv:
            r = next((r for r in dr if pi(r.get("n", "")) == n), None)
            gf_g.append(pf(r.get("gemmul8_gflops", "")) or 0 if r else 0)
            gf_e.append(pf(r.get("exact_i32_i32_i64_gflops", "")) or 0 if r else 0)
            if show_int8_aux:
                gf_c.append(pf(r.get("cublas_i8_single_gflops", "")) or 0 if r else 0)
        x = range(len(nv))
        xl = [_fmt_n(n) for n in nv]

        if show_int8_aux:
            ax.plot(x, gf_c, color=C["teal"], marker="^", ls=":", lw=1.6, markersize=5,
                    label="cuBLAS i8 (single)", markeredgecolor="white", markeredgewidth=0.4)
        ax.plot(x, gf_e, color=C["orange"], marker="s", ls="--", lw=1.6, markersize=5,
                label=u"Exact i32\u00d7i32\u2192i64", markeredgecolor="white", markeredgewidth=0.4)
        ax.plot(x, gf_g, color=C["blue"], marker="o", ls="-", lw=2.0, markersize=6,
                label="GEMMul8 (ours)", markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        # fill between ours and exact
        ax.fill_between(list(x), gf_e, gf_g, alpha=0.08, color=C["blue"])

        ax.set_xticks(list(x)); ax.set_xticklabels(xl)
        ax.set_xlabel("Matrix dimension ($n$)")
        ax.set_ylabel("GFLOPS")
        ax.set_yscale("log")
        ax.set_title(title, fontweight="bold", pad=8)
        _clean_ax(ax, "both")
        ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    out = out_dir / "fig_i32_gflops.png"
    fig.savefig(out, dpi=dpi)
    if not show:
        plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ===================================================================
#  INT32 (OZ2) — Figure: Speedup
# ===================================================================

def plot_i32_oz2_speedup(plt, csv_path, out_dir, dpi, show):
    _, rows = load_csv(csv_path)
    n_vals = list(dict.fromkeys(pi(r.get("n", "")) for r in rows if pi(r.get("n", "")) is not None))
    x = range(len(n_vals))
    xl = [_fmt_n(n) for n in n_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={"wspace": 0.28})

    variants = [("I32-accu-9-extra",   C["blue"],   "o", "OZ2 (extra)"),
                ("I32-accu-9-compact", C["orange"], "s", "OZ2 (compact)")]

    for var, col, mk, lbl in variants:
        tfl, sp = [], []
        for n in n_vals:
            er = next((r for r in rows if pi(r.get("n", "")) == n
                       and r.get("function", "") == "I32-exact-NN"), None)
            vr = next((r for r in rows if pi(r.get("n", "")) == n
                       and r.get("function", "") == var), None)
            et = pf(er.get("TFLOPS", "")) if er else None
            vt = pf(vr.get("TFLOPS", "")) if vr else None
            tfl.append(vt or 0)
            sp.append(vt / et if et and vt and et > 0 else 0)
        ax1.plot(list(x), tfl, f"{mk}-", color=col, markersize=5, lw=1.8,
                 markeredgecolor="white", markeredgewidth=0.4, label=lbl)
        ax2.plot(list(x), sp, f"{mk}-", color=col, markersize=5, lw=1.8,
                 markeredgecolor="white", markeredgewidth=0.4, label=lbl)
        # annotate last speedup
        if sp[-1] > 0:
            ax2.annotate(f"{sp[-1]:.1f}x", xy=(list(x)[-1], sp[-1]),
                         xytext=(5, 2), textcoords="offset points",
                         fontsize=7, fontweight="bold", color=col)

    # Exact baseline on throughput
    exact_t = [pf(next((r for r in rows if pi(r.get("n", "")) == n
                        and r.get("function", "") == "I32-exact-NN"), {}).get("TFLOPS", "")) or 0
               for n in n_vals]
    ax1.plot(list(x), exact_t, color=C["grey"], ls="--", marker="^", markersize=4,
             lw=1.4, label=u"Exact i32\u00d7i32\u2192i64", zorder=1)
    ax1.fill_between(list(x), exact_t, [0]*len(x), alpha=0.06, color=C["grey"])

    for ax in (ax1, ax2):
        ax.set_xticks(list(x)); ax.set_xticklabels(xl)
        ax.set_xlabel("Matrix dimension ($n$)")
        _clean_ax(ax, "both")

    ax1.set_ylabel("TFLOPS"); ax1.set_title("Throughput", fontweight="bold", pad=8)
    ax1.legend(fontsize=7.5, loc="upper left")
    ax2.axhline(1.0, color="#AAAAAA", ls="--", lw=0.7)
    ax2.set_ylabel("Speedup vs exact i32")
    ax2.set_title("Speedup", fontweight="bold", pad=8)
    ax2.legend(fontsize=7.5, loc="upper left")

    fig.tight_layout()
    out = out_dir / "fig_i32_oz2_speedup.png"
    fig.savefig(out, dpi=dpi)
    if not show:
        plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ===================================================================
#  INT32 (OZ2) — Figure: Time breakdown
# ===================================================================

OZ2_I32_COLS = ["quantization", "low_prec_gemm", "requantization", "dequantization"]
OZ2_I32_LBLS = ["Quantization", "Low-prec GEMM", "Requantization", "Dequantization"]


def plot_i32_oz2_timebreakdown(plt, csv_path, out_dir, dpi, show):
    _, rows = load_csv(csv_path)
    variants = [("I32-accu-9-extra", "Extra workspace"),
                ("I32-accu-9-compact", "Compact (in-place)")]
    n_vals = list(dict.fromkeys(pi(r.get("n", "")) for r in rows if pi(r.get("n", "")) is not None))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True, gridspec_kw={"wspace": 0.10})

    for ax, (var, title) in zip(axes, variants):
        x = range(len(n_vals))
        bot = [0.0] * len(n_vals)
        for si, (col, lbl) in enumerate(zip(OZ2_I32_COLS, OZ2_I32_LBLS)):
            h = []
            for n in n_vals:
                r = next((r for r in rows if pi(r.get("n", "")) == n
                          and r.get("function", "") == var), None)
                tot = pf(r.get("total_time[sec]", "")) if r else None
                val = pf(r.get(col, "")) if r else None
                h.append(100 * val / tot if tot and val and tot > 0 else 0)
            ax.bar(x, h, bottom=bot, color=BAR_PAL4[si], width=0.62,
                   edgecolor="white", linewidth=0.4,
                   label=lbl if ax is axes[0] else None)
            for xi, hi, bi in zip(x, h, bot):
                if hi > 8:
                    ax.text(xi, bi + hi / 2, f"{hi:.0f}%", ha="center", va="center",
                            fontsize=6, color="white", fontweight="bold")
            bot = [b + hi for b, hi in zip(bot, h)]

        ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(list(x))
        ax.set_xticklabels([_fmt_n(n) for n in n_vals])
        ax.set_xlabel("Matrix dimension ($n$)")
        ax.set_title(title, fontweight="bold", pad=8)
        _clean_ax(ax, "y")

    axes[0].set_ylabel("Time breakdown (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01),
               fontsize=8, handlelength=1.2, handletextpad=0.4, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = out_dir / "fig_i32_oz2_timebreakdown.png"
    fig.savefig(out, dpi=dpi)
    if not show:
        plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ===================================================================
#  Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="./figures")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show", action="store_true")
    p.add_argument("--show-int8-aux", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg" if not args.show else "TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required", file=sys.stderr)
        return 1

    apply_style(plt)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent

    gen, err = [], []
    print("Generating figures …")

    # FP64
    for csv_path in [p for p in find_csvs(data_dir, "oz2_results_d_time_NVIDIA_*.csv")
                     if "ozaki1" not in p.name and "rect" not in p.name]:
        for fn in [plot_fp64_timebreakdown, plot_fp64_speedup, plot_fp64_accuracy_vs_tflops]:
            try:
                gen.append(fn(plt, csv_path, out_dir, args.dpi, args.show))
            except Exception as e:
                err.append(f"{fn.__name__}: {e}")

    for csv_path in find_csvs(data_dir, "oz2_results_d_accuracy_NVIDIA_*.csv"):
        try:
            gen.append(plot_fp64_accuracy(plt, csv_path, out_dir, args.dpi, args.show))
        except Exception as e:
            err.append(f"fp64_accuracy: {e}")

    # INT32
    csvs = find_csvs(data_dir, "i32_bench_speedup_NVIDIA_*.csv")
    if csvs:
        cp = csvs[-1]
        try:
            gen.append(plot_i32_speedup(plt, cp, out_dir, args.dpi, args.show, args.show_int8_aux))
        except Exception as e:
            err.append(f"{plot_i32_speedup.__name__}: {e}")
        try:
            gen.append(plot_i32_timebreakdown(plt, cp, out_dir, args.dpi, args.show))
        except Exception as e:
            err.append(f"{plot_i32_timebreakdown.__name__}: {e}")
        try:
            gen.append(plot_i32_gflops(plt, cp, out_dir, args.dpi, args.show, args.show_int8_aux))
        except Exception as e:
            err.append(f"{plot_i32_gflops.__name__}: {e}")

    for csv_path in find_csvs(data_dir, "oz2_results_i32_time_NVIDIA_*.csv"):
        for fn in [plot_i32_oz2_speedup, plot_i32_oz2_timebreakdown]:
            try:
                gen.append(fn(plt, csv_path, out_dir, args.dpi, args.show))
            except Exception as e:
                err.append(f"{fn.__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"  {len(gen)} figures generated in {out_dir}/")
    for p in gen:
        print(f"    {p.name}")
    if err:
        print(f"\n  {len(err)} error(s):")
        for e in err:
            print(f"    {e}", file=sys.stderr)
    if args.show:
        plt.show()
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
