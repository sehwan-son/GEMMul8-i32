#!/usr/bin/env python3
"""Python port of plot_flops.m for oz2_results_{f,d}_time CSV files."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from plot_oz2_common import (
    PRECISION_CONFIG,
    apply_paper_style,
    choose_y_tick_step,
    collect_moduli_series,
    env_label_from_filename,
    find_first_value,
    legend_ncol,
    load_csv_rows,
    parse_float,
    parse_int,
    replace_stem_token,
    robust_linear_upper,
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
        description="Plot TFLOPS curves from oz2_results_{f,d}_time_NVIDIA_*.csv."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Input CSV files. If omitted, auto-loads ../oz2_results_<precision>_time_NVIDIA_*.csv.",
    )
    parser.add_argument("--precision", choices=["d", "f"], default="d")
    parser.add_argument(
        "--glob",
        default="",
        help="Override glob pattern searched in ../ (relative to this script).",
    )
    parser.add_argument(
        "--accuracy-file",
        default="",
        help="Optional paired accuracy CSV. Default: auto-resolve by replacing _time_ with _accuracy_.",
    )
    parser.add_argument("--output-dir", default=".", help="Directory to save images.")
    parser.add_argument("--ylim-mode", choices=["robust", "full"], default="robust")
    parser.add_argument("--q-high", type=float, default=0.95, help="Upper quantile for robust y-limit.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Display plot windows.")
    return parser.parse_args()


def _baseline_defs(precision: str) -> List[Tuple[str, str, bool, int, int]]:
    if precision == "d":
        return [
            ("DGEMM", "DGEMM", False, 1, 1),
            ("ozIMMU_EF-8", "ozIMMU_EF-8", True, 1, 2),
            ("ozIMMU_EF-9", "ozIMMU_EF-9", True, 1, 5),
        ]
    return [
        ("SGEMM", "SGEMM", False, 1, 1),
        ("TF32GEMM", "SGEMM-TF32", False, 1, 2),
        ("BF16x9", "SGEMM-BF16X9", False, 1, 5),
        ("cuMpSGEMM", "FP16TCEC_SCALING", False, 1, 6),
    ]


def _series_max(values: Sequence[float]) -> float | None:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return None
    return max(finite)


def _paired_accuracy_file(time_file: Path, arg_accuracy_file: str) -> Path | None:
    if arg_accuracy_file:
        p = Path(arg_accuracy_file).expanduser().resolve()
        return p if p.exists() else None

    name = time_file.name
    if "_time_" not in name:
        return None
    candidate = time_file.with_name(name.replace("_time_", "_accuracy_", 1))
    if candidate.exists():
        return candidate
    return None


def _resolve_precision_moduli(
    accuracy_file: Path | None,
    time_rows: Sequence[Dict[str, str]],
    n_list: Sequence[int],
    baseline_name: str,
) -> Tuple[Dict[int, Dict[str, int | None]], float | None]:
    if accuracy_file is None:
        return {}, None

    header, acc_rows = load_csv_rows(accuracy_file)
    moduli = [int(h) for h in header if h.isdigit()]
    if not moduli:
        return {}, None

    rec: Dict[Tuple[float, int, str], List[float | None]] = {}
    phi_list: List[float] = []
    for row in acc_rows:
        phi = parse_float(row.get("phi", ""))
        if phi is None:
            continue
        base, k = split_function_and_k(row.get("function", ""))
        if k is None:
            continue
        vals = [parse_float(row.get(str(m), "")) for m in moduli]
        rec[(phi, k, base)] = vals
        phi_list.append(phi)

    if not rec:
        return {}, None

    time_phi = next((parse_float(r.get("phi", "")) for r in time_rows if parse_float(r.get("phi", "")) is not None), None)
    uniq_phi = sorted(set(phi_list))
    if not uniq_phi:
        return {}, None
    if time_phi is None:
        phi_ref = uniq_phi[0]
    else:
        phi_ref = min(uniq_phi, key=lambda p: abs(p - time_phi))

    out: Dict[int, Dict[str, int | None]] = {}
    for n in n_list:
        baseline = rec.get((phi_ref, n, baseline_name))
        if baseline is None:
            continue
        out[n] = {}
        for mode in ("OS2-fast", "OS2-accu"):
            target = rec.get((phi_ref, n, mode))
            moduli_hit: int | None = None
            if target is not None:
                for m, t_err, b_err in zip(moduli, target, baseline):
                    if t_err is None or b_err is None:
                        continue
                    if t_err <= b_err:
                        moduli_hit = m
                        break
            out[n][mode] = moduli_hit

    return out, phi_ref


def _plot_one_file(plt, path: Path, args: argparse.Namespace) -> Tuple[Path, List[str]]:
    cfg = PRECISION_CONFIG[args.precision]
    _, rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in CSV: {path}")

    n_list = unique_stable(
        n for row in rows if (n := parse_int(row.get("n", ""))) is not None
    )
    if not n_list:
        raise ValueError(f"Missing valid n column values: {path}")

    xx = list(range(cfg.moduli_min, cfg.moduli_max + 1))
    ncols = len(n_list)
    fig, axes = plt.subplots(1, ncols, figsize=(1.95 * ncols + 1.7, 3.3), squeeze=False)
    axes_1d = axes[0]

    baselines = _baseline_defs(args.precision)
    ref_label, ref_token, ref_contains, _, _ = baselines[0]
    accuracy_file = _paired_accuracy_file(path, args.accuracy_file)
    precision_moduli, phi_ref = _resolve_precision_moduli(
        accuracy_file,
        rows,
        n_list,
        baseline_name=ref_label,
    )
    speedup_summaries: List[str] = []
    if accuracy_file is not None and phi_ref is not None:
        speedup_summaries.append(
            f"accuracy reference: {accuracy_file.name} (phi={phi_ref:g}, criterion: OS2 error <= {ref_label} error)"
        )
    elif accuracy_file is None:
        speedup_summaries.append("accuracy reference: not found (showing max-based speedup fallback)")
    else:
        speedup_summaries.append("accuracy reference: unusable data (showing max-based speedup fallback)")

    for i, n in enumerate(n_list):
        ax = axes_1d[i]
        ax.grid(True, axis="both", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        y_candidates: List[float] = []

        ref_value = find_first_value(
            rows, n=n, token=ref_token, value_col="TFLOPS", contains=ref_contains
        )

        for label, token, contains, line_i, color_i in baselines:
            baseline = find_first_value(
                rows, n=n, token=token, value_col="TFLOPS", contains=contains
            )
            if baseline is None:
                continue
            y_candidates.append(baseline)
            ax.plot(xx, [baseline] * len(xx), wrapped_mark(line_i, color_i), label=label, linewidth=1.0)

        fast = collect_moduli_series(rows, n=n, prefix="OS2-fast", value_col="TFLOPS")
        accu = collect_moduli_series(rows, n=n, prefix="OS2-accu", value_col="TFLOPS")
        y_fast = [fast.get(m, math.nan) for m in xx]
        y_accu = [accu.get(m, math.nan) for m in xx]
        y_candidates.extend(v for v in y_fast if not math.isnan(v))
        y_candidates.extend(v for v in y_accu if not math.isnan(v))
        if any(not math.isnan(v) for v in y_fast):
            ax.plot(xx, y_fast, wrapped_mark(1, 3), label="OS II-fast", linewidth=1.0)
        if any(not math.isnan(v) for v in y_accu):
            ax.plot(xx, y_accu, wrapped_mark(1, 4), label="OS II-accu", linewidth=1.0)

        fast_max = _series_max(y_fast)
        accu_max = _series_max(y_accu)
        if ref_value is not None and ref_value > 0.0:
            txt_lines: List[str] = [f"vs {ref_label}"]
            summary_parts: List[str] = []
            m_fast = precision_moduli.get(n, {}).get("OS2-fast")
            m_accu = precision_moduli.get(n, {}).get("OS2-accu")

            displayed = False
            if m_fast is not None and m_fast in fast:
                fast_x = fast[m_fast] / ref_value
                txt_lines.append(f"fast@M{m_fast}: {fast_x:.2f}x")
                summary_parts.append(f"OS2-fast@M{m_fast} {fast_x:.2f}x")
                ax.plot([m_fast], [fast[m_fast]], "ok", markersize=4, markerfacecolor="none")
                displayed = True
            if m_accu is not None and m_accu in accu:
                accu_x = accu[m_accu] / ref_value
                txt_lines.append(f"accu@M{m_accu}: {accu_x:.2f}x")
                summary_parts.append(f"OS2-accu@M{m_accu} {accu_x:.2f}x")
                ax.plot([m_accu], [accu[m_accu]], "ok", markersize=4, markerfacecolor="none")
                displayed = True

            if not displayed:
                if fast_max is not None:
                    fast_x = fast_max / ref_value
                    txt_lines.append(f"fast(max): {fast_x:.2f}x")
                    summary_parts.append(f"OS2-fast(max) {fast_x:.2f}x")
                if accu_max is not None:
                    accu_x = accu_max / ref_value
                    txt_lines.append(f"accu(max): {accu_x:.2f}x")
                    summary_parts.append(f"OS2-accu(max) {accu_x:.2f}x")

            if summary_parts:
                ax.text(
                    0.02,
                    0.98,
                    "\n".join(txt_lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 2.0},
                )
                speedup_summaries.append(f"n={n}: {', '.join(summary_parts)} vs {ref_label}")

        ax.axvspan(cfg.highlight_min, cfg.highlight_max, facecolor="red", alpha=0.08)
        ax.set_xlim(cfg.xlim_min, cfg.moduli_max)
        ticks = list(range(cfg.xlim_min, cfg.moduli_max + 1))
        ax.set_xticks(ticks)
        ax.set_xticklabels(sparse_tick_labels(ticks))
        ax.tick_params(direction="out", labelsize=8)

        y_top = robust_linear_upper(
            y_candidates,
            mode=args.ylim_mode,
            q_high=args.q_high,
        )
        if y_top is not None:
            ax.set_ylim(0.0, y_top)
            if y_top >= 2.0:
                step = choose_y_tick_step(y_top)
                ax.set_yticks(list(range(0, int(y_top) + step, step)))

        ax.set_title(f"n={n}", fontsize=11)

    handles: List[object] = []
    labels: List[str] = []
    for ax in axes_1d:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        dedup = {}
        for h, lbl in zip(handles, labels):
            if lbl not in dedup:
                dedup[lbl] = h
        ncol = legend_ncol(len(dedup), max_cols=4)
        fig.legend(
            list(dedup.values()),
            list(dedup.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
            ncol=ncol,
        )

    fig.supxlabel("Number of moduli", fontsize=10)
    fig.supylabel("TFLOPS", fontsize=10)
    fig.suptitle(env_label_from_filename(path), fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = replace_stem_token(path.stem, "_time_", "_flops_", "_flops")
    out_path = out_dir / f"{out_stem}.png"
    fig.savefig(out_path, dpi=args.dpi)
    if not args.show:
        plt.close(fig)
    return out_path, speedup_summaries


def main() -> int:
    args = parse_args()
    plt = _import_matplotlib()
    if plt is None:
        return 1
    apply_paper_style(plt)

    script_dir = Path(__file__).resolve().parent
    default_glob = args.glob or f"oz2_results_{args.precision}_time_NVIDIA_*.csv"
    input_files = resolve_input_files(args.files, default_glob, script_dir)
    if not input_files:
        print(f"No input CSV files matched: ../{default_glob}", file=sys.stderr)
        return 1

    failures = 0
    for path in input_files:
        try:
            out, speedups = _plot_one_file(plt, path, args)
            print(f"Saved plot: {out}")
            for line in speedups:
                print(f"  {line}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {path}: {exc}", file=sys.stderr)

    if args.show:
        plt.show()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
