#!/usr/bin/env python3
"""Plot i32 accuracy-related metrics from i32_bench_speedup_*.csv."""

import argparse
import os
import sys
from collections import defaultdict

from plot_i32_common import LinePlotOptions, metric_value, plot_lines, read_fieldnames, read_filtered_rows


ERROR_KEYS = [
    "gemmul8_mismatch_count",
    "gemmul8_max_abs_error",
    "gemmul8_exact_match",
    "max_relative_error",
    "max_rel_error",
    "relative_error",
]

SPEEDUP_ITEMS = [
    ("speedup_vs_exact_i32_i32_i64", "speedup_vs_exact"),
    ("speedup_vs_cublas_i8_single", "speedup_vs_cublas_i8_single"),
    ("speedup_vs_cublas_i8_x_moduli", "speedup_vs_cublas_i8_x_moduli"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot i32 accuracy(-proxy) metrics.")
    parser.add_argument("csv", help="Path to i32_bench_speedup_*.csv")
    parser.add_argument("--op", default="all", choices=["all", "NN", "NT", "TN", "TT"])
    parser.add_argument("--use-extra", default="all", choices=["all", "true", "false"])
    parser.add_argument("--num-moduli", default="all")
    parser.add_argument("--x-axis", default="n", choices=["n", "m", "k", "mnk"])
    parser.add_argument("--yscale", default="auto", choices=["auto", "linear", "log"])
    parser.add_argument("--ylim-mode", default="robust", choices=["robust", "full"])
    parser.add_argument(
        "--only-exact-speedup",
        action="store_true",
        help="When using speedup proxy mode, plot only speedup_vs_exact_i32_i32_i64.",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = read_filtered_rows(args.csv, args.op, args.use_extra, args.num_moduli, x_axis=args.x_axis)
        fieldnames = read_fieldnames(args.csv)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    grouped = defaultdict(list)
    is_fixed_config = args.op != "all" and args.use_extra != "all" and args.num_moduli != "all"

    err_key = next((k for k in ERROR_KEYS if k in fieldnames), None)
    if err_key is not None:
        for row in rows:
            val = metric_value(row, err_key)
            if val is None:
                continue
            x = int(row["_x"])
            label = err_key if is_fixed_config else f"{err_key}, {row['_config']}"
            grouped[label].append((x, val))
        title = "GEMMul8 i32 Accuracy"
        ylabel = err_key
        suffix = "accuracy"
    else:
        items = [SPEEDUP_ITEMS[0]] if args.only_exact_speedup else SPEEDUP_ITEMS
        for row in rows:
            x = int(row["_x"])
            for key, short_name in items:
                val = metric_value(row, key)
                if val is None:
                    continue
                label = short_name if is_fixed_config else f"{short_name}, {row['_config']}"
                grouped[label].append((x, val))
        title = "GEMMul8 i32 Accuracy Proxy (Speedup)"
        ylabel = "Speedup"
        suffix = "accuracy_proxy"
        print(
            "[WARN] No explicit error column in CSV. "
            "Plotted speedup-based proxy instead.",
            file=sys.stderr,
        )

    if not grouped:
        print("No rows matched filters (or selected metric missing).", file=sys.stderr)
        return 1

    xlabel = args.x_axis if args.x_axis != "mnk" else "m*n*k"
    output = args.output
    if not output:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        if args.x_axis != "n":
            suffix += f"_{args.x_axis}"
        output = f"{base}_{suffix}"

    use_yscale = args.yscale
    include_zero = True
    if err_key == "gemmul8_exact_match":
        # exact_match is binary data; log scale is invalid and not informative.
        if use_yscale == "auto":
            use_yscale = "linear"
        include_zero = False

    opts = LinePlotOptions(
        yscale=use_yscale,
        ylim_mode=args.ylim_mode,
        include_zero=include_zero,
    )
    saved, backend = plot_lines(grouped, title, xlabel, ylabel, output, options=opts)
    if backend == "svg":
        print("matplotlib not found: generated SVG plot instead.")
    print(f"Saved plot: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
