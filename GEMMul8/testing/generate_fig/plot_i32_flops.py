#!/usr/bin/env python3
"""Plot i32 throughput curves from i32_bench_speedup_*.csv."""

import argparse
import os
import sys
from collections import defaultdict

from plot_i32_common import LinePlotOptions, metric_value, plot_lines, read_filtered_rows


FLOPS_ITEMS = [
    ("gemmul8_gflops", "gemmul8"),
    ("exact_i32_i32_i64_gflops", "exact_i32_i64"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot i32 GFLOPS metrics.")
    parser.add_argument("csv", help="Path to i32_bench_speedup_*.csv")
    parser.add_argument("--op", default="all", choices=["all", "NN", "NT", "TN", "TT"])
    parser.add_argument("--i32-scheme", default="all", choices=["all", "oz2", "oz1"])
    parser.add_argument("--use-extra", default="all", choices=["all", "true", "false"])
    parser.add_argument("--num-moduli", default="all")
    parser.add_argument("--x-axis", default="n", choices=["n", "m", "k", "mnk"])
    parser.add_argument("--yscale", default="auto", choices=["auto", "linear", "log"])
    parser.add_argument("--ylim-mode", default="robust", choices=["robust", "full"])
    parser.add_argument(
        "--show-int8-aux",
        action="store_true",
        help="Include int8 auxiliary baseline throughput series.",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = read_filtered_rows(
            args.csv,
            args.op,
            args.use_extra,
            args.num_moduli,
            i32_scheme=args.i32_scheme,
            x_axis=args.x_axis,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    grouped = defaultdict(list)
    is_fixed_config = (
        args.op != "all"
        and args.i32_scheme != "all"
        and args.use_extra != "all"
        and args.num_moduli != "all"
    )

    flops_items = list(FLOPS_ITEMS)
    if args.show_int8_aux:
        flops_items.extend([
            ("cublas_i8_single_gflops", "cublas_i8_single"),
            ("cublas_i8_x_moduli_gflops", "cublas_i8_x_moduli"),
        ])

    for row in rows:
        x = int(row["_x"])
        for key, short_name in flops_items:
            val = metric_value(row, key)
            if val is None:
                continue
            label = short_name if is_fixed_config else f"{short_name}, {row['_config']}"
            grouped[label].append((x, val))

    if not grouped:
        print("No rows matched filters (or selected metric missing).", file=sys.stderr)
        return 1

    title = "GEMMul8 i32 Throughput"
    xlabel = args.x_axis if args.x_axis != "mnk" else "m*n*k"
    ylabel = "GFLOPS"
    output = args.output
    if not output:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        suffix = "flops" if args.x_axis == "n" else f"flops_{args.x_axis}"
        output = f"{base}_{suffix}"

    opts = LinePlotOptions(
        yscale=args.yscale,
        ylim_mode=args.ylim_mode,
        include_zero=True,
    )
    saved, backend = plot_lines(grouped, title, xlabel, ylabel, output, options=opts)
    if backend == "svg":
        print("matplotlib not found: generated SVG plot instead.")
    print(f"Saved plot: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
