#!/usr/bin/env python3
"""Plot i32 GFLOPS/W from i32_bench_speedup_*.csv."""

import argparse
import os
import sys
from collections import defaultdict
from typing import Optional

from plot_i32_common import LinePlotOptions, metric_value, plot_lines, read_fieldnames, read_filtered_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot i32 GFLOPS/W.")
    parser.add_argument("csv", help="Path to i32_bench_speedup_*.csv")
    parser.add_argument("--op", default="all", choices=["all", "NN", "NT", "TN", "TT"])
    parser.add_argument("--use-extra", default="all", choices=["all", "true", "false"])
    parser.add_argument("--num-moduli", default="all")
    parser.add_argument("--x-axis", default="n", choices=["n", "m", "k", "mnk"])
    parser.add_argument("--yscale", default="auto", choices=["auto", "linear", "log"])
    parser.add_argument("--ylim-mode", default="robust", choices=["robust", "full"])
    parser.add_argument("--gemmul8-watt", type=float, default=None)
    parser.add_argument("--exact-watt", type=float, default=None)
    parser.add_argument("--cublas-single-watt", type=float, default=None)
    parser.add_argument("--cublas-xmoduli-watt", type=float, default=None)
    parser.add_argument(
        "--show-int8-aux",
        action="store_true",
        help="Include int8 auxiliary baseline efficiency series.",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _get_watt(
    row: dict,
    fieldnames: list,
    col_name: str,
    fallback: Optional[float],
) -> Optional[float]:
    if col_name in fieldnames:
        parsed = metric_value(row, col_name)
        if parsed is not None:
            return parsed
    return fallback


def main() -> int:
    args = parse_args()
    try:
        rows = read_filtered_rows(args.csv, args.op, args.use_extra, args.num_moduli, x_axis=args.x_axis)
        fieldnames = read_fieldnames(args.csv)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    selected_watt_columns = ["gemmul8_watt", "exact_i32_i32_i64_watt"]
    if args.show_int8_aux:
        selected_watt_columns.extend(["cublas_i8_single_watt", "cublas_i8_x_moduli_watt"])
    has_watt_columns = any(name in fieldnames for name in selected_watt_columns)
    if not has_watt_columns and args.gemmul8_watt is None:
        print(
            "[FAIL] CSV has no watt columns. Provide watt values via "
            "--gemmul8-watt/--exact-watt/--cublas-single-watt/--cublas-xmoduli-watt.",
            file=sys.stderr,
        )
        return 1

    grouped = defaultdict(list)
    is_fixed_config = args.op != "all" and args.use_extra != "all" and args.num_moduli != "all"

    for row in rows:
        x = int(row["_x"])

        gemmul8_gflops = metric_value(row, "gemmul8_gflops")
        exact_gflops = metric_value(row, "exact_i32_i32_i64_gflops")
        gemmul8_watt = _get_watt(row, fieldnames, "gemmul8_watt", args.gemmul8_watt)
        exact_watt = _get_watt(row, fieldnames, "exact_i32_i32_i64_watt", args.exact_watt)

        pairs = [
            ("gemmul8_gflops_per_watt", gemmul8_gflops, gemmul8_watt),
            ("exact_i32_i64_gflops_per_watt", exact_gflops, exact_watt),
        ]
        if args.show_int8_aux:
            cublas_single_gflops = metric_value(row, "cublas_i8_single_gflops")
            cublas_x_gflops = metric_value(row, "cublas_i8_x_moduli_gflops")
            cublas_single_watt = _get_watt(row, fieldnames, "cublas_i8_single_watt", args.cublas_single_watt)
            cublas_x_watt = _get_watt(row, fieldnames, "cublas_i8_x_moduli_watt", args.cublas_xmoduli_watt)
            pairs.extend([
                ("cublas_i8_single_gflops_per_watt", cublas_single_gflops, cublas_single_watt),
                ("cublas_i8_x_moduli_gflops_per_watt", cublas_x_gflops, cublas_x_watt),
            ])

        for short_name, gflops, watt in pairs:
            if gflops is None or watt is None or watt <= 0.0:
                continue
            val = gflops / watt
            label = short_name if is_fixed_config else f"{short_name}, {row['_config']}"
            grouped[label].append((x, val))

    if not grouped:
        print("No plottable points. Check watt inputs/columns and filters.", file=sys.stderr)
        return 1

    title = "GEMMul8 i32 GFLOPS/W"
    xlabel = args.x_axis if args.x_axis != "mnk" else "m*n*k"
    ylabel = "GFLOPS/W"
    output = args.output
    if not output:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        suffix = "gflops_per_watt" if args.x_axis == "n" else f"gflops_per_watt_{args.x_axis}"
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
