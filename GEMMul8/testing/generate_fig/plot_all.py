#!/usr/bin/env python3
"""Run all relevant plotting scripts from a single CSV filename."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all plots from one CSV path (i32 or oz2)."
    )
    parser.add_argument("csv", help="Input CSV path")

    # i32 shared filters
    parser.add_argument("--op", default="all", choices=["all", "NN", "NT", "TN", "TT"])
    parser.add_argument("--use-extra", default="all", choices=["all", "true", "false"])
    parser.add_argument("--num-moduli", default="all")
    parser.add_argument("--x-axis", default="n", choices=["n", "m", "k", "mnk"])
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Also generate publication-oriented i32 figures (paper_speedup/throughput/timebreakdown/correctness).",
    )
    parser.add_argument(
        "--show-int8-aux",
        action="store_true",
        help="Include int8 auxiliary baseline series in i32 plots.",
    )

    # i32 watt fallback values (used if watt columns are empty)
    parser.add_argument("--gemmul8-watt", type=float, default=None)
    parser.add_argument("--exact-watt", type=float, default=None)
    parser.add_argument("--cublas-single-watt", type=float, default=None)
    parser.add_argument("--cublas-xmoduli-watt", type=float, default=None)
    parser.add_argument(
        "--watt-fallback",
        type=float,
        default=None,
        help="Shortcut to set all 4 fallback watt values to one number.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: Path) -> int:
    print("[RUN]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        print(f"[FAIL] exit={proc.returncode}: {' '.join(cmd)}", file=sys.stderr)
    return proc.returncode


def csv_has_nonempty_columns(csv_path: Path, columns: List[str]) -> bool:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for c in columns:
                raw = (row.get(c) or "").strip()
                if raw != "":
                    return True
    return False


def _call_i32_all(script_dir: Path, csv_path: Path, args: argparse.Namespace) -> int:
    py = sys.executable
    common = [
        "--op",
        args.op,
        "--use-extra",
        args.use_extra,
        "--num-moduli",
        args.num_moduli,
        "--x-axis",
        args.x_axis,
    ]

    speedup_cmd = [py, "plot_i32_speedup.py", str(csv_path), "--metric", "speedup_vs_exact_i32_i32_i64", *common]
    flops_cmd = [py, "plot_i32_flops.py", str(csv_path), *common]
    if args.show_int8_aux:
        speedup_cmd.append("--show-int8-aux")
        flops_cmd.append("--show-int8-aux")

    calls: List[List[str]] = [
        speedup_cmd,
        flops_cmd,
        [py, "plot_i32_timebreakdown.py", str(csv_path), *common, "--mode", "percent"],
        [py, "plot_i32_accuracy.py", str(csv_path), *common],
    ]
    if args.paper:
        calls.append([py, "plot_i32_paper.py", str(csv_path), *common])

    watt_cols = [
        "gemmul8_watt",
        "exact_i32_i32_i64_watt",
        "cublas_i8_single_watt",
        "cublas_i8_x_moduli_watt",
    ]
    has_watt_data = csv_has_nonempty_columns(csv_path, watt_cols)

    if args.watt_fallback is not None:
        gemmul8_watt = args.watt_fallback
        exact_watt = args.watt_fallback
        cublas_single_watt = args.watt_fallback
        cublas_xmoduli_watt = args.watt_fallback
    else:
        gemmul8_watt = args.gemmul8_watt
        exact_watt = args.exact_watt
        cublas_single_watt = args.cublas_single_watt
        cublas_xmoduli_watt = args.cublas_xmoduli_watt

    watt_cmd = [py, "plot_i32_watt.py", str(csv_path), *common]
    if args.show_int8_aux:
        watt_cmd.append("--show-int8-aux")
    if not has_watt_data:
        if gemmul8_watt is None:
            print(
                "[WARN] i32 CSV has empty watt columns. Skipping watt plot. "
                "Use --watt-fallback <W> or --gemmul8-watt/--exact-watt/--cublas-single-watt/--cublas-xmoduli-watt.",
                file=sys.stderr,
            )
        else:
            watt_cmd.extend(
                [
                    "--gemmul8-watt",
                    str(gemmul8_watt),
                    "--exact-watt",
                    str(exact_watt if exact_watt is not None else gemmul8_watt),
                    "--cublas-single-watt",
                    str(cublas_single_watt if cublas_single_watt is not None else gemmul8_watt),
                    "--cublas-xmoduli-watt",
                    str(cublas_xmoduli_watt if cublas_xmoduli_watt is not None else gemmul8_watt),
                ]
            )
            calls.append(watt_cmd)
    else:
        calls.append(watt_cmd)

    failures = 0
    for cmd in calls:
        failures += int(run_cmd(cmd, script_dir) != 0)
    return failures


def _call_oz2_all(script_dir: Path, csv_path: Path) -> int:
    py = sys.executable
    name = csv_path.name
    m = re.match(r"^oz2_results_([fd])_([A-Za-z0-9_-]+)_(NVIDIA_.*\.csv)$", name)
    if not m:
        print(f"[FAIL] unsupported oz2 filename: {name}", file=sys.stderr)
        return 1

    precision = m.group(1)
    kind = m.group(2)
    tail = m.group(3)
    parent = csv_path.parent

    if "rect" in kind:
        time_kind = "time-rect"
        watt_kind = "watt-rect"
        accuracy_kind = None
    else:
        time_kind = "time"
        watt_kind = "watt"
        accuracy_kind = "accuracy"

    calls: List[List[str]] = []

    if accuracy_kind is not None:
        acc = parent / f"oz2_results_{precision}_{accuracy_kind}_{tail}"
        if acc.exists():
            calls.append([py, "plot_accuracy.py", str(acc), "--precision", precision])
        else:
            print(f"[WARN] missing accuracy csv: {acc.name}", file=sys.stderr)

    tm = parent / f"oz2_results_{precision}_{time_kind}_{tail}"
    if tm.exists():
        calls.append([py, "plot_flops.py", str(tm), "--precision", precision])
        calls.append([py, "plot_timebreakdown.py", str(tm), "--precision", precision])
    else:
        print(f"[WARN] missing time csv: {tm.name}", file=sys.stderr)

    wt = parent / f"oz2_results_{precision}_{watt_kind}_{tail}"
    if wt.exists():
        calls.append([py, "plot_watt.py", str(wt), "--precision", precision])
    else:
        print(f"[WARN] missing watt csv: {wt.name}", file=sys.stderr)

    if not calls:
        print("[FAIL] no plot jobs found for input.", file=sys.stderr)
        return 1

    failures = 0
    for cmd in calls:
        failures += int(run_cmd(cmd, script_dir) != 0)
    return failures


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"[FAIL] file not found: {csv_path}", file=sys.stderr)
        return 1

    name = csv_path.name
    if name.startswith("i32_bench_speedup_"):
        failures = _call_i32_all(script_dir, csv_path, args)
    elif name.startswith("oz2_results_"):
        failures = _call_oz2_all(script_dir, csv_path)
    else:
        print(
            "[FAIL] unsupported csv type. expected i32_bench_speedup_* or oz2_results_*",
            file=sys.stderr,
        )
        return 1

    if failures:
        print(f"[DONE] completed with {failures} failed job(s).", file=sys.stderr)
        return 1
    print("[DONE] all plots generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
