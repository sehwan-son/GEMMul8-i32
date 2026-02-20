#!/usr/bin/env python3
"""Shared helpers for OZ2 CSV plotting scripts."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from mark import COLORS, MARKERS, mark


@dataclass(frozen=True)
class PrecisionConfig:
    precision: str
    moduli_min: int
    moduli_max: int
    xlim_min: int
    highlight_min: int
    highlight_max: int
    accuracy_ylim: Tuple[float, float]
    accuracy_yticks_exp: range


PRECISION_CONFIG: Dict[str, PrecisionConfig] = {
    "f": PrecisionConfig(
        precision="f",
        moduli_min=2,
        moduli_max=15,
        xlim_min=2,
        highlight_min=6,
        highlight_max=9,
        accuracy_ylim=(1e-7, 1e6),
        accuracy_yticks_exp=range(-22, 31, 4),
    ),
    "d": PrecisionConfig(
        precision="d",
        moduli_min=2,
        moduli_max=20,
        xlim_min=8,
        highlight_min=14,
        highlight_max=18,
        accuracy_ylim=(1e-16, 1e6),
        accuracy_yticks_exp=range(-20, 31, 4),
    ),
}


def load_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_raw = next(reader, None)
        if header_raw is None:
            return [], []

        header = [h.strip() for h in header_raw if h.strip()]
        rows: List[Dict[str, str]] = []
        for raw in reader:
            if not any(cell.strip() for cell in raw):
                continue
            if len(raw) < len(header):
                raw = raw + [""] * (len(header) - len(raw))
            row = {header[i]: raw[i].strip() for i in range(len(header))}
            rows.append(row)
    return header, rows


def parse_float(text: str) -> Optional[float]:
    if text is None:
        return None
    t = text.strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def parse_int(text: str) -> Optional[int]:
    v = parse_float(text)
    if v is None:
        return None
    return int(v)


def unique_stable(values: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def alternating_tick_labels(ticks: Sequence[int]) -> List[str]:
    labels: List[str] = []
    for i, t in enumerate(ticks):
        # MATLAB: str(2:2:end) = "" => keep the first tick label, hide the second.
        labels.append(str(t) if i % 2 == 0 else "")
    return labels


def sparse_tick_labels(ticks: Sequence[int], dense_threshold: int = 10) -> List[str]:
    if len(ticks) <= dense_threshold:
        return [str(t) for t in ticks]
    return alternating_tick_labels(ticks)


def wrapped_mark(line_idx: int, color_idx: int) -> str:
    i = ((line_idx - 1) % len(MARKERS)) + 1
    j = ((color_idx - 1) % len(COLORS)) + 1
    return mark(i, j)


def env_label_from_filename(path: Path) -> str:
    stem = path.stem
    m = re.search(r"_NVIDIA_(.*?)_\d{4}-\d{2}-\d{2}", stem)
    env = m.group(1) if m else stem
    env = env.replace("_", " ").replace("-", " ")
    if "GH200" in env:
        return "GH200"
    if "A100" in env:
        return "A100 SXM4"
    if "RTX 4090" in env:
        return "RTX 4090"
    if "RTX 5080" in env:
        return "RTX 5080"
    return env


def resolve_input_files(
    positional_files: Sequence[str],
    default_glob: str,
    script_dir: Path,
) -> List[Path]:
    if positional_files:
        out = [Path(p).expanduser().resolve() for p in positional_files]
    else:
        out = sorted((script_dir.parent).glob(default_glob))
    return [p for p in out if p.exists()]


def find_first_value(
    rows: Sequence[Dict[str, str]],
    *,
    n: int,
    token: str,
    value_col: str,
    contains: bool = False,
) -> Optional[float]:
    for row in rows:
        n_val = parse_int(row.get("n", ""))
        if n_val != n:
            continue
        func = row.get("function", "")
        matched = token in func if contains else func == token
        if not matched:
            continue
        v = parse_float(row.get(value_col, ""))
        if v is not None:
            return v
    return None


def collect_moduli_series(
    rows: Sequence[Dict[str, str]],
    *,
    n: int,
    prefix: str,
    value_col: str,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    pat = re.compile(re.escape(prefix) + r"-(\d+)$")
    for row in rows:
        n_val = parse_int(row.get("n", ""))
        if n_val != n:
            continue
        func = row.get("function", "")
        m = pat.search(func)
        if not m:
            continue
        moduli = int(m.group(1))
        v = parse_float(row.get(value_col, ""))
        if v is None:
            continue
        out[moduli] = v
    return out


def choose_y_tick_step(ymax: float) -> int:
    for inc in (200, 150, 100, 50, 25, 10, 5, 2, 1):
        ticks_count = int(math.floor(max(ymax, 0.0) / inc)) + 1
        if ticks_count >= 4:
            return inc
    return 1


def apply_paper_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.9,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.8,
            "lines.linewidth": 1.8,
        }
    )


def quantile(sorted_vals: Sequence[float], q: float) -> float:
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


def robust_linear_upper(
    values: Sequence[float],
    *,
    mode: str = "robust",
    q_high: float = 0.95,
    margin: float = 1.08,
) -> Optional[float]:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return None
    if mode == "full":
        y = finite[-1]
    else:
        q = min(max(q_high, 0.0), 1.0)
        y = quantile(finite, q)
        y = max(y, finite[min(len(finite) - 1, len(finite) // 2)])
    return max(1e-12, y * margin)


def robust_log_bounds(
    values: Sequence[float],
    *,
    mode: str = "robust",
    q_low: float = 0.02,
    q_high: float = 0.98,
    margin_low: float = 0.85,
    margin_high: float = 1.12,
) -> Optional[Tuple[float, float]]:
    finite = sorted(v for v in values if math.isfinite(v) and v > 0.0)
    if not finite:
        return None
    if mode == "full":
        y0, y1 = finite[0], finite[-1]
    else:
        lo = min(max(q_low, 0.0), 1.0)
        hi = min(max(q_high, 0.0), 1.0)
        if hi < lo:
            lo, hi = hi, lo
        y0 = quantile(finite, lo)
        y1 = quantile(finite, hi)
        if y1 <= y0:
            y0, y1 = finite[0], finite[-1]
    y0 = max(1e-300, y0 * margin_low)
    y1 = max(y0 * 1.0001, y1 * margin_high)
    return y0, y1


def legend_ncol(num_items: int, max_cols: int = 4) -> int:
    if num_items <= 4:
        return 1
    if num_items <= 10:
        return min(max_cols, 2)
    if num_items <= 18:
        return min(max_cols, 3)
    return max_cols


def replace_stem_token(stem: str, old: str, new: str, fallback_suffix: str) -> str:
    if old in stem:
        return stem.replace(old, new)
    return stem + fallback_suffix


def split_function_and_k(function_name: str) -> Tuple[str, Optional[int]]:
    m = re.search(r"\(k\s*=\s*(\d+)\)", function_name)
    if not m:
        return function_name.strip(), None
    k = int(m.group(1))
    base = re.sub(r"\(k\s*=\s*\d+\)", "", function_name).strip()
    return base, k
