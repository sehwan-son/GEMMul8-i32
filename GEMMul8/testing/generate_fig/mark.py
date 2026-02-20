#!/usr/bin/env python3
"""Python port of mark.m."""

MARKERS = ["-", "--", "-.", "-d", "-+", "-o", "-s", "-x", "-p", "-h", "-^", "-v", "->", "-<"]
COLORS = ["k", "m", "r", "b", "g", "c"]


def mark(i: int, j: int) -> str:
    if i < 1 or i > len(MARKERS):
        raise ValueError(f"i must be in [1, {len(MARKERS)}], got {i}")
    if j < 1 or j > len(COLORS):
        raise ValueError(f"j must be in [1, {len(COLORS)}], got {j}")
    return MARKERS[i - 1] + COLORS[j - 1]

