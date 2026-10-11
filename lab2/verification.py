"""ЛР2, п. 3. Незалежна перевірка пошуку порога прямим перебором.

Перевірочний алгоритм написано на чистому Python: він не імпортує thresholds.py
і не використовує його результатів. Для кожного кандидата виконується повний
прохід по вибірці (O(n^2)) — це дозволено лише тут, на малих наборах.
"""

from __future__ import annotations

import math

# Набори з умови: (оцінки s, мітки y).
SETS: dict[str, tuple[list[float], list[int]]] = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1] + [1] + [0] * 10 + [0]),
}


def brute_force_table(y: list[int], s: list[float]) -> list[dict[str, float | int]]:
    """Матриця плутанини та C для кожного кандидата прямим застосуванням s >= t."""
    candidates = [math.inf] + sorted(set(s), reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for label, score in zip(y, s, strict=True):
            predicted = score >= t
            if predicted and label == 1:
                tp += 1
            elif predicted and label == 0:
                fp += 1
            elif not predicted and label == 1:
                fn += 1
            else:
                tn += 1
        rows.append({"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": 10 * fn + fp})
    return rows


def brute_force_best(rows: list[dict[str, float | int]]) -> float:
    """Мінімум C; серед рівних мінімумів — найбільший поріг."""
    best_t = None
    best_c = None
    for row in rows:
        better = best_c is None or row["C"] < best_c
        tie_with_larger_t = best_c is not None and row["C"] == best_c and row["threshold"] > best_t
        if better or tie_with_larger_t:
            best_t, best_c = row["threshold"], row["C"]
    return best_t


def compare(
    fast_rows: list[dict[str, float | int]],
    fast_best: float,
    slow_rows: list[dict[str, float | int]],
    slow_best: float,
) -> list[str]:
    """Список розбіжностей; порожній список означає точний збіг."""
    problems = []
    if len(fast_rows) != len(slow_rows):
        return [f"різна кількість кандидатів: {len(fast_rows)} проти {len(slow_rows)}"]
    for fast, slow in zip(fast_rows, slow_rows, strict=True):
        for key in ("threshold", "TP", "FP", "FN", "TN", "C"):
            if fast[key] != slow[key]:
                problems.append(f"t={slow['threshold']}: {key} {fast[key]} проти {slow[key]}")
    if fast_best != slow_best:
        problems.append(f"обраний поріг: {fast_best} проти {slow_best}")
    return problems
