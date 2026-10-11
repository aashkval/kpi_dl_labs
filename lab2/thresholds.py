"""ЛР2. Лічильники матриці плутанини, вартість помилок і швидкий пошук порога.

Правило рішення всюди однакове: прогноз «шахрайство», якщо s >= t.
Вартість: C = 10*FN + FP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COST_FN = 10
COST_FP = 1
COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]


def cost(fp: int, fn: int) -> int:
    return COST_FN * fn + COST_FP * fp


def confusion(y: np.ndarray, s: np.ndarray, t: float) -> dict[str, float | int]:
    """TP, FP, FN, TN і C для одного порога t за правилом s >= t."""
    y = np.asarray(y)
    pred = np.asarray(s, dtype=np.float64) >= t
    pos = y == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    tn = int(np.count_nonzero(~pred & ~pos))
    return {"threshold": float(t), "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fp, fn)}


def candidate_table(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Усі кандидати (+inf та унікальні оцінки) за одне сортування, O(n log n).

    Рядки впорядковано за спаданням порога: першим іде +inf («усі негативні»),
    останнім — найменша оцінка («усі позитивні»).
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)  # float32 -> float64 без втрати точності
    if y.shape != s.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("y та s мають бути непорожніми одновимірними масивами однакової довжини.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Мітки мають бути 0/1.")
    if not np.isfinite(s).all():
        raise ValueError("Оцінки мають бути скінченними.")

    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    pos_sorted = (y[order] == 1).astype(np.int64)
    cum_tp = np.cumsum(pos_sorted)
    cum_fp = np.cumsum(1 - pos_sorted)

    # Останній індекс кожної групи рівних оцінок: поріг, що дорівнює оцінці групи,
    # позначає позитивною всю групу разом із усіма більшими оцінками.
    last = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])

    thresholds = np.r_[np.inf, s_sorted[last]]
    tp = np.r_[0, cum_tp[last]]
    fp = np.r_[0, cum_fp[last]]
    fn = int(cum_tp[-1]) - tp
    tn = int(cum_fp[-1]) - fp
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "C": COST_FN * fn + COST_FP * fp,
        },
        columns=COLUMNS,
    )


def best_threshold(table: pd.DataFrame) -> float:
    """Поріг із мінімальною вартістю; за рівної вартості — найбільший."""
    c = table["C"].to_numpy()
    t = table["threshold"].to_numpy()
    return float(t[c == c.min()].max())
