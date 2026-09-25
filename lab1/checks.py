"""Звірка з PyTorch і чисельна перевірка окремих градієнтів."""
from __future__ import annotations

import numpy as np

from model_numpy import PARAM_NAMES, NumpyMLP
from model_torch import torch_loss_and_grads

TORCH_TOL = 1e-12   # |a - b| <= 1e-12 для кожної пари скалярів
NUMERIC_TOL = 1e-7  # |g_num - g_manual| <= 1e-7
EPS = 1e-6

NUMERIC_TARGETS = (("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,)))


def compare_with_torch(params, X, y, divide_by_n: bool = True) -> dict:
    model = NumpyMLP(params)
    loss_np, g_np = model.loss_and_grads(X, y, divide_by_n=divide_by_n)
    loss_pt, g_pt = torch_loss_and_grads(params, X, y)

    rows = []
    diff = abs(loss_np - loss_pt)
    finite = bool(np.isfinite(loss_np) and np.isfinite(loss_pt))
    rows.append({"name": "Втрата", "max_abs_diff": diff,
                 "passed": bool(finite and diff <= TORCH_TOL)})
    for k in PARAM_NAMES:
        assert g_np[k].shape == g_pt[k].shape, (k, g_np[k].shape, g_pt[k].shape)
        d = np.abs(g_np[k] - g_pt[k])
        finite = bool(np.all(np.isfinite(g_np[k])) and np.all(np.isfinite(g_pt[k])))
        rows.append({"name": f"Градієнт {k}", "shape": list(g_np[k].shape),
                     "max_abs_diff": float(d.max()),
                     "passed": bool(finite and np.all(d <= TORCH_TOL))})
    return {"loss_numpy": loss_np, "loss_torch": loss_pt, "rows": rows,
            "grads_numpy": g_np, "grads_torch": g_pt}


def numerical_check(params, X, y, divide_by_n: bool = True, eps: float = EPS) -> list[dict]:
    """Центральна різниця (L+ - L-) / (2 eps) для вибраних елементів.

    Формула чисельної похідної не залежить від backward(), тому в досліді
    з помилкою вона лишається незмінним еталоном.
    """
    model = NumpyMLP(params)
    _, g_manual = model.loss_and_grads(X, y, divide_by_n=divide_by_n)

    rows = []
    for name, idx in NUMERIC_TARGETS:
        theta = model.params[name]
        original = theta[idx]                 # 1. зберегти
        theta[idx] = original + eps           # 2. +eps
        L_plus = model.forward(X, y)
        theta[idx] = original - eps           # 3. -eps
        L_minus = model.forward(X, y)
        theta[idx] = original                 # 4. відновити
        g_num = (L_plus - L_minus) / (2 * eps)
        g_man = float(g_manual[name][idx])
        diff = abs(g_num - g_man)
        rows.append({
            "name": f"{name}[{', '.join(map(str, idx))}]",
            "grad_backward": g_man, "grad_numeric": g_num, "abs_diff": diff,
            "passed": bool(np.isfinite(g_num) and np.isfinite(g_man) and diff <= NUMERIC_TOL),
        })
    # Перевіряємо, що параметри справді відновлено.
    for k in PARAM_NAMES:
        assert np.array_equal(model.params[k], params[k]), k
    return rows
