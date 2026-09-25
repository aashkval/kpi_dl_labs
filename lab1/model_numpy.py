"""Двошарова мережа X -> Linear(4, 8) -> ReLU -> Linear(8, 3) на чистому NumPy.

Прямий і зворотний прохід реалізовано вручну, без автоматичного
диференціювання. Ваги зберігаються у формі (in, out), тобто Z = X @ W + b.
"""
from __future__ import annotations

import numpy as np

PARAM_NAMES = ("W1", "b1", "W2", "b2")


def init_params(seed: int = 0) -> dict[str, np.ndarray]:
    """Новий default_rng(seed); спочатку генерується W1, потім W2.

    W1: He, N(0, sqrt(2 / fan_in)) = N(0, sqrt(2/4));
    W2: Xavier, N(0, sqrt(2 / (fan_in + fan_out))) = N(0, sqrt(2/11));
    зсуви — нулі.
    """
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, np.sqrt(2.0 / 4), size=(4, 8))
    W2 = rng.normal(0.0, np.sqrt(2.0 / (8 + 3)), size=(8, 3))
    return {
        "W1": W1,
        "b1": np.zeros(8, dtype=np.float64),
        "W2": W2,
        "b2": np.zeros(3, dtype=np.float64),
    }


def log_softmax(Z: np.ndarray) -> np.ndarray:
    """Чисельно стабільний log-softmax по рядках: зсув на максимум рядка.

    log p_k = (z_k - m) - log(sum_j exp(z_j - m)),  m = max_j z_j.
    Після зсуву всі показники експоненти <= 0, тож exp не переповнюється,
    а сума не менша за 1, тож логарифм не дає -inf.
    """
    shifted = Z - Z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def cross_entropy(Z: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Середня крос-ентропія та log-ймовірності (N, C)."""
    logp = log_softmax(Z)
    N = Z.shape[0]
    loss = -logp[np.arange(N), y].mean()
    return float(loss), logp


class NumpyMLP:
    def __init__(self, params: dict[str, np.ndarray]):
        # Копії, щоб зовнішні зміни не впливали на модель непомітно.
        self.params = {k: np.array(v, dtype=np.float64, copy=True) for k, v in params.items()}
        self.cache: dict[str, np.ndarray] | None = None

    # ---------------- прямий прохід ----------------
    def forward(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.params
        Z1 = X @ p["W1"] + p["b1"]          # (N, 8)
        H = np.maximum(Z1, 0.0)              # (N, 8)
        Z2 = H @ p["W2"] + p["b2"]          # (N, 3) — логіти
        loss, logp = cross_entropy(Z2, y)
        # Зберігаємо рівно те, що потрібно зворотному проходу.
        self.cache = {
            "X": X,                # для dW1
            "relu_mask": Z1 > 0,   # похідна ReLU
            "H": H,                # для dW2
            "P": np.exp(logp),     # softmax, для dZ2
            "y": y,                # мітки, для dZ2
        }
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.params
        H = np.maximum(X @ p["W1"] + p["b1"], 0.0)
        return (H @ p["W2"] + p["b2"]).argmax(axis=1)

    # ---------------- зворотний прохід ----------------
    def backward(self, divide_by_n: bool = True) -> dict[str, np.ndarray]:
        """Градієнти середньої крос-ентропії за W1, b1, W2, b2.

        divide_by_n=False — навмисна помилка з п. 4: у градієнті за логітами
        пропущено ділення на N (функція втрат при цьому не змінюється).
        """
        if self.cache is None:
            raise RuntimeError("Спершу викличте forward().")
        c = self.cache
        X, mask, H, P, y = c["X"], c["relu_mask"], c["H"], c["P"], c["y"]
        N = X.shape[0]

        Y = np.zeros_like(P)
        Y[np.arange(N), y] = 1.0                 # one-hot, (N, 3)

        dZ2 = P - Y                              # (N, 3)
        if divide_by_n:
            dZ2 = dZ2 / N
        dW2 = H.T @ dZ2                          # (8, 3)
        db2 = dZ2.sum(axis=0)                    # (3,)
        dH = dZ2 @ self.params["W2"].T           # (N, 8)
        dZ1 = dH * mask                          # (N, 8)
        dW1 = X.T @ dZ1                          # (4, 8)
        db1 = dZ1.sum(axis=0)                    # (8,)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def loss_and_grads(self, X, y, divide_by_n: bool = True):
        loss = self.forward(X, y)
        return loss, self.backward(divide_by_n=divide_by_n)
