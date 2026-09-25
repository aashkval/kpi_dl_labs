"""Еталонна реалізація в PyTorch (autograd) — лише для звірки."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)


def build_torch_model(params: dict[str, np.ndarray]) -> nn.Sequential:
    """Мережа Linear(4, 8) -> ReLU -> Linear(8, 3) з параметрами NumPy-моделі.

    nn.Linear зберігає weight у формі (out, in) і обчислює x @ weight.T + bias,
    тому ваги NumPy (in, out) транспонуються.
    """
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).to(torch.float64)
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(params["W1"].T.copy()))
        model[0].bias.copy_(torch.from_numpy(params["b1"].copy()))
        model[2].weight.copy_(torch.from_numpy(params["W2"].T.copy()))
        model[2].bias.copy_(torch.from_numpy(params["b2"].copy()))
    return model


def torch_loss_and_grads(params, X: np.ndarray, y: np.ndarray):
    """Втрата й градієнти autograd, повернуті у формі NumPy-параметрів (in, out)."""
    model = build_torch_model(params)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).long()
    loss = nn.functional.cross_entropy(model(Xt), yt, reduction="mean")
    model.zero_grad()
    loss.backward()
    grads = {
        "W1": model[0].weight.grad.detach().numpy().T.copy(),
        "b1": model[0].bias.grad.detach().numpy().copy(),
        "W2": model[2].weight.grad.detach().numpy().T.copy(),
        "b2": model[2].bias.grad.detach().numpy().copy(),
    }
    return float(loss.item()), grads
