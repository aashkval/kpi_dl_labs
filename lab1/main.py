"""Основний запуск ЛР1: звірка ручного backprop з PyTorch і чисельна перевірка.

    uv run python main.py            # правильна реалізація (за замовчуванням)
    uv run python main.py --bug      # навмисна помилка: без ділення на N у dL/dZ2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sklearn
import torch

from checks import compare_with_torch, numerical_check
from data import load_data
from model_numpy import NumpyMLP, cross_entropy, init_params
from report import numeric_table, torch_table

RESULTS = Path(__file__).parent / "results"


def stability_demo() -> dict:
    """Наївна формула log(exp(z)/sum exp(z)) проти стабільного log-softmax."""
    Z = np.array([[1000.0, 0.0, -1000.0], [-1000.0, -1001.0, -1002.0]])
    y = np.array([1, 0])
    with np.errstate(all="ignore"):
        naive = -np.log(np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True))[np.arange(2), y].mean()
    stable, _ = cross_entropy(Z, y)
    return {"naive": float(naive), "stable": stable}


def run(bug: bool) -> dict:
    ds = load_data(seed=0)
    params = init_params(seed=0)
    X, y = ds.X_train, ds.y_train
    divide_by_n = not bug

    model = NumpyMLP(params)
    loss0 = model.forward(X, y)
    shapes = {k: list(v.shape) for k, v in model.cache.items()}

    torch_res = compare_with_torch(params, X, y, divide_by_n=divide_by_n)
    num_rows = numerical_check(params, X, y, divide_by_n=divide_by_n)

    return {
        "mode": "bug_no_division_by_N" if bug else "correct",
        "versions": {"python": __import__("sys").version.split()[0], "numpy": np.__version__,
                     "torch": torch.__version__, "scikit-learn": sklearn.__version__},
        "n_train": int(X.shape[0]), "n_test": int(ds.X_test.shape[0]),
        "train_mean": ds.mean.tolist(), "train_std": ds.std.tolist(),
        "initial_loss": loss0,
        "initial_test_accuracy": float((model.predict(ds.X_test) == ds.y_test).mean()),
        "cache_shapes": shapes,
        "stability_demo": stability_demo(),
        "torch": {k: v for k, v in torch_res.items() if not k.startswith("grads")},
        "grads_numpy": {k: v.tolist() for k, v in torch_res["grads_numpy"].items()},
        "grads_torch": {k: v.tolist() for k, v in torch_res["grads_torch"].items()},
        "numeric": num_rows,
        "_torch_full": torch_res,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bug", action="store_true",
                    help="увімкнути навмисну помилку (без ділення на N у градієнті за логітами)")
    args = ap.parse_args()

    res = run(bug=args.bug)
    title = "НАВМИСНА ПОМИЛКА (dZ2 без ділення на N)" if args.bug else "ПРАВИЛЬНА РЕАЛІЗАЦІЯ"
    md = [f"## {title}", "", "### Звірка з PyTorch", "", torch_table(res["_torch_full"]), "",
          "### Чисельна перевірка (ε = 1e-6, поріг 1e-7)", "", numeric_table(res["numeric"]), ""]
    text = "\n".join(md)
    print(text)
    s = res["stability_demo"]
    print(f"Стабільність: наївна CE = {s['naive']}, стабільна CE = {s['stable']}")

    all_ok = all(r["passed"] for r in res["torch"]["rows"]) and all(r["passed"] for r in res["numeric"])
    print("\nПідсумок:", "усі перевірки пройдено" if all_ok else "ПЕРЕВІРКИ ВИЯВИЛИ ПОМИЛКУ")

    RESULTS.mkdir(exist_ok=True)
    stem = "bug" if args.bug else "correct"
    res.pop("_torch_full")
    (RESULTS / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / f"{stem}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
