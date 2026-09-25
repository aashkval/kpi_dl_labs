"""П. 4: дослід із навмисною помилкою — порівняння правильного й помилкового backward().

    uv run python bug_experiment.py

Обидва варіанти обчислюються на тих самих початкових параметрах і тих самих
105 об'єктах, без жодного оновлення ваг між порівняннями.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from checks import compare_with_torch, numerical_check
from data import load_data
from model_numpy import PARAM_NAMES, NumpyMLP, init_params
from report import numeric_table, torch_table

RESULTS = Path(__file__).parent / "results"


def main() -> None:
    ds = load_data(0)
    params = init_params(0)
    X, y = ds.X_train, ds.y_train
    N = X.shape[0]

    m = NumpyMLP(params)
    loss_ok, g_ok = m.loss_and_grads(X, y, divide_by_n=True)
    loss_bug, g_bug = m.loss_and_grads(X, y, divide_by_n=False)
    for k in PARAM_NAMES:  # ваги не змінювались
        assert np.array_equal(m.params[k], params[k])

    ratio_rows = []
    for k in PARAM_NAMES:
        nz = g_ok[k] != 0
        r = g_bug[k][nz] / g_ok[k][nz]
        ratio_rows.append({
            "param": k, "n_elements": int(g_ok[k].size), "n_nonzero": int(nz.sum()),
            "ratio_min": float(r.min()), "ratio_max": float(r.max()),
            "zeros_stay_zero": bool(np.all(g_bug[k][~nz] == 0)),
        })

    torch_bug = compare_with_torch(params, X, y, divide_by_n=False)
    num_bug = numerical_check(params, X, y, divide_by_n=False)

    lines = [
        "## Дослід із навмисною помилкою", "",
        f"Втрата (правильна) = `{loss_ok:.17g}`, втрата (з помилкою) = `{loss_bug:.17g}`, "
        f"різниця = {abs(loss_ok - loss_bug):.1e}", "",
        f"Відношення градієнтів «помилка / правильно» для ненульових елементів (N = {N}):", "",
        "| Параметр | Елементів | Ненульових | min відношення | max відношення | Нулі лишились нулями |",
        "|---|---|---|---|---|---|",
    ]
    for r in ratio_rows:
        lines.append(f"| {r['param']} | {r['n_elements']} | {r['n_nonzero']} | {r['ratio_min']:.12f} "
                     f"| {r['ratio_max']:.12f} | {'так' if r['zeros_stay_zero'] else 'ні'} |")
    lines += ["", "### Звірка з PyTorch (з помилкою)", "", torch_table(torch_bug), "",
              "### Чисельна перевірка (з помилкою)", "", numeric_table(num_bug), ""]
    text = "\n".join(lines)
    print(text)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "bug_experiment.md").write_text(text, encoding="utf-8")
    (RESULTS / "bug_experiment.json").write_text(json.dumps({
        "loss_correct": loss_ok, "loss_bug": loss_bug, "ratios": ratio_rows,
        "torch_rows": torch_bug["rows"], "numeric_rows": num_bug,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
