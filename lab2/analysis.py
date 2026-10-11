"""ЛР2, пп. 2–4. Аналіз порогів зі збережених прогнозів, без CSV і без навчання.

    uv run python analysis.py --saved results --out analysis

Порядок кроків фіксований: поріг t* обирається на validation, далі реалізація
пошуку перевіряється на наборах A/B/C, і лише після цього читаються масиви test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

import verification
from lab2_starter import load_predictions, split_summary
from thresholds import COLUMNS, best_threshold, candidate_table, confusion

COUNTERS = ["TP", "FP", "FN", "TN", "C"]
N_EXAMPLES = 5


def fmt(value: object) -> str:
    """Округлення лише для відображення; у CSV/JSON числа зберігаються повністю."""
    if value is None:
        return "не визначена"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if math.isinf(value):
        return "+∞"
    return f"{value:.6g}"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v) if not isinstance(v, str) else v for v in r) + " |" for r in rows]
    return "\n".join(lines)


def ratio(numerator: int, denominator: int) -> float | None:
    """Частка або None, якщо знаменник нульовий (метрика не визначена)."""
    return numerator / denominator if denominator else None


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return "inf" if math.isinf(value) else float(value)
    return value


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(json_safe(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def plot_curves(y: np.ndarray, s: np.ndarray, points: dict[str, dict], out: Path) -> float:
    """PR- і ROC-криві на validation; повертає ROC-AUC."""
    auc = float(roc_auc_score(y, s))
    precision, recall, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=1.5)
    for label, row in points.items():
        p = ratio(row["TP"], row["TP"] + row["FP"])
        r = ratio(row["TP"], row["TP"] + row["FN"])
        if p is not None and r is not None:
            ax.scatter([r], [p], s=40, zorder=3, label=label)
    ax.set(xlabel="recall", ylabel="precision", title="PR-крива (validation)")
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "pr_curve_validation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=1.5, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="gray")
    for label, row in points.items():
        ax.scatter(
            [row["FP"] / (row["FP"] + row["TN"])],
            [row["TP"] / (row["TP"] + row["FN"])],
            s=40,
            zorder=3,
            label=label,
        )
    ax.set(xlabel="FPR = FP / (FP + TN)", ylabel="TPR = recall", title="ROC-крива (validation)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_curve_validation.png", dpi=150)
    plt.close(fig)
    return auc


def run_verification() -> tuple[list[dict], bool]:
    """Порівняти швидкий пошук із прямим перебором на наборах A/B/C."""
    results = []
    for name, (s, y) in verification.SETS.items():
        slow_rows = verification.brute_force_table(y, s)
        slow_best = verification.brute_force_best(slow_rows)
        fast = candidate_table(np.array(y), np.array(s))
        fast_rows = fast.to_dict("records")
        fast_best = best_threshold(fast)
        problems = verification.compare(fast_rows, fast_best, slow_rows, slow_best)
        results.append(
            {
                "set": name,
                "scores": s,
                "labels": y,
                "brute_force": slow_rows,
                "cumulative": fast_rows,
                "t_star_brute_force": slow_best,
                "t_star_cumulative": fast_best,
                "problems": problems,
                "passed": not problems,
            }
        )
    return results, all(r["passed"] for r in results)


def error_summary(kind: str, errors: pd.DataFrame) -> list[object]:
    part = errors[errors["type"] == kind]
    if part.empty:
        return [kind, 0] + [None] * 6
    amount, dist = part["amount"], part["s_minus_t"].abs()
    return [
        kind,
        len(part),
        f"{amount.median():.2f}",
        f"{amount.mean():.2f}",
        f"{amount.min():.2f} – {amount.max():.2f}",
        float(dist.median()),
        float(dist.min()),
        float(dist.max()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--saved", type=Path, default=Path("results"), help="Тека зі збереженими прогнозами")
    parser.add_argument("--out", type=Path, default=Path("analysis"), help="Тека для результатів аналізу")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data = load_predictions(args.saved)
    report: list[str] = []

    # --- п. 1: опис поділу ---
    summary = split_summary(data)
    report += [
        "## Поділ",
        md_table(
            ["Вибірка", "Об'єктів", "Шахрайських", "Частка шахрайських"],
            [[r.split, r.n, r.positives, f"{r.positive_fraction:.6f}"] for r in summary.itertuples()],
        ),
    ]

    # --- п. 2: вибір порога на validation ---
    y_va = data["validation_y"]
    s_va = data["validation_scores"].astype(np.float64)
    table = candidate_table(y_va, s_va)
    t_star = best_threshold(table)
    table.to_csv(out / "candidates_validation.csv", index=False)

    rules = {"Стандартний поріг": 0.5, "Мінімум вартості": t_star, "Усі негативні": math.inf}
    va_rows = {name: confusion(y_va, s_va, t) for name, t in rules.items()}
    # Таблиця кандидатів і прямий підрахунок мають давати ті самі числа для t* та +inf.
    for name in ("Мінімум вартості", "Усі негативні"):
        from_table = table[table["threshold"] == rules[name]].iloc[0].to_dict()
        if any(from_table[k] != va_rows[name][k] for k in COLUMNS):
            raise SystemExit(f"Розбіжність між таблицею кандидатів і прямим підрахунком: {name}")

    auc = plot_curves(
        y_va,
        s_va,
        {"t = 0.5": va_rows["Стандартний поріг"], f"t* = {fmt(t_star)}": va_rows["Мінімум вартості"]},
        out,
    )
    write_json(
        out / "threshold.json",
        {
            "t_star": t_star,
            "t_star_hex": float(t_star).hex(),
            "selected_on": "validation",
            "rule": "s >= t; мінімум C = 10*FN + FP; за рівної вартості найбільший поріг",
            "candidates": len(table),
            "validation": va_rows["Мінімум вартості"],
        },
    )
    report += [
        "## Validation",
        f"ROC-AUC = {auc:.6f}; кандидатів: {len(table)}; t* = {t_star!r}",
        md_table(
            ["Правило на validation", "Поріг", *COUNTERS],
            [[name, row["threshold"], *(row[k] for k in COUNTERS)] for name, row in va_rows.items()],
        ),
    ]

    # --- п. 3: перевірка пошуку; до test не переходимо, доки вона не пройдена ---
    checks, passed = run_verification()
    write_json(out / "verification.json", checks)
    report.append("## Перевірка пошуку порога")
    for check in checks:
        rows = [
            [slow["threshold"], *(slow[k] for k in COUNTERS), *(fast[k] for k in COUNTERS),
             "так" if all(slow[k] == fast[k] for k in COLUMNS) else "НІ"]
            for slow, fast in zip(check["brute_force"], check["cumulative"])
        ]
        report += [
            f"### Набір {check['set']}",
            md_table(
                ["Поріг", *(f"{k} перебір" for k in COUNTERS), *(f"{k} кум. суми" for k in COUNTERS), "Збіг"],
                rows,
            ),
            f"t* (перебір) = {fmt(check['t_star_brute_force'])}; "
            f"t* (кумулятивні суми) = {fmt(check['t_star_cumulative'])}; "
            f"перевірка {'пройдена' if check['passed'] else 'НЕ ПРОЙДЕНА: ' + '; '.join(check['problems'])}",
        ]
    if not passed:
        (out / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
        raise SystemExit("Перевірка пошуку порога не пройдена — тестова оцінка не виконується.")

    # --- п. 4: тестова оцінка з уже зафіксованим t* ---
    y_te = data["test_y"]
    s_te = data["test_scores"].astype(np.float64)
    te = confusion(y_te, s_te, t_star)
    precision = ratio(te["TP"], te["TP"] + te["FP"])
    recall = ratio(te["TP"], te["TP"] + te["FN"])
    # Довідкові рядки для висновків; на вибір порога вони не впливають.
    te_reference = {"0.5": confusion(y_te, s_te, 0.5), "+inf": confusion(y_te, s_te, math.inf)}

    pred = s_te >= t_star
    kind = np.where(pred & (y_te == 0), "FP", np.where(~pred & (y_te == 1), "FN", ""))
    errors = pd.DataFrame(
        {
            "csv_row": data["test_row_ids"],
            "type": kind,
            "label": y_te,
            "score": s_te,
            "amount": data["test_amount"],
            "s_minus_t": s_te - t_star,
        }
    )
    errors = errors[errors["type"] != ""].sort_values("csv_row").reset_index(drop=True)
    errors.to_csv(out / "test_errors.csv", index=False)
    examples = pd.concat([errors[errors["type"] == k].head(N_EXAMPLES) for k in ("FP", "FN")])
    examples.to_csv(out / "test_error_examples.csv", index=False)

    write_json(
        out / "test_evaluation.json",
        {
            "t_star": t_star,
            "confusion": te,
            "precision": precision,
            "recall": recall,
            "reference_not_used_for_selection": te_reference,
        },
    )
    report += [
        "## Test",
        md_table(
            ["Поріг", *COUNTERS, "precision", "recall"],
            [[t_star, *(te[k] for k in COUNTERS), precision, recall]],
        ),
        md_table(
            ["", "Прогноз 1", "Прогноз 0"],
            [["Мітка 1", f"TP = {te['TP']}", f"FN = {te['FN']}"], ["Мітка 0", f"FP = {te['FP']}", f"TN = {te['TN']}"]],
        ),
        "Довідково (не використовується для вибору):",
        md_table(
            ["Поріг на test", *COUNTERS],
            [[0.5, *(te_reference["0.5"][k] for k in COUNTERS)],
             [t_star, *(te[k] for k in COUNTERS)],
             [math.inf, *(te_reference["+inf"][k] for k in COUNTERS)]],
        ),
        "## Приклади помилок на test",
        md_table(
            ["Рядок CSV (від 0, без заголовка)", "Тип помилки", "Мітка", "Оцінка s", "Amount", "s − t*"],
            [[r.csv_row, r.type, r.label, r.score, f"{r.amount:.2f}", f"{r.s_minus_t:+.6g}"]
             for r in examples.itertuples()],
        ),
        md_table(
            ["Тип", "Кількість", "Amount: медіана", "Amount: середнє", "Amount: min – max",
             "abs(s − t*): медіана", "abs(s − t*): min", "abs(s − t*): max"],
            [error_summary(k, errors) for k in ("FP", "FN")],
        ),
    ]
    (out / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print("\n\n".join(report))
    print(f"\nРезультати аналізу збережено: {out.resolve()}")


if __name__ == "__main__":
    main()
