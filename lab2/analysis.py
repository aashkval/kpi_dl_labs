"""ЛР2, пп. 2–4. Аналіз порогів зі збережених прогнозів, без CSV і без навчання.

    uv run python analysis.py --saved results --out analysis

Порядок кроків фіксований: поріг t* обирається на validation, далі реалізація
пошуку перевіряється на наборах A/B/C, і лише після цього читаються масиви test.
Кроки винесено в окремі функції, щоб їх міг викликати й ноутбук lab2.ipynb.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
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


# --- п. 1: опис поділу ---


def split_table(data: dict[str, np.ndarray]) -> str:
    summary = split_summary(data)
    return md_table(
        ["Вибірка", "Об'єктів", "Шахрайських", "Частка шахрайських"],
        [[r.split, r.n, r.positives, f"{r.positive_fraction:.6f}"] for r in summary.itertuples()],
    )


# --- п. 2: вибір порога на validation ---


def select_threshold(data: dict[str, np.ndarray]) -> dict[str, object]:
    """Таблиця кандидатів, t* і рядки порівняння 0.5 / t* / +inf на validation."""
    y = data["validation_y"]
    s = data["validation_scores"].astype(np.float64)
    table = candidate_table(y, s)
    t_star = best_threshold(table)
    rules = {"Стандартний поріг": 0.5, "Мінімум вартості": t_star, "Усі негативні": math.inf}
    rows = {name: confusion(y, s, t) for name, t in rules.items()}
    # Таблиця кандидатів і прямий підрахунок мають давати ті самі числа для t* та +inf.
    for name in ("Мінімум вартості", "Усі негативні"):
        from_table = table[table["threshold"] == rules[name]].iloc[0].to_dict()
        if any(from_table[k] != rows[name][k] for k in COLUMNS):
            raise SystemExit(f"Розбіжність між таблицею кандидатів і прямим підрахунком: {name}")
    return {"y": y, "s": s, "table": table, "t_star": t_star, "rows": rows}


def validation_table(rows: dict[str, dict]) -> str:
    return md_table(
        ["Правило на validation", "Поріг", *COUNTERS],
        [[name, row["threshold"], *(row[k] for k in COUNTERS)] for name, row in rows.items()],
    )


def curve_points(validation: dict[str, object]) -> dict[str, dict]:
    rows = validation["rows"]
    return {
        "t = 0.5": rows["Стандартний поріг"],
        f"t* = {fmt(validation['t_star'])}": rows["Мінімум вартості"],
    }


def plot_curves(y: np.ndarray, s: np.ndarray, points: dict[str, dict]) -> tuple[float, Figure, Figure]:
    """PR- і ROC-криві на validation; повертає ROC-AUC та обидва рисунки."""
    auc = float(roc_auc_score(y, s))
    precision, recall, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)

    fig_pr, ax = plt.subplots(figsize=(6, 5))
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
    fig_pr.tight_layout()

    fig_roc, ax = plt.subplots(figsize=(6, 5))
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
    fig_roc.tight_layout()
    return auc, fig_pr, fig_roc


# --- п. 3: перевірка пошуку порога ---


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


def verification_table(check: dict) -> str:
    rows = [
        [slow["threshold"], *(slow[k] for k in COUNTERS), *(fast[k] for k in COUNTERS),
         "так" if all(slow[k] == fast[k] for k in COLUMNS) else "НІ"]
        for slow, fast in zip(check["brute_force"], check["cumulative"])
    ]
    return md_table(
        ["Поріг", *(f"{k} перебір" for k in COUNTERS), *(f"{k} кум. суми" for k in COUNTERS), "Збіг"],
        rows,
    )


def verification_verdict(check: dict) -> str:
    return (
        f"t* (перебір) = {fmt(check['t_star_brute_force'])}; "
        f"t* (кумулятивні суми) = {fmt(check['t_star_cumulative'])}; "
        f"перевірка {'пройдена' if check['passed'] else 'НЕ ПРОЙДЕНА: ' + '; '.join(check['problems'])}"
    )


# --- п. 4: тестова оцінка з уже зафіксованим t* ---


def evaluate_test(data: dict[str, np.ndarray], t_star: float) -> dict[str, object]:
    y = data["test_y"]
    s = data["test_scores"].astype(np.float64)
    te = confusion(y, s, t_star)
    # Довідкові рядки для висновків; на вибір порога вони не впливають.
    reference = {"0.5": confusion(y, s, 0.5), "+inf": confusion(y, s, math.inf)}

    pred = s >= t_star
    kind = np.where(pred & (y == 0), "FP", np.where(~pred & (y == 1), "FN", ""))
    errors = pd.DataFrame(
        {
            "csv_row": data["test_row_ids"],
            "type": kind,
            "label": y,
            "score": s,
            "amount": data["test_amount"],
            "s_minus_t": s - t_star,
        }
    )
    errors = errors[errors["type"] != ""].sort_values("csv_row").reset_index(drop=True)
    examples = pd.concat([errors[errors["type"] == k].head(N_EXAMPLES) for k in ("FP", "FN")])
    return {
        "t_star": t_star,
        "confusion": te,
        "precision": ratio(te["TP"], te["TP"] + te["FP"]),
        "recall": ratio(te["TP"], te["TP"] + te["FN"]),
        "reference": reference,
        "errors": errors,
        "examples": examples,
    }


def test_metrics_table(test: dict[str, object]) -> str:
    te = test["confusion"]
    return md_table(
        ["Поріг", *COUNTERS, "precision", "recall"],
        [[test["t_star"], *(te[k] for k in COUNTERS), test["precision"], test["recall"]]],
    )


def test_confusion_table(test: dict[str, object]) -> str:
    te = test["confusion"]
    return md_table(
        ["", "Прогноз 1", "Прогноз 0"],
        [["Мітка 1", f"TP = {te['TP']}", f"FN = {te['FN']}"], ["Мітка 0", f"FP = {te['FP']}", f"TN = {te['TN']}"]],
    )


def test_reference_table(test: dict[str, object]) -> str:
    te, reference = test["confusion"], test["reference"]
    return md_table(
        ["Поріг на test", *COUNTERS],
        [[0.5, *(reference["0.5"][k] for k in COUNTERS)],
         [test["t_star"], *(te[k] for k in COUNTERS)],
         [math.inf, *(reference["+inf"][k] for k in COUNTERS)]],
    )


def error_examples_table(examples: pd.DataFrame) -> str:
    return md_table(
        ["Рядок CSV (від 0, без заголовка)", "Тип помилки", "Мітка", "Оцінка s", "Amount", "s − t*"],
        [[r.csv_row, r.type, r.label, r.score, f"{r.amount:.2f}", f"{r.s_minus_t:+.6g}"]
         for r in examples.itertuples()],
    )


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


def error_summary_table(errors: pd.DataFrame) -> str:
    return md_table(
        ["Тип", "Кількість", "Amount: медіана", "Amount: середнє", "Amount: min – max",
         "abs(s − t*): медіана", "abs(s − t*): min", "abs(s − t*): max"],
        [error_summary(k, errors) for k in ("FP", "FN")],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--saved", type=Path, default=Path("results"), help="Тека зі збереженими прогнозами")
    parser.add_argument("--out", type=Path, default=Path("analysis"), help="Тека для результатів аналізу")
    args = parser.parse_args()
    matplotlib.use("Agg")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data = load_predictions(args.saved)
    report = ["## Поділ", split_table(data)]

    validation = select_threshold(data)
    table, t_star = validation["table"], validation["t_star"]
    table.to_csv(out / "candidates_validation.csv", index=False)
    auc, fig_pr, fig_roc = plot_curves(validation["y"], validation["s"], curve_points(validation))
    fig_pr.savefig(out / "pr_curve_validation.png", dpi=150)
    fig_roc.savefig(out / "roc_curve_validation.png", dpi=150)
    plt.close(fig_pr)
    plt.close(fig_roc)
    write_json(
        out / "threshold.json",
        {
            "t_star": t_star,
            "t_star_hex": float(t_star).hex(),
            "selected_on": "validation",
            "rule": "s >= t; мінімум C = 10*FN + FP; за рівної вартості найбільший поріг",
            "candidates": len(table),
            "validation": validation["rows"]["Мінімум вартості"],
        },
    )
    report += [
        "## Validation",
        f"ROC-AUC = {auc:.6f}; кандидатів: {len(table)}; t* = {t_star!r}",
        validation_table(validation["rows"]),
    ]

    # До test не переходимо, доки перевірка не пройдена.
    checks, passed = run_verification()
    write_json(out / "verification.json", checks)
    report.append("## Перевірка пошуку порога")
    for check in checks:
        report += [f"### Набір {check['set']}", verification_table(check), verification_verdict(check)]
    if not passed:
        (out / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
        raise SystemExit("Перевірка пошуку порога не пройдена — тестова оцінка не виконується.")

    test = evaluate_test(data, t_star)
    test["errors"].to_csv(out / "test_errors.csv", index=False)
    test["examples"].to_csv(out / "test_error_examples.csv", index=False)
    write_json(
        out / "test_evaluation.json",
        {
            "t_star": t_star,
            "confusion": test["confusion"],
            "precision": test["precision"],
            "recall": test["recall"],
            "reference_not_used_for_selection": test["reference"],
        },
    )
    report += [
        "## Test",
        test_metrics_table(test),
        test_confusion_table(test),
        "Довідково (не використовується для вибору):",
        test_reference_table(test),
        "## Приклади помилок на test",
        error_examples_table(test["examples"]),
        error_summary_table(test["errors"]),
    ]
    (out / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print("\n\n".join(report))
    print(f"\nРезультати аналізу збережено: {out.resolve()}")


if __name__ == "__main__":
    main()
