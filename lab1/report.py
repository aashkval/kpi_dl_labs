"""Форматування таблиць у Markdown."""
from __future__ import annotations


def fmt(x: float) -> str:
    return f"{x:.3e}"


def yes(ok: bool) -> str:
    return "так" if ok else "**ні**"


def torch_table(res: dict) -> str:
    lines = ["| Величина | Максимальна абсолютна різниця NumPy / PyTorch | Перевірку пройдено |",
             "|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['name']} | {fmt(r['max_abs_diff'])} | {yes(r['passed'])} |")
    lines.append("")
    lines.append(f"Втрата NumPy:   `{res['loss_numpy']:.17g}`  ")
    lines.append(f"Втрата PyTorch: `{res['loss_torch']:.17g}`")
    return "\n".join(lines)


def numeric_table(rows: list[dict]) -> str:
    lines = ["| Параметр | Градієнт backward() | Чисельна похідна | Абсолютна різниця | Перевірку пройдено |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['grad_backward']:.12e} | {r['grad_numeric']:.12e} "
                     f"| {fmt(r['abs_diff'])} | {yes(r['passed'])} |")
    return "\n".join(lines)
