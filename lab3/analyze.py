"""ЛР3, команда аналізу: журнали → таблиця, криві, ефекти, перевірка вибору → одноразова оцінка test на CPU.

    uv run python analyze.py --data data

--data — тека з розпакованим набором (потрібна лише для зображень test).
--root — тека з split.csv, runs/ і lab3_model.pt (за замовчуванням lab3/).
Результати: analysis/report.md, analysis/results.json, рисунки в figures/.
Це єдиний модуль, що завантажує test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import confusion_matrix

import common as C
from train import RUNS, md_table, select

N_ERRORS = 5


def f(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def pts(x: float) -> str:
    """Різниця macro-F1 у пунктах (×100)."""
    return f"{100 * x:+.2f}"


# ---------------------------------------------------------------- журнали

def load_runs(root: Path) -> tuple[dict, str]:
    split_hash = C.sha256_file(root / "split.csv")
    records, bad = {}, []
    for regime, seed in RUNS:
        run_dir = root / "runs" / f"{regime}_seed{seed}"
        rec = json.loads((run_dir / "run.json").read_text())
        if rec["split_csv_sha256"] != split_hash:
            bad.append(f"{rec['run']}: {rec['split_csv_sha256']}")
        logits = np.load(run_dir / "val_logits.npy")
        rec["_val_logits"] = logits
        records[rec["run"]] = rec
    if bad:
        raise SystemExit(f"Хеш split.csv у журналах не збігається з поточним ({split_hash}):\n" + "\n".join(bad))
    if len({r["split_csv_sha256"] for r in records.values()}) != 1:
        raise SystemExit("Журнали записані на різних поділах — порівнювати не можна.")
    return records, split_hash


def check_logits(records: dict, val_labels: list[int]) -> list[str]:
    """F_{r,s} із журналу має збігатися з macro-F1, переобчисленим із збережених логітів."""
    problems = []
    for name, rec in records.items():
        lg = rec["_val_logits"]
        if lg.shape != (1498, 6):
            problems.append(f"{name}: форма логітів {lg.shape}")
        elif abs(C.macro_f1(val_labels, lg) - rec["final"]["val_macro_f1"]) > 1e-12:
            problems.append(f"{name}: macro-F1 з логітів не збігається з журналом")
    return problems


# ---------------------------------------------------------------- п. 3–4

def runs_table(records: dict) -> str:
    rows = []
    for i, (regime, seed) in enumerate(RUNS, 1):
        r = records[f"{regime}_seed{seed}"]
        rows.append([i, regime, seed, f(r["final"]["train_loss"]), f(r["final"]["val_macro_f1"]), f"{r['train_time_s']:.1f}"])
    return md_table(["#", "Режим", "Зерно", "Втрата train, епоха 8", "macro-F1 validation, епоха 8", "Час навчання, с"], rows)


def curves_figure(records: dict, path: Path) -> None:
    colors = {"scratch": "#c0392b", "frozen": "#2471a3", "unfreeze": "#1e8449"}
    styles = {1: "-", 2: "--", 3: ":"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for regime, seed in RUNS:
        r = records[f"{regime}_seed{seed}"]
        ep = [h["epoch"] for h in r["history"]]
        kw = dict(color=colors[regime], linestyle=styles[seed], marker="o", markersize=3, label=f"{regime}, зерно {seed}")
        a1.plot(ep, [h["train_loss"] for h in r["history"]], **kw)
        a2.plot(ep, [h["val_macro_f1"] for h in r["history"]], **kw)
    a1.set(title="Втрата на train", xlabel="епоха", ylabel="CrossEntropy")
    a2.set(title="macro-F1 на validation", xlabel="епоха", ylabel="macro-F1", ylim=(0, 1))
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.set_xticks(range(1, C.EPOCHS + 1))
    a2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def effects(records: dict) -> dict:
    F = {r: [records[f"{r}_seed{s}"]["final"]["val_macro_f1"] for s in (1, 2, 3)] for r in ("frozen", "unfreeze")}
    mean = {r: float(np.mean(v)) for r, v in F.items()}
    rng = {r: float(max(v) - min(v)) for r, v in F.items()}
    R = max(rng.values())
    f_scratch = records["scratch_seed1"]["final"]["val_macro_f1"]
    d_pre = mean["frozen"] - f_scratch
    d_unf = mean["unfreeze"] - mean["frozen"]

    def verdict(d):
        return "вартий перевірки (|Δ| > R)" if abs(d) > R else "не відрізняється від розкиду (|Δ| ≤ R)"

    return {"F": F, "mean": mean, "range": rng, "R": R, "F_scratch_1": f_scratch,
            "delta_pre": d_pre, "delta_unf": d_unf, "delta_total": d_pre + d_unf,
            "verdict_pre": verdict(d_pre), "verdict_unf": verdict(d_unf),
            "ratio_pre": abs(d_pre) / R if R else math.inf, "ratio_unf": abs(d_unf) / R if R else math.inf}


def effects_table(e: dict) -> str:
    return md_table(["Величина", "Значення"], [
        ["F̄_frozen, R_frozen", f"{f(e['mean']['frozen'])}, {f(e['range']['frozen'])}"],
        ["F̄_unfreeze, R_unfreeze", f"{f(e['mean']['unfreeze'])}, {f(e['range']['unfreeze'])}"],
        ["F_scratch,1", f(e["F_scratch_1"])],
        ["R = max(R_frozen, R_unfreeze)", f(e["R"])],
        ["Δ_pre = F̄_frozen − F_scratch,1", f"{f(e['delta_pre'])} ({pts(e['delta_pre'])} п.; \\|Δ\\|/R = {e['ratio_pre']:.1f}) — {e["verdict_pre"].replace("|", "\\|")}"],
        ["Δ_unf = F̄_unfreeze − F̄_frozen", f"{f(e['delta_unf'])} ({pts(e['delta_unf'])} п.; \\|Δ\\|/R = {e['ratio_unf']:.1f}) — {e["verdict_unf"].replace("|", "\\|")}"],
        ["Δ_pre + Δ_unf = F̄_unfreeze − F_scratch,1", f"{f(e['delta_total'])} ({pts(e['delta_total'])} п.)"],
    ])


def time_table(records: dict) -> str:
    rows = []
    for regime in C.REGIMES:
        ts = [r["train_time_s"] for r in records.values() if r["regime"] == regime]
        tv = [r["train_plus_validation_time_s"] for r in records.values() if r["regime"] == regime]
        rows.append([regime, len(ts), f"{np.mean(ts):.1f}", f"{np.mean(tv):.1f}", next(r["device"] for r in records.values())])
    return md_table(["Режим", "Запусків", "Середній час навчання, с", "Навчання + validation, с", "Пристрій"], rows)


# ---------------------------------------------------------------- п. 5: вибір

def verify_selection(records: dict, root: Path) -> tuple[dict, dict]:
    expected = select(records)
    ckpt = torch.load(root / "lab3_model.pt", map_location="cpu", weights_only=True)
    if set(ckpt) != {"state_dict", "regime", "seed", "classes", "img_size"}:
        raise SystemExit(f"lab3_model.pt має неочікувані ключі: {sorted(ckpt)}")
    run = records[expected["run"]]
    checks = {
        "regime": ckpt["regime"] == expected["regime"],
        "seed": ckpt["seed"] == expected["seed"],
        "classes": list(ckpt["classes"]) == C.CLASSES,
        "img_size": ckpt["img_size"] == C.IMG_SIZE,
        "weights_digest": C.state_dict_digest(ckpt["state_dict"]) == run["weights_digest"],
    }
    if not all(checks.values()):
        raise SystemExit(f"lab3_model.pt не відповідає правилу вибору за журналами: {checks}")
    return expected | {"checks": checks}, ckpt


# ---------------------------------------------------------------- п. 5: test

def load_test(data: Path, split_rows: list[dict]):
    rows = [r for r in split_rows if r["part"] == "test"]
    assert len(rows) == 1502
    return C.ImageList(data, rows, C.eval_transform())


def build_eval_model(ckpt: dict) -> torch.nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, C.N_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def confusion_figure(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), C.CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(6), C.CLASSES)
    ax.set(xlabel="передбачений клас", ylabel="справжній клас", title=title)
    thr = cm.max() / 2
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thr else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def pick_errors(cm: np.ndarray, rows: list[dict], y: np.ndarray, pred: np.ndarray) -> list[dict]:
    pairs = [(int(cm[i, j]), i, j) for i in range(6) for j in range(6) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    files = {(i, j): sorted((rows[k]["file"], k) for k in np.flatnonzero((y == i) & (pred == j)))
             for _, i, j in pairs}
    # Ранг 0: перший за іменем файл кожної пари (у порядку частоти); якщо пар менше п'яти —
    # ранг 1: другий файл найчастішої пари, потім другої за частотою і т. д.
    chosen, rank = [], 0
    while len(chosen) < N_ERRORS and any(len(v) > rank for v in files.values()):
        for cnt, i, j in pairs:
            if len(chosen) == N_ERRORS:
                break
            if len(files[(i, j)]) > rank:
                name, k = files[(i, j)][rank]
                chosen.append({"index": int(k), "file": name, "true": C.CLASSES[i], "pred": C.CLASSES[j], "count": cnt})
        rank += 1
    return chosen


def errors_figure(errors: list[dict], test_ds, path: Path) -> None:
    fig, axes = plt.subplots(1, len(errors), figsize=(3.2 * len(errors), 3.9))
    for ax, e in zip(np.atleast_1d(axes), errors):
        ax.imshow(test_ds.images[e["index"]])
        ax.set_title(f"{e['file']}\nсправжній: {e['true']}\nпередбачено: {e['pred']} ({e['prob']:.2f})", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- головна

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="тека набору з seg_test/")
    ap.add_argument("--root", type=Path, default=C.HERE, help="тека з split.csv, runs/, lab3_model.pt")
    args = ap.parse_args()
    root = args.root
    (root / "analysis").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)

    records, split_hash = load_runs(root)
    split_rows = C.read_split(root / "split.csv")
    val_labels = [C.CLASS_TO_IDX[r["label"]] for r in split_rows if r["part"] == "validation"]
    problems = check_logits(records, val_labels)
    if problems:
        raise SystemExit("Журнали не узгоджені з логітами:\n" + "\n".join(problems))

    commits = sorted({(r["git"]["commit"], r["git"]["uncommitted_changes_in_lab3"]) for r in records.values()}, key=str)
    code = sorted({json.dumps(r["git"]["code_sha256"], sort_keys=True) for r in records.values()})

    e = effects(records)
    curves_figure(records, root / "figures" / "curves.png")
    sel, ckpt = verify_selection(records, root)
    noise_choice = sel["regime"] in ("frozen", "unfreeze") and abs(e["delta_unf"]) <= e["R"]

    # ---- test: один раз, на CPU, лише для вибраної моделі
    torch.set_num_threads(max(1, torch.get_num_threads()))
    test_ds = load_test(args.data, split_rows)
    model = build_eval_model(ckpt)
    with torch.no_grad():
        logits = torch.cat([model(x) for x, _ in C.eval_loader(test_ds)]).numpy()
    y = np.array(test_ds.labels)
    pred = logits.argmax(1)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    test_f1 = C.macro_f1(y, logits)
    acc = float((pred == y).mean())
    n = len(y)
    se = math.sqrt(acc * (1 - acc) / n)
    cm = confusion_matrix(y, pred, labels=list(range(6)))
    np.save(root / "analysis" / "test_logits.npy", logits.astype(np.float32))
    confusion_figure(cm, root / "figures" / "confusion_test.png",
                     f"Test, {sel['run']}: macro-F1 = {test_f1:.4f}")
    errors = pick_errors(cm, test_ds.rows, y, pred)
    for er in errors:
        er["prob"] = float(probs[er["index"], C.CLASS_TO_IDX[er["pred"]]])
    errors_figure(errors, test_ds, root / "figures" / "test_errors.png")
    with (root / "analysis" / "test_predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["file", "label", "pred", "prob_pred"])
        for k, r in enumerate(test_ds.rows):
            w.writerow([r["file"], r["label"], C.CLASSES[pred[k]], f"{probs[k, pred[k]]:.6f}"])

    # ---- звіт
    cm_rows = [[f"**{C.CLASSES[i]}**", *cm[i].tolist(), int(cm[i].sum())] for i in range(6)]
    rep = [
        "# Аналіз ЛР3 (вивід analyze.py)", "",
        f"SHA-256 split.csv: `{split_hash}` — однаковий у всіх семи журналах і збігається з поточним файлом.",
        "macro-F1 у журналах збігається з переобчисленим зі збережених val_logits.npy.",
        f"Коміти й стан коду в журналах: {commits}; різних наборів хешів коду: {len(code)}.", "",
        "## 3. Сім запусків", "", runs_table(records), "",
        "Криві: `figures/curves.png`.", "",
        "### Час навчання за режимами", "", time_table(records), "",
        "## 4. Розклад ефектів", "",
        "F_{r,s} по зернах 1, 2, 3: " + "; ".join(f"{r}: " + ", ".join(f(v) for v in e["F"][r]) for r in e["F"]), "",
        effects_table(e), "",
        "## 5. Вибір і тестова оцінка", "",
        "Середні F̄_r на validation: " + ", ".join(f"{r} = {f(v)}" for r, v in sel["means"].items()) + ".",
        f"Вибрано режим **{sel['regime']}**, запуск **{sel['run']}** (F = {f(sel['val_macro_f1'])} на validation).",
        f"Перевірка lab3_model.pt проти журналів: {sel['checks']} — **відповідає правилу вибору**.",
        ("Оскільки |Δ_unf| ≤ R, вибір між frozen і unfreeze визначив шум." if noise_choice else
         "|Δ_unf| > R: вибір режиму не визначається лише шумом за прийнятим правилом."), "",
        md_table(["Величина", "Значення"], [
            ["macro-F1 test", f(test_f1)],
            ["accuracy test (p)", f(acc)],
            ["n", n],
            ["SE = sqrt(p(1−p)/n)", f"{se:.4f} ({100 * se:.2f} п.)"],
            [f"F вибраного запуску на validation", f(sel["val_macro_f1"])],
            ["test − validation (macro-F1)", f"{test_f1 - sel['val_macro_f1']:+.4f} ({pts(test_f1 - sel['val_macro_f1'])} п.)"],
        ]), "",
        "### Матриця плутанини test (рядки — справжній клас, стовпці — передбачений)", "",
        md_table(["", *C.CLASSES, "усього"], cm_rows), "",
        "Рисунок: `figures/confusion_test.png`.", "",
        "### П'ять характерних помилок", "",
        md_table(["Файл", "Справжній клас", "Передбачений клас", "Імовірність передбаченого класу", "Кількість таких помилок у test"],
                 [[er["file"], er["true"], er["pred"], f(er["prob"]), er["count"]] for er in errors]), "",
        "Рисунок: `figures/test_errors.png`.", "",
    ]
    (root / "analysis" / "report.md").write_text("\n".join(rep))
    results = {
        "split_csv_sha256": split_hash,
        "runs": {k: {"regime": r["regime"], "seed": r["seed"], **r["final"], "train_time_s": r["train_time_s"]}
                 for k, r in records.items()},
        "effects": e, "selection": sel, "choice_determined_by_noise": noise_choice,
        "test": {"macro_f1": test_f1, "accuracy": acc, "n": n, "se_accuracy": se,
                 "confusion_matrix": cm.tolist(), "errors": errors},
    }
    (root / "analysis" / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
