"""ЛР3, команда навчання: поділ → перевірки пп. 1–2 → сім запусків → вибір lab3_model.pt.

    uv run python train.py --data data

--data — тека з розпакованим набором (у ній seg_train/ і seg_test/).
--out  — куди писати результати (за замовчуванням тека lab3/). Test тут не завантажується:
усі дані беруться з split.csv лише для частин train і validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import common as C

RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3), ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]
CHECK_STEPS = 5

BN_PREDICTION = (
    "Прогноз (записано до запуску досліду). Якщо в режимі frozen замість set_train_mode викликати "
    "model.train(), то параметри замороженого кістяка (conv1, bn1, layer1–layer4) не зміняться: "
    "вони мають requires_grad=False, градієнтів не отримують і в оптимізатор не передані. "
    "Зміняться ковзні статистики BN (running_mean, running_var, num_batches_tracked) у bn1 і в усіх "
    "BN шарів layer1–layer4: у режимі train() BN на кожному прямому проході оновлює їх за статистиками "
    "пакета незалежно від requires_grad. У conv1 BN немає; fc змінить параметри, як і в правильному варіанті. "
    "З кривої втрати на train цього майже не видно: втрата далі спадає, бо в train() BN нормує за статистиками "
    "пакета, і голова бачить узгоджені ознаки. Помилка проявиться лише при оцінці в eval(): там "
    "використовуються ковзні статистики, які вже не ImageNet-івські й залежать від історії пакетів, тобто "
    "ознаки на validation/test відрізняються від тих, на яких навчалась голова, а «заморожений» кістяк насправді "
    "змінюється від запуску до запуску."
)


# ---------------------------------------------------------------- допоміжне

def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def json_dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return repr(v)


# ---------------------------------------------------------------- п. 1

def check_split(rows, out: Path, log: list[str]) -> dict:
    table = C.split_control(rows)
    ok = True
    md_rows = []
    totals = {p: [0, 0] for p in C.PARTS}
    for cls in C.CLASSES:
        cells = []
        for p in C.PARTS:
            got, exp = table[cls][p], C.EXPECTED_SPLIT[cls][p]
            match = got == exp
            ok &= match
            totals[p][0] += got[0]
            totals[p][1] += got[1]
            cells.append(f"{got[0]} / {thousands(got[1])} {'✓' if match else '✗ очікувано ' + str(exp)}")
        md_rows.append([cls, *cells])
    md_rows.append(["**Разом**", *[f"{t[0]} / {thousands(t[1])}" for t in totals.values()]])

    by_part = {p: {r["file"] for r in rows if r["part"] == p} for p in C.PARTS}
    overlaps = {f"{a}∩{b}": len(by_part[a] & by_part[b]) for i, a in enumerate(C.PARTS) for b in C.PARTS[i + 1:]}
    unique = len({r["file"] for r in rows}) == len(rows)

    log += ["## 1. Перевірки до навчання", "", "### Поділ", "",
            md_table(["Клас", "train", "validation", "test"], md_rows), "",
            f"Усі 18 клітинок збігаються з контрольною таблицею: **{'так' if ok else 'НІ'}**.",
            f"Перетини частин (кількість спільних файлів): {overlaps}; усі імена в split.csv унікальні: {unique}.", ""]
    return {"control": table, "all_match": ok, "overlaps": overlaps, "unique_files": unique,
            "passed": ok and unique and not any(overlaps.values())}


def batch_figure(train_ds, path: Path) -> None:
    torch.manual_seed(0)
    x, y = next(iter(C.train_loader(train_ds)))
    mean = torch.tensor(C.MEAN).view(3, 1, 1)
    std = torch.tensor(C.STD).view(3, 1, 1)
    fig, axes = plt.subplots(4, 4, figsize=(9, 10))
    for ax, img, label in zip(axes.flat, x[:16], y[:16]):
        ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(C.CLASSES[label], fontsize=10)
        ax.axis("off")
    fig.suptitle("Перші 16 зображень навчального пакета після перетворень (обернене нормування)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def initial_loss(train_ds, device, log: list[str]) -> dict:
    res = {}
    for regime in ["scratch", "frozen"]:
        model, _ = C.build_model(regime, 1, device)
        C.set_train_mode(model, regime)
        x, y = next(iter(C.train_loader(train_ds)))
        with torch.no_grad():
            logits = model(x.to(device))
            loss = nn.CrossEntropyLoss()(logits, y.to(device)).item()
        res[regime] = {"L0": loss, "logit_std": logits.float().std().item(),
                       "logit_abs_max": logits.float().abs().max().item()}
        del model
    res["ln6"] = math.log(6)
    log += ["### Початкова втрата L0 (зерно 1, перший пакет, після set_train_mode, без кроків)", "",
            md_table(["Режим", "L0", "L0 − ln 6", "std логітів", "max \\|логіт\\|"],
                     [[r, f"{res[r]['L0']:.4f}", f"{res[r]['L0'] - res['ln6']:+.4f}",
                       f"{res[r]['logit_std']:.3f}", f"{res[r]['logit_abs_max']:.3f}"] for r in ["scratch", "frozen"]]),
            "", f"ln 6 = {res['ln6']:.4f}.", ""]
    return res


# ---------------------------------------------------------------- п. 2

def check_params(device, log: list[str]) -> dict:
    res, rows = {}, []
    ok = True
    for regime in C.REGIMES:
        model, opt = C.build_model(regime, 1, device)
        total = C.count(model.parameters())
        trainable = C.count(p for p in model.parameters() if p.requires_grad)
        in_opt = [p for g in opt.param_groups for p in g["params"]]
        frozen_in_opt = sum(1 for p in in_opt if not p.requires_grad)
        missing = {id(p) for p in model.parameters() if p.requires_grad} - {id(p) for p in in_opt}
        groups = [{"modules": mods, "n_params": C.count(g["params"]), "lr": g["lr"], "weight_decay": g["weight_decay"]}
                  for (mods, _), g in zip(C.REGIMES[regime]["groups"], opt.param_groups)]
        expected_trainable = {"scratch": 11_179_590, "frozen": 3_078, "unfreeze": 8_396_806}[regime]
        good = total == 11_179_590 and trainable == expected_trainable and frozen_in_opt == 0 and not missing
        ok &= good
        res[regime] = {"total": total, "trainable": trainable, "groups": groups,
                       "requires_grad_false_in_optimizer": frozen_in_opt,
                       "trainable_missing_from_optimizer": len(missing), "matches": good}
        gtxt = "; ".join(f"{'+'.join(g['modules']) if len(g['modules']) < 7 else 'уся мережа'}: "
                         f"{thousands(g['n_params'])} @ lr={g['lr']:g}" for g in groups)
        rows.append([regime, thousands(total), thousands(trainable), gtxt, frozen_in_opt, "так" if good else "НІ"])
        del model, opt
    log += ["## 2. Перевірка режимів навчання", "", "### Параметри", "",
            md_table(["Режим", "Усього параметрів", "requires_grad=True", "Групи AdamW (параметрів @ lr)",
                      "Параметрів із requires_grad=False в оптимізаторі", "Збігається"], rows), ""]
    return {"regimes": res, "passed": ok}


def five_steps(regime, train_ds, device, use_set_train_mode=True):
    """Новий екземпляр (зерно 1), копія state_dict, 5 кроків на перших 5 пакетах."""
    model, opt = C.build_model(regime, 1, device)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if use_set_train_mode:
        C.set_train_mode(model, regime)
    else:
        model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for step, (x, y) in enumerate(C.train_loader(train_ds)):
        if step == CHECK_STEPS:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    after = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    param_names = {n for n, _ in model.named_parameters()}
    return before, after, param_names, losses


def module_changes(before, after, param_names) -> dict:
    out = {}
    for m in C.TOP_MODULES:
        keys = [k for k in before if k.split(".")[0] == m]
        pkeys = [k for k in keys if k in param_names]
        bkeys = [k for k in keys if k.rsplit(".", 1)[-1] in ("running_mean", "running_var", "num_batches_tracked")]
        out[m] = {
            "params_changed": any(not torch.equal(before[k], after[k]) for k in pkeys),
            "has_bn": bool(bkeys),
            "bn_changed": any(not torch.equal(before[k], after[k]) for k in bkeys) if bkeys else None,
        }
    return out


def yes_no(v) -> str:
    return "немає BN" if v is None else ("так" if v else "ні")


def check_changes(train_ds, device, log: list[str]) -> dict:
    res, rows = {}, []
    ok = True
    for regime in ["frozen", "unfreeze"]:
        before, after, pnames, losses = five_steps(regime, train_ds, device)
        ch = module_changes(before, after, pnames)
        frozen_mods = C.REGIMES[regime]["eval_modules"]
        for m, c in ch.items():
            if m in frozen_mods:
                expected = (not c["params_changed"]) and not c["bn_changed"]
            else:
                expected = c["params_changed"] and (c["bn_changed"] if c["has_bn"] else True)
            c["matches_expected"] = expected
            ok &= expected
            rows.append([regime, m, yes_no(c["params_changed"]), yes_no(c["bn_changed"]), "так" if expected else "НІ"])
        res[regime] = {"modules": ch, "losses_5_steps": losses}
    log += ["### Що змінюється за 5 кроків (порівняння torch.equal, без допуску)", "",
            md_table(["Режим", "Модуль", "Параметри змінилися", "Ковзні статистики BN змінилися",
                      "Збігається з очікуваним"], rows), "",
            f"Усі рядки збігаються з очікуваним: **{'так' if ok else 'НІ'}**.", ""]
    return {"regimes": res, "passed": ok}


def bn_experiment(train_ds, device, out_md: Path, log: list[str]) -> dict:
    # Прогноз фіксується у файлі до запуску досліду.
    log += ["### Дослід із навмисною помилкою: frozen з model.train() замість set_train_mode", "", BN_PREDICTION, ""]
    out_md.write_text("\n".join(log) + "\n")

    b_ok, a_ok, pn, loss_ok = five_steps("frozen", train_ds, device, use_set_train_mode=True)
    b, a, pn, loss_bug = five_steps("frozen", train_ds, device, use_set_train_mode=False)
    ch = module_changes(b, a, pn)
    key = "layer1.0.bn1"
    d_mean = (a[f"{key}.running_mean"] - b[f"{key}.running_mean"]).abs().max().item()
    d_var = (a[f"{key}.running_var"] - b[f"{key}.running_var"]).abs().max().item()
    d_mean_ok = (a_ok[f"{key}.running_mean"] - b_ok[f"{key}.running_mean"]).abs().max().item()
    d_var_ok = (a_ok[f"{key}.running_var"] - b_ok[f"{key}.running_var"]).abs().max().item()

    predicted = {m: (m == "fc", None if m in ("conv1", "fc") else (m != "fc")) for m in C.TOP_MODULES}
    rows, ok = [], True
    for m, c in ch.items():
        exp_p, exp_bn = predicted[m]
        match = c["params_changed"] == exp_p and c["bn_changed"] == exp_bn
        ok &= match
        c["matches_prediction"] = match
        rows.append([m, yes_no(c["params_changed"]), yes_no(c["bn_changed"]), "так" if match else "НІ"])
    log += ["Результат:", "",
            md_table(["Модуль", "Параметри змінилися", "Ковзні статистики BN змінилися", "Збігається з прогнозом"], rows), "",
            md_table(["Величина", "set_train_mode (правильно)", "model.train() (помилка)"], [
                [f"max \\|Δ running_mean\\| у {key}", f"{d_mean_ok:.6g}", f"{d_mean:.6g}"],
                [f"max \\|Δ running_var\\| у {key}", f"{d_var_ok:.6g}", f"{d_var:.6g}"],
                ["втрата на 5 кроках", ", ".join(f"{v:.3f}" for v in loss_ok), ", ".join(f"{v:.3f}" for v in loss_bug)],
            ]), "",
            f"Результат збігається з прогнозом: **{'так' if ok else 'НІ'}**.", ""]
    return {"prediction": BN_PREDICTION, "modules": ch, "layer": key,
            "max_abs_delta_running_mean": d_mean, "max_abs_delta_running_var": d_var,
            "correct_max_abs_delta_running_mean": d_mean_ok, "correct_max_abs_delta_running_var": d_var_ok,
            "losses_correct": loss_ok, "losses_bug": loss_bug, "matches_prediction": ok}


# ---------------------------------------------------------------- п. 3

def transform_config(t) -> list[dict]:
    out = []
    for step in t.transforms:
        d = {"name": type(step).__name__}
        for k, v in vars(step).items():
            if not k.startswith("_") and k != "training":
                d[k] = jsonable(v)
        out.append(d)
    return out


def run_config(regime, seed, model, opt, criterion, train_ds, loader) -> dict:
    spec = C.REGIMES[regime]
    n_batches = len(loader)
    return {
        "seed": seed,
        "regime": regime,
        "dtype": "float32",
        "transforms": {
            "train": transform_config(train_ds.transform),
            "validation_test": transform_config(C.eval_transform()),
            "mean": C.MEAN, "std": C.STD,
            "image_open": "PIL.Image.open(...).convert('RGB')",
            "test_time_augmentation": False,
        },
        "model": {
            "arch": "torchvision.models.resnet18",
            "weights": f"ResNet18_Weights.{spec['weights']}" if spec["weights"] else None,
            "fc": "torch.nn.Linear(512, 6), стандартна ініціалізація",
            "trainable_modules": spec["trainable"],
            "eval_modules_during_training": spec["eval_modules"],
            "params_total": C.count(model.parameters()),
            "params_trainable": C.count(p for p in model.parameters() if p.requires_grad),
            "batchnorm": {"momentum": model.bn1.momentum, "eps": model.bn1.eps},
            "build_order": ["torch.manual_seed(seed); np.random.seed(seed)", "resnet18(weights=W)",
                            "fc = Linear(512, 6)", "requires_grad", "to(device); AdamW"],
        },
        "optimizer": {
            "class": "torch.optim.AdamW",
            "defaults": jsonable(opt.defaults),
            "groups": [{"modules": mods, "lr": g["lr"], "weight_decay": g["weight_decay"], "betas": g["betas"],
                        "eps": g["eps"], "amsgrad": g["amsgrad"], "n_params": C.count(g["params"])}
                       for (mods, _), g in zip(spec["groups"], opt.param_groups)],
        },
        "loss": {"class": "torch.nn.CrossEntropyLoss", "reduction": criterion.reduction,
                 "label_smoothing": criterion.label_smoothing, "weight": None},
        "loaders": {
            "train": {"batch_size": C.TRAIN_BATCH, "shuffle": True, "drop_last": False, "num_workers": 0,
                      "generator": None, "sampler": "default (RandomSampler з глобальним генератором PyTorch)"},
            "validation": {"batch_size": C.EVAL_BATCH, "shuffle": False, "num_workers": 0, "order": "як у split.csv"},
        },
        "epochs": C.EPOCHS,
        "steps_per_epoch": n_batches,
        "total_optimizer_steps": n_batches * C.EPOCHS,
        "last_batch_size": len(train_ds) - (n_batches - 1) * C.TRAIN_BATCH,
        "epoch_loss": "sum_b L_b * |b| / 1200, L_b — втрата пакета до кроку оптимізатора",
        "final_weights": "після 8-ї епохи (без вибору епохи за validation)",
        "not_used": ["lr scheduler", "gradient clipping", "mixed precision", "label smoothing", "class weights",
                     "early stopping", "checkpoint selection by validation"],
    }


def run_one(regime, seed, train_ds, val_ds, device, run_dir: Path, split_hash: str) -> dict:
    model, opt = C.build_model(regime, seed, device)
    criterion = nn.CrossEntropyLoss()
    loader = C.train_loader(train_ds)
    vloader = C.eval_loader(val_ds)
    assert len(loader) == 38 and len(train_ds) - 37 * 32 == 16
    config = run_config(regime, seed, model, opt, criterion, train_ds, loader)

    history = []
    train_time = 0.0
    t_all = time.perf_counter()
    for epoch in range(1, C.EPOCHS + 1):
        t0 = time.perf_counter()
        loss = C.train_epoch(model, opt, loader, regime, device, criterion)
        train_time += time.perf_counter() - t0
        logits = C.predict_logits(model, vloader, device)
        f1 = C.macro_f1(val_ds.labels, logits)
        history.append({"epoch": epoch, "train_loss": loss, "val_macro_f1": f1})
        print(f"  {regime} seed{seed} епоха {epoch}: втрата {loss:.4f}, macro-F1 {f1:.4f}", flush=True)
    total_time = time.perf_counter() - t_all

    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "val_logits.npy", logits.astype(np.float32))
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, run_dir / "model.pt")
    record = {
        "run": f"{regime}_seed{seed}",
        "seed": seed,
        "regime": regime,
        "config": config,
        "git": C.git_info(),
        "split_csv_sha256": split_hash,
        "versions": C.versions(),
        "device": C.device_name(device),
        "history": history,
        "final": {"train_loss": history[-1]["train_loss"], "val_macro_f1": history[-1]["val_macro_f1"]},
        "train_time_s": train_time,
        "train_plus_validation_time_s": total_time,
        "val_logits": {"file": "val_logits.npy", "shape": list(logits.shape)},
        "weights_digest": C.state_dict_digest(state),
    }
    json_dump(record, run_dir / "run.json")
    return record


# ---------------------------------------------------------------- п. 5

def select(records: dict[str, dict]) -> dict:
    by_regime: dict[str, list] = {}
    for rec in records.values():
        by_regime.setdefault(rec["regime"], []).append(rec)
    means = {r: float(np.mean([x["final"]["val_macro_f1"] for x in recs])) for r, recs in by_regime.items()}
    best_regime = max(C.REGIMES, key=lambda r: means[r])  # за рівності — перший у фіксованому порядку
    best = max(sorted(by_regime[best_regime], key=lambda x: x["seed"]), key=lambda x: x["final"]["val_macro_f1"])
    return {"means": means, "regime": best_regime, "seed": best["seed"], "run": best["run"],
            "val_macro_f1": best["final"]["val_macro_f1"]}


# ---------------------------------------------------------------- головна

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="тека набору з seg_train/ і seg_test/")
    ap.add_argument("--out", type=Path, default=C.HERE, help="тека результатів (за замовчуванням lab3/)")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (за замовчуванням — найкращий доступний)")
    ap.add_argument("--checks-only", action="store_true", help="лише поділ і перевірки пп. 1–2, без семи запусків")
    args = ap.parse_args()

    out = args.out
    (out / "checks").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else C.pick_device()
    print(f"Пристрій: {C.device_name(device)}")

    # Поділ
    rows = C.build_split(args.data)
    split_path = out / "split.csv"
    C.write_split(rows, split_path)
    split_hash = C.sha256_file(split_path)
    split_rows = C.read_split(split_path)  # далі дані — лише з файлу

    log = ["# Перевірки пп. 1–2 (вивід train.py)", "", f"Пристрій: {C.device_name(device)}.", ""]
    checks = {"device": C.device_name(device), "split_csv_sha256": split_hash}
    checks["split"] = check_split(split_rows, out, log)
    if not checks["split"]["passed"]:
        raise SystemExit("Поділ не збігається з контрольною таблицею — див. checks/report.md")

    print("Завантаження зображень train і validation...")
    train_ds = C.ImageList(args.data, C.part_rows(split_rows, "train"), C.train_transform())
    val_ds = C.ImageList(args.data, C.part_rows(split_rows, "validation"), C.eval_transform())
    assert len(train_ds) == 1200 and len(val_ds) == 1498

    batch_figure(train_ds, out / "figures" / "train_batch.png")
    log += ["### Пакет після перетворень", "", "Рисунок: `figures/train_batch.png`.", ""]
    checks["initial_loss"] = initial_loss(train_ds, device, log)
    checks["params"] = check_params(device, log)
    checks["changes"] = check_changes(train_ds, device, log)
    checks["bn_experiment"] = bn_experiment(train_ds, device, out / "checks" / "report.md", log)
    (out / "checks" / "report.md").write_text("\n".join(log) + "\n")
    json_dump(jsonable(checks), out / "checks" / "checks.json")
    print("\n".join(log))
    if not (checks["params"]["passed"] and checks["changes"]["passed"]):
        raise SystemExit("Перевірки п. 2 не збіглися з очікуваним — основний експеримент не запускається.")
    if args.checks_only:
        return

    # Сім запусків
    records = {}
    for regime, seed in RUNS:
        print(f"Запуск {regime} seed {seed}", flush=True)
        rec = run_one(regime, seed, train_ds, val_ds, device, out / "runs" / f"{regime}_seed{seed}", split_hash)
        records[rec["run"]] = rec

    # Вибір лише за validation
    sel = select(records)
    state = torch.load(out / "runs" / sel["run"] / "model.pt", map_location="cpu", weights_only=True)
    torch.save({"state_dict": state, "regime": sel["regime"], "seed": sel["seed"],
                "classes": C.CLASSES, "img_size": C.IMG_SIZE}, out / "lab3_model.pt")
    json_dump(sel, out / "runs" / "selection.json")
    print(f"Вибрано {sel['run']} (macro-F1 validation {sel['val_macro_f1']:.4f}); середні: {sel['means']}")


if __name__ == "__main__":
    main()
