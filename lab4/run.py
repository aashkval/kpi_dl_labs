"""ЛР4, єдина команда: контроль поділу й калібрувальної вибірки → пп. 1–5 → lab4_results.json.

    uv run python run.py --model ../lab3/lab3_model.pt --split ../lab3/split.csv --data ../lab3/data

--data — тека набору (у ній seg_train/ і seg_test/). --out — тека для .onnx і lab4_results.json
(за замовчуванням lab4/). Прогноз досліду п. 4 зчитується з README.md (між мітками forecast)
до побудови потензорної моделі; без прогнозу команда зупиняється.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import sklearn
import torch
import torchvision
from sklearn.metrics import f1_score

import data as D
import onnx_tools as O

HERE = Path(__file__).resolve().parent
N_ROUNDS, N_WARMUP, N_TIMED = 3, 20, 100
EVAL_BATCH = 64
FORECAST_RE = re.compile(r"<!-- forecast:start -->(.*?)<!-- forecast:end -->", re.S)


# ---------------------------------------------------------------- допоміжне

def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro", labels=list(range(len(D.CLASSES)))))


def sh(*cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def environment() -> dict:
    cpu = sh("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor() or platform.machine()
    mac = platform.mac_ver()[0]
    power = sh("pmset", "-g", "batt").splitlines()
    return {
        "cpu": cpu,
        "machine": platform.machine(),
        "os": f"macOS {mac} ({platform.platform()})" if mac else platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "power_source": power[0].replace("Now drawing from ", "").strip("'") if power else None,
        "threads": {"intra_op_num_threads": 1, "inter_op_num_threads": 1},
        "latency_batch_size": 1,
    }


def git_info() -> dict:
    files = ["data.py", "onnx_tools.py", "run.py", "pyproject.toml", "uv.lock"]
    commit = sh("git", "-C", str(HERE), "rev-parse", "HEAD")
    dirty = bool(sh("git", "-C", str(HERE), "status", "--porcelain", "--", *files))
    return {"commit": commit or None, "uncommitted_code_changes": dirty}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def size_mb(path: Path) -> dict:
    b = Path(path).stat().st_size
    return {"bytes": b, "mb": b / 1e6}


class Results:
    def __init__(self, path: Path):
        self.path = path
        self.d: dict = {}

    def __setitem__(self, k, v):
        self.d[k] = v
        self.save()

    def __getitem__(self, k):
        return self.d[k]

    def save(self):
        self.path.write_text(json.dumps(self.d, ensure_ascii=False, indent=2) + "\n")


def fail(res: Results, msg: str):
    res["stopped"] = msg
    raise SystemExit(msg)


# ---------------------------------------------------------------- п. 3: якість і парне порівняння

def quality_row(y, pred, pred_fp32) -> dict:
    n = len(y)
    ok, ok32 = pred == y, pred_fp32 == y
    b = int((ok & ~ok32).sum())   # INT8 правильно, FP32 помиляється
    c = int((~ok & ok32).sum())   # навпаки
    se = math.sqrt(b + c) / n
    noise = 2 * math.sqrt(b + c)
    return {
        "macro_f1": macro_f1(y, pred),
        "accuracy": float(ok.mean()),
        "agreement_with_fp32": float((pred == pred_fp32).mean()),
        "n_disagree_with_fp32": int((pred != pred_fp32).sum()),
        "b": b, "c": c,
        "delta_acc": (b - c) / n,
        "se_delta_acc": se,
        "abs_b_minus_c": abs(b - c),
        "two_sqrt_b_plus_c": noise,
        "verdict": ("не відрізняється від шуму" if abs(b - c) <= noise else
                    ("відрізняється від шуму: INT8 краща" if b > c else "відрізняється від шуму: INT8 гірша")),
    }


# ---------------------------------------------------------------- п. 5: затримка

def latency(sessions: dict, x1: np.ndarray) -> dict:
    """Протокол п. 5: 3 раунди; у кожному FP32, потім INT8; 20 прогрівальних + 100 вимірів."""
    feed = {O.INPUT_NAME: x1}
    times = {name: [] for name in sessions}
    for _ in range(N_ROUNDS):
        for name, sess in sessions.items():
            run = sess.run
            for _ in range(N_WARMUP):
                run(None, feed)
            rt = []
            for _ in range(N_TIMED):
                t0 = time.perf_counter()
                run(None, feed)
                t1 = time.perf_counter()
                rt.append(t1 - t0)
            times[name].append(rt)
    out = {}
    for name, rounds in times.items():
        ms = np.array(rounds) * 1e3  # (3, 100)
        flat = ms.ravel()
        out[name] = {
            "p50_ms": float(np.percentile(flat, 50)),
            "p95_ms": float(np.percentile(flat, 95)),
            "p50_by_round_ms": [float(np.percentile(r, 50)) for r in ms],
            "p95_by_round_ms": [float(np.percentile(r, 95)) for r in ms],
            "mean_ms": float(flat.mean()), "min_ms": float(flat.min()), "max_ms": float(flat.max()),
            "all_ms_by_round": ms.tolist(),
        }
    return out


# ---------------------------------------------------------------- головна

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="lab3_model.pt")
    ap.add_argument("--split", type=Path, required=True, help="split.csv із ЛР3")
    ap.add_argument("--data", type=Path, required=True, help="тека набору з seg_train/ і seg_test/")
    ap.add_argument("--out", type=Path, default=HERE, help="тека для .onnx і lab4_results.json")
    ap.add_argument("--readme", type=Path, default=HERE / "README.md", help="README із прогнозом п. 4")
    ap.add_argument("--stop-before-per-tensor", action="store_true",
                    help="діагностика: зупинитися після діапазонів каналів, не будуючи потензорну модель")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    fp32_path, prep_path = out / "lab4_fp32.onnx", out / "lab4_fp32_prep.onnx"
    int8_path, pt_path = out / "lab4_int8.onnx", out / "lab4_int8_per_tensor.onnx"

    res = Results(out / "lab4_results.json")
    res["started_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    res["environment"] = environment()
    res["git"] = git_info()
    res["inputs"] = {"model": args.model.name, "model_sha256": sha256(args.model),
                     "split": args.split.name, "split_sha256": sha256(args.split)}
    print(f"Середовище: {res['environment']['cpu']}, {res['environment']['os']}, ORT {ort.__version__}")

    # ---- контроль поділу й калібрувальної вибірки (до всього іншого)
    rows = D.read_split(args.split)
    res["split_check"] = D.check_split(rows)
    if not res["split_check"]["all_match"]:
        fail(res, f"Поділ не збігається з контрольними числами: {res['split_check']}")
    calib_rows = D.calibration_rows(rows)
    res["calibration_check"] = D.check_calibration(calib_rows, rows)
    if not res["calibration_check"]["match"]:
        fail(res, "Калібрувальна вибірка не збігається з контрольними числами")
    print("Контроль поділу й калібрувальної вибірки: збіг.")

    # ---- п. 1: експорт і перевірка відповідності
    model, meta = O.load_lab3_model(args.model)
    img = int(meta["img_size"])
    res["model"] = meta
    test_rows = [r for r in rows if r["part"] == "test"]
    y = D.labels(test_rows)
    x_test = D.load_array(args.data, test_rows, img)

    t0 = time.perf_counter()
    O.export_fp32(model, img, fp32_path)
    t_export = time.perf_counter() - t0
    with torch.no_grad():
        z_pt = torch.cat([model(torch.from_numpy(x_test[i:i + EVAL_BATCH])) for i in range(0, len(x_test), EVAL_BATCH)]).numpy()
    sess_fp32 = O.session(fp32_path)
    z_fp32 = O.run_batches(sess_fp32, x_test, EVAL_BATCH)
    pred_pt, pred_fp32 = z_pt.argmax(1), z_fp32.argmax(1)
    ops = O.count_ops(fp32_path)
    sec1 = {
        "export": {"function": "torch.onnx.export", "dynamo": False, "opset_version": O.OPSET,
                   "input_names": [O.INPUT_NAME], "output_names": [O.OUTPUT_NAME],
                   "dynamic_axes": {O.INPUT_NAME: {"0": "batch"}, O.OUTPUT_NAME: {"0": "batch"}},
                   "model_mode": "eval", "time_s": t_export},
        "max_abs_logit_diff": float(np.abs(z_fp32 - z_pt).max()),
        "argmax_agreement": float((pred_pt == pred_fp32).mean()),
        "macro_f1_pytorch": macro_f1(y, pred_pt),
        "macro_f1_onnx_fp32": macro_f1(y, pred_fp32),
        "nodes_conv": ops.get("Conv", 0),
        "nodes_batchnorm": ops.get("BatchNormalization", 0),
        "op_counts": ops,
        "session": O.session_config(sess_fp32),
        "quality_eval": {"n_test": len(y), "batch_size": EVAL_BATCH, "order": "як рядки test у split.csv"},
    }
    sec1["checks"] = {
        "diff_below_1e-3": sec1["max_abs_logit_diff"] < 1e-3,
        "argmax_agreement_is_1": sec1["argmax_agreement"] == 1.0,
        "macro_f1_equal": sec1["macro_f1_pytorch"] == sec1["macro_f1_onnx_fp32"],
        "conv_20_bn_0": sec1["nodes_conv"] == 20 and sec1["nodes_batchnorm"] == 0,
    }
    res["section1_export"] = sec1
    print(f"П. 1: max|Δz| = {sec1['max_abs_logit_diff']:.3g}, збіг класів {sec1['argmax_agreement']:.4f}, "
          f"macro-F1 {sec1['macro_f1_pytorch']:.4f} / {sec1['macro_f1_onnx_fp32']:.4f}, Conv {sec1['nodes_conv']}, BN {sec1['nodes_batchnorm']}")
    if not all(sec1["checks"].values()):
        fail(res, f"Перевірка експорту не пройдена: {sec1['checks']}")

    # ---- п. 2: калібрування й квантизація (лише train)
    x_calib = D.load_array(args.data, calib_rows, img)
    t0 = time.perf_counter()
    O.preprocess(fp32_path, prep_path)
    t_prep = time.perf_counter() - t0
    t0 = time.perf_counter()
    O.quantize(prep_path, int8_path, x_calib, per_channel=True)
    t_quant = time.perf_counter() - t0
    res["section2_quantization"] = {
        "pre_process": {"function": "quant_pre_process", "skip_symbolic_shape": True, "time_s": t_prep},
        "quantize": O.quant_config(per_channel=True) | {"time_s": t_quant},
        "calibration_images": len(x_calib),
        "export_time_s": t_export,
        "pre_process_plus_quantize_time_s": t_prep + t_quant,
    }
    print(f"П. 2: експорт {t_export:.2f} с, підготовка {t_prep:.2f} с, квантизація {t_quant:.2f} с")

    # ---- п. 3: якість
    sess_int8 = O.session(int8_path)
    pred_int8 = O.run_batches(sess_int8, x_test, EVAL_BATCH).argmax(1)
    acc32 = float((pred_fp32 == y).mean())
    q_fp32 = quality_row(y, pred_fp32, pred_fp32)
    q_int8 = quality_row(y, pred_int8, pred_fp32)
    se_single = math.sqrt(acc32 * (1 - acc32) / len(y))
    res["section3_quality"] = {
        "n": len(y), "fp32": q_fp32, "int8": q_int8,
        "se_single_model_accuracy": se_single,
        "se_single_over_se_paired": se_single / q_int8["se_delta_acc"] if q_int8["se_delta_acc"] else None,
        "int8_predictions": pred_int8.tolist(), "fp32_predictions": pred_fp32.tolist(),
    }
    print(f"П. 3: INT8 macro-F1 {q_int8['macro_f1']:.4f}, згода {q_int8['agreement_with_fp32']:.4f}, "
          f"b={q_int8['b']}, c={q_int8['c']}, {q_int8['verdict']}")

    # ---- п. 4: діапазони каналів → прогноз (з README) → потензорна модель
    ranges = O.channel_ranges(fp32_path)
    rs = np.array([c["r"] for c in ranges])
    top = sorted(ranges, key=lambda c: -c["r"])[:3]
    sec4 = {"n_conv": len(ranges), "per_conv": ranges,
            "top3": [{"node": c["node"], "r": c["r"], "log2_r": c["log2_r"]} for c in top],
            "median_r": float(np.median(rs)), "log2_median_r": math.log2(float(np.median(rs)))}
    res["section4_per_tensor"] = sec4
    print("П. 4: найбільші r: " + ", ".join(f"{c['node']} {c['r']:.2f}" for c in top) +
          f"; медіана {sec4['median_r']:.2f} ({sec4['log2_median_r']:.2f} біта)")
    if args.stop_before_per_tensor:
        res["stopped"] = "--stop-before-per-tensor"
        return

    m = FORECAST_RE.search(args.readme.read_text()) if args.readme.exists() else None
    if not m or not m.group(1).strip():
        fail(res, f"У {args.readme} немає прогнозу між <!-- forecast:start --> і <!-- forecast:end --> — потензорну модель не будуємо")
    forecast = m.group(1).strip()
    sec4["forecast"] = {"text": forecast, "sha256": hashlib.sha256(forecast.encode()).hexdigest(),
                        "read_before_build_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    res["section4_per_tensor"] = sec4

    t0 = time.perf_counter()
    O.quantize(prep_path, pt_path, x_calib, per_channel=False)
    sec4["quantize"] = O.quant_config(per_channel=False) | {"time_s": time.perf_counter() - t0}
    pred_pt8 = O.run_batches(O.session(pt_path), x_test, EVAL_BATCH).argmax(1)
    sec4["quality"] = quality_row(y, pred_pt8, pred_fp32)
    sec4["quality_vs_int8_per_channel"] = {"agreement": float((pred_pt8 == pred_int8).mean())}
    sec4["predictions"] = pred_pt8.tolist()
    res["section4_per_tensor"] = sec4
    q = sec4["quality"]
    print(f"П. 4: потензорна INT8 macro-F1 {q['macro_f1']:.4f}, згода {q['agreement_with_fp32']:.4f}, "
          f"b={q['b']}, c={q['c']}, {q['verdict']}")

    # ---- п. 5: затримка
    sessions = {"fp32": O.session(fp32_path), "int8": O.session(int8_path)}  # створені заздалегідь
    x1 = np.ascontiguousarray(x_test[:1].copy())  # перше зображення test у порядку split.csv
    print("П. 5: вимірювання затримки (3 раунди × (20 + 100) на варіант)...", flush=True)
    res["environment"] = environment()  # стан живлення саме на момент вимірювання
    lat = latency(sessions, x1)
    for name, p in (("fp32", fp32_path), ("int8", int8_path)):
        lat[name]["file"] = p.name
        lat[name]["size"] = size_mb(p)
    f_r, i_r = lat["fp32"]["p50_by_round_ms"], lat["int8"]["p50_by_round_ms"]
    overlap = max(min(f_r), min(i_r)) <= min(max(f_r), max(i_r))
    res["section5_latency"] = {
        "protocol": {"rounds": N_ROUNDS, "warmup_per_variant_per_round": N_WARMUP,
                     "timed_per_variant_per_round": N_TIMED, "order_in_round": ["fp32", "int8"],
                     "batch_size": 1, "input": f"перше зображення test ({test_rows[0]['file']}), float32 {list(x1.shape)}",
                     "timer": "time.perf_counter() навколо одного sess.run",
                     "threads": {"intra_op_num_threads": 1, "inter_op_num_threads": 1},
                     "percentiles": "numpy.percentile, лінійна інтерполяція"},
        **lat,
        "size_ratio_fp32_over_int8": lat["fp32"]["size"]["bytes"] / lat["int8"]["size"]["bytes"],
        "speedup_p50": lat["fp32"]["p50_ms"] / lat["int8"]["p50_ms"],
        "speedup_p95": lat["fp32"]["p95_ms"] / lat["int8"]["p95_ms"],
        "round_p50_intervals_overlap": overlap,
        "change_measured": not overlap,
    }
    s5 = res["section5_latency"]
    print(f"П. 5: FP32 p50 {lat['fp32']['p50_ms']:.2f} мс, INT8 p50 {lat['int8']['p50_ms']:.2f} мс, "
          f"прискорення {s5['speedup_p50']:.2f}; виміряно: {s5['change_measured']}")
    res["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
