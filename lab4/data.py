"""ЛР4: поділ із split.csv ЛР3, контрольні числа, калібрувальна вибірка, перетворення.

Перетворення повторює eval_transform з lab3/common.py (без імпорту: ЛР4 самостійна).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Рядок «Разом» контрольної таблиці ЛР3 (кількість, сума числових імен файлів).
SPLIT_CONTROL = {"train": (1200, 11_993_084), "test": (1502, 33_351_444)}
# Контроль калібрувальної вибірки з умови ЛР4.
CALIB_SIZE = 200
CALIB_CONTROL = {"count": 200, "name_sum": 1_921_470,
                 "per_class": {"buildings": 40, "forest": 35, "glacier": 36, "mountain": 33, "sea": 29, "street": 27}}


class ControlError(SystemExit):
    pass


def read_split(path: Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def name_sum(rows: list[dict]) -> int:
    return sum(int(Path(r["file"]).stem) for r in rows)


def check_split(rows: list[dict]) -> dict:
    out = {}
    for part, (n_exp, s_exp) in SPLIT_CONTROL.items():
        sel = [r for r in rows if r["part"] == part]
        got = (len(sel), name_sum(sel))
        out[part] = {"count": got[0], "name_sum": got[1], "expected_count": n_exp,
                     "expected_name_sum": s_exp, "match": got == (n_exp, s_exp)}
    out["all_match"] = all(v["match"] for v in out.values() if isinstance(v, dict))
    return out


def calibration_rows(rows: list[dict]) -> list[dict]:
    """1200 рядків train, відсортовані за іменем файлу як рядки; номери permutation(1200)[:200]."""
    train = sorted((r for r in rows if r["part"] == "train"), key=lambda r: r["file"])
    assert len(train) == 1200
    idx = np.random.default_rng(0).permutation(len(train))[:CALIB_SIZE]
    return [train[i] for i in idx]


def check_calibration(calib: list[dict], rows: list[dict]) -> dict:
    per_class = {c: sum(r["label"] == c for r in calib) for c in CLASSES}
    test_files = {r["file"] for r in rows if r["part"] == "test"}
    out = {"count": len(calib), "name_sum": name_sum(calib), "per_class": per_class,
           "in_test": sum(r["file"] in test_files for r in calib),
           "all_from_train": all(r["part"] == "train" for r in calib),
           "files": [r["file"] for r in calib]}
    out["match"] = (out["count"] == CALIB_CONTROL["count"] and out["name_sum"] == CALIB_CONTROL["name_sum"]
                    and per_class == CALIB_CONTROL["per_class"] and out["in_test"] == 0 and out["all_from_train"])
    return out


def eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def image_path(data: Path, row: dict) -> Path:
    folder = "seg_train" if row["part"] == "train" else "seg_test"
    return Path(data) / folder / folder / row["label"] / row["file"]


def load_array(data: Path, rows: list[dict], img_size: int) -> np.ndarray:
    """Масив float32 форми (N, 3, img_size, img_size) у порядку rows."""
    t = eval_transform(img_size)
    out = np.empty((len(rows), 3, img_size, img_size), dtype=np.float32)
    for i, r in enumerate(rows):
        with Image.open(image_path(data, r)) as im:
            out[i] = t(im.convert("RGB")).numpy()
    return out


def labels(rows: list[dict]) -> np.ndarray:
    return np.array([CLASS_TO_IDX[r["label"]] for r in rows], dtype=np.int64)
