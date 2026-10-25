"""ЛР3: спільні частини обох команд — поділ, дані, перетворення, модель, режими, цикл навчання.

Тут немає коду, що читає test: набір test завантажує лише analyze.py.
"""

from __future__ import annotations

import csv
import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
PARTS = ["train", "validation", "test"]

TRAIN_PER_CLASS = 200
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

EPOCHS = 8
TRAIN_BATCH = 32
EVAL_BATCH = 64
WEIGHT_DECAY = 1e-4
HEAD_LR = 3e-3
BACKBONE_LR = 3e-4
N_CLASSES = len(CLASSES)

TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]

# Таблиця режимів із завдання: що навчається, групи AdamW, які модулі в eval() під час навчання.
REGIMES = {
    "scratch": {
        "weights": None,
        "trainable": TOP_MODULES,
        "groups": [(TOP_MODULES, HEAD_LR)],
        "eval_modules": [],
    },
    "frozen": {
        "weights": "IMAGENET1K_V1",
        "trainable": ["fc"],
        "groups": [(["fc"], HEAD_LR)],
        "eval_modules": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    },
    "unfreeze": {
        "weights": "IMAGENET1K_V1",
        "trainable": ["layer4", "fc"],
        "groups": [(["fc"], HEAD_LR), (["layer4"], BACKBONE_LR)],
        "eval_modules": ["conv1", "bn1", "layer1", "layer2", "layer3"],
    },
}

# Контрольна таблиця поділу із завдання: (кількість, сума числових імен файлів).
EXPECTED_SPLIT = {
    "buildings": {"train": (200, 1996761), "validation": (218, 4837323), "test": (219, 4837559)},
    "forest": {"train": (200, 2032625), "validation": (237, 5276962), "test": (237, 5291326)},
    "glacier": {"train": (200, 2183923), "validation": (276, 6135652), "test": (277, 6157250)},
    "mountain": {"train": (200, 1950345), "validation": (262, 5789535), "test": (263, 5849765)},
    "sea": {"train": (200, 1987613), "validation": (255, 5655068), "test": (255, 5668011)},
    "street": {"train": (200, 1841817), "validation": (250, 5541845), "test": (251, 5547533)},
}

HERE = Path(__file__).resolve().parent
CODE_FILES = ["common.py", "train.py"]  # код, від якого залежить навчання


# ---------------------------------------------------------------- поділ

def class_dir(data: Path, part_folder: str, cls: str) -> Path:
    return data / part_folder / part_folder / cls


def build_split(data: Path) -> list[dict]:
    """Поділ строго за процедурою завдання (імена відсортовано як рядки)."""
    rows = []
    rng = np.random.default_rng(0)
    for cls in CLASSES:
        names = sorted(p.name for p in class_dir(data, "seg_train", cls).glob("*.jpg"))
        idx = np.arange(len(names))
        rng.shuffle(idx)
        rows += [{"file": names[i], "label": cls, "part": "train"} for i in idx[:TRAIN_PER_CLASS]]

    rng = np.random.default_rng(0)
    for cls in CLASSES:
        names = sorted(p.name for p in class_dir(data, "seg_test", cls).glob("*.jpg"))
        idx = np.arange(len(names))
        rng.shuffle(idx)
        cut = len(names) // 2
        rows += [{"file": names[i], "label": cls, "part": "validation"} for i in idx[:cut]]
        rows += [{"file": names[i], "label": cls, "part": "test"} for i in idx[cut:]]
    return rows


def write_split(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "part"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_split(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def split_control(rows: list[dict]) -> dict:
    """Кількість і сума числових імен файлів для кожної клітинки (клас × частина)."""
    table = {c: {p: [0, 0] for p in PARTS} for c in CLASSES}
    for r in rows:
        cell = table[r["label"]][r["part"]]
        cell[0] += 1
        cell[1] += int(Path(r["file"]).stem)
    return {c: {p: tuple(v) for p, v in d.items()} for c, d in table.items()}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- дані

def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def image_path(data: Path, row: dict) -> Path:
    folder = "seg_train" if row["part"] == "train" else "seg_test"
    return class_dir(data, folder, row["label"]) / row["file"]


class ImageList(Dataset):
    """Зображення з переліку рядків split.csv, у тому самому порядку.

    Декодовані зображення тримаються в пам'яті (їх ~3 тис. по 150×150), щоб не
    читати JPEG щоепохи; випадкові перетворення застосовуються при кожному зверненні,
    тож споживання генератора PyTorch від кешу не залежить.
    """

    def __init__(self, data: Path, rows: list[dict], transform):
        self.rows = rows
        self.transform = transform
        self.labels = [CLASS_TO_IDX[r["label"]] for r in rows]
        self.images = []
        for r in rows:
            with Image.open(image_path(data, r)) as im:
                self.images.append(im.convert("RGB"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


def part_rows(split_rows: list[dict], part: str) -> list[dict]:
    if part == "test":
        raise RuntimeError("test завантажується лише в analyze.py")
    return [r for r in split_rows if r["part"] == part]


def train_loader(ds: Dataset) -> DataLoader:
    return DataLoader(ds, batch_size=TRAIN_BATCH, shuffle=True, drop_last=False, num_workers=0)


def eval_loader(ds: Dataset) -> DataLoader:
    return DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=0)


# ---------------------------------------------------------------- модель і режими

def build_model(regime: str, seed: int, device: torch.device):
    """Кроки 1–5 завдання: зерна → resnet18 → нова голова → requires_grad → пристрій і AdamW."""
    spec = REGIMES[regime]
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = ResNet18_Weights.IMAGENET1K_V1 if spec["weights"] else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(512, N_CLASSES)
    for name, p in model.named_parameters():
        p.requires_grad = name.split(".")[0] in spec["trainable"]
    model.to(device)
    groups = [
        {"params": [p for m in mods for p in getattr(model, m).parameters()], "lr": lr}
        for mods, lr in spec["groups"]
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    return model, optimizer


def set_train_mode(model: nn.Module, regime: str) -> None:
    """model.train(), потім eval() для заморожених модулів (їхні ковзні статистики BN не змінюються)."""
    model.train()
    for name in REGIMES[regime]["eval_modules"]:
        getattr(model, name).eval()


def count(params) -> int:
    return sum(p.numel() for p in params)


# ---------------------------------------------------------------- навчання та оцінка

def train_epoch(model, optimizer, loader, regime, device, criterion) -> float:
    set_train_mode(model, regime)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def predict_logits(model, loader, device) -> np.ndarray:
    model.eval()
    out = [model(x.to(device)).float().cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def macro_f1(labels, logits) -> float:
    return float(f1_score(labels, np.asarray(logits).argmax(1), average="macro", labels=list(range(N_CLASSES))))


def state_dict_digest(state_dict: dict) -> str:
    """SHA-256 вмісту всіх тензорів state_dict (імена, форми, типи, байти) — не залежить від формату файлу."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- середовище

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda: {torch.cuda.get_device_name(device)}"
    if device.type == "mps":
        chip = ""
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
        return f"mps: {chip or platform.machine()}"
    return f"cpu: {platform.processor() or platform.machine()}"


def git_info() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        # лише код і залежності: результати (runs/, split.csv, рисунки) пише саме навчання
        dirty = bool(run("status", "--porcelain", "--", *CODE_FILES, "pyproject.toml", "uv.lock"))
    except OSError:
        commit, dirty = "", None
    code = {f: sha256_file(HERE / f) for f in CODE_FILES if (HERE / f).exists()}
    return {"commit": commit or None, "uncommitted_changes_in_lab3": dirty, "code_sha256": code}


def versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
