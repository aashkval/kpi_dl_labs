"""ЛР4: модель ЛР3 → ONNX FP32 → підготовка → статична INT8-квантизація; діапазони каналів згорток."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from data import CLASSES

OPSET = 17
INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def load_lab3_model(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if list(ckpt["classes"]) != CLASSES:
        raise SystemExit(f"Порядок класів у {path} не збігається з ЛР3: {ckpt['classes']}")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {k: ckpt[k] for k in ("regime", "seed", "classes", "img_size")}
    return model, meta


def export_fp32(model, img_size: int, path: Path) -> None:
    assert not model.training
    dummy = torch.zeros(1, 3, img_size, img_size)
    torch.onnx.export(
        model, (dummy,), str(path),
        dynamo=False, opset_version=OPSET,
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
    )


def count_ops(path: Path) -> dict:
    g = onnx.load(str(path)).graph
    ops: dict[str, int] = {}
    for n in g.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return ops


def session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    # рівень оптимізації графа — за замовчуванням (не змінюємо)
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def session_config(sess: ort.InferenceSession) -> dict:
    return {"providers": sess.get_providers(), "intra_op_num_threads": 1, "inter_op_num_threads": 1,
            "graph_optimization_level": str(ort.SessionOptions().graph_optimization_level)}


def run_batches(sess: ort.InferenceSession, x: np.ndarray, batch: int = 64) -> np.ndarray:
    return np.concatenate([sess.run([OUTPUT_NAME], {INPUT_NAME: x[i:i + batch]})[0]
                           for i in range(0, len(x), batch)])


class ArrayReader(CalibrationDataReader):
    """Калібрувальні зображення по одному, форма (1, 3, H, W)."""

    def __init__(self, images: np.ndarray):
        self.images = images
        self.i = 0

    def get_next(self):
        if self.i >= len(self.images):
            return None
        x = np.ascontiguousarray(self.images[self.i:self.i + 1])
        self.i += 1
        return {INPUT_NAME: x}

    def rewind(self):
        self.i = 0


def preprocess(fp32: Path, prep: Path) -> None:
    quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)


QUANT_ARGS = {
    "quant_format": QuantFormat.QDQ,
    "activation_type": QuantType.QUInt8,
    "weight_type": QuantType.QInt8,
    "calibrate_method": CalibrationMethod.MinMax,
}


def quantize(prep: Path, out: Path, calib: np.ndarray, per_channel: bool) -> None:
    quantize_static(str(prep), str(out), ArrayReader(calib), per_channel=per_channel, **QUANT_ARGS)


def quant_config(per_channel: bool) -> dict:
    return {"function": "onnxruntime.quantization.quantize_static", "per_channel": per_channel,
            **{k: str(v) for k, v in QUANT_ARGS.items()},
            "other_arguments": "за замовчуванням", "calibration_batch": 1}


def channel_ranges(path: Path) -> list[dict]:
    """Для кожного Conv: α_c = max|w| по вихідному каналу, r = max α / median α."""
    g = onnx.load(str(path)).graph
    inits = {i.name: i for i in g.initializer}
    out = []
    for n in g.node:
        if n.op_type != "Conv":
            continue
        w = numpy_helper.to_array(inits[n.input[1]])
        assert w.ndim == 4
        alpha = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
        r = float(alpha.max() / np.median(alpha))
        out.append({"node": n.name, "weight": n.input[1], "shape": list(w.shape),
                    "alpha_max": float(alpha.max()), "alpha_median": float(np.median(alpha)),
                    "alpha_min": float(alpha.min()), "r": r, "log2_r": math.log2(r)})
    return out
