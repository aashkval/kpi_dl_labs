"""Завантаження Iris, стратифікований поділ 70/30 і стандартизація.

Усі масиви повертаються у float64.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris

N_TRAIN_PER_CLASS = 35
N_TEST_PER_CLASS = 15


@dataclass(frozen=True)
class Dataset:
    X_train: np.ndarray  # (105, 4), стандартизовано
    y_train: np.ndarray  # (105,), int64
    X_test: np.ndarray   # (45, 4), стандартизовано статистиками train
    y_test: np.ndarray   # (45,)
    mean: np.ndarray     # (4,), середнє по train
    std: np.ndarray      # (4,), std по train, ddof=0


def stratified_split(y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Для класів 0, 1, 2 (саме в цьому порядку) перемішує індекси класу
    одним генератором default_rng(seed) і бере перші 35 у train, решту 15 у test."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in (0, 1, 2):
        idx = np.flatnonzero(y == c)
        idx = rng.permutation(idx)
        train_idx.append(idx[:N_TRAIN_PER_CLASS])
        test_idx.append(idx[N_TRAIN_PER_CLASS:])
    return np.concatenate(train_idx), np.concatenate(test_idx)


def load_data(seed: int = 0) -> Dataset:
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    tr, te = stratified_split(y, seed)
    X_tr, X_te = X[tr], X[te]

    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0, ddof=0)
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std  # ті самі статистики train

    ds = Dataset(X_tr, y[tr], X_te, y[te], mean, std)
    _validate(ds)
    return ds


def _validate(ds: Dataset) -> None:
    assert ds.X_train.shape == (105, 4) and ds.X_test.shape == (45, 4)
    assert np.all(np.bincount(ds.y_train) == N_TRAIN_PER_CLASS)
    assert np.all(np.bincount(ds.y_test) == N_TEST_PER_CLASS)
    assert ds.X_train.dtype == np.float64
