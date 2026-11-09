"""Pinned UCI Iris data and leakage-free 60/20/20 stratified splits."""
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path
import numpy as np

DATA_PATH = Path(__file__).resolve().parents[1] / "data/iris/bezdekIris.data"
DATA_SHA256 = "0fed2a99db77ec533a62dc66894d3ec6df3b58b6a8f3cf4a6b47e4086b7f97dc"
CLASS_NAMES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
FEATURE_NAMES = ("sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm")


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def features(X):
    raw = np.asarray(X)
    if np.iscomplexobj(raw):
        raise ValueError("Iris features must be real")
    X = np.array(raw, dtype=np.float64, copy=True)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != 4 or not np.all(np.isfinite(X)):
        raise ValueError("features must be a finite nonempty (samples, 4) array")
    return X


def labels(y, count):
    y = np.asarray(y)
    if y.shape != (count,) or y.dtype.kind not in "iu" or np.any((y < 0) | (y > 2)):
        raise ValueError("labels must be integer class IDs 0, 1, 2, one per sample")
    return y.astype(np.int64, copy=True)


def load_iris(path=DATA_PATH):
    """Read the exact vendored UCI file, with no network dependency at run time."""
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("Iris file checksum does not match the pinned UCI source")
    rows = [line.split(",") for line in raw.decode("ascii").splitlines() if line.strip()]
    X = features([[float(v) for v in row[:4]] for row in rows])
    y = labels([CLASS_NAMES.index(row[4]) for row in rows], len(X))
    if X.shape != (150, 4) or not np.array_equal(np.bincount(y), [50, 50, 50]):
        raise ValueError("unexpected Iris data shape or class counts")
    return X, y


def stratified_split(y, seed):
    """Each Iris class contributes 30 train, 10 validation and 10 test rows."""
    seed = integer(seed, "seed")
    y = labels(y, 150)
    if not np.array_equal(np.bincount(y, minlength=3), [50, 50, 50]):
        raise ValueError("the fixed Iris protocol requires 50 samples in each class")
    rng = np.random.default_rng(seed)
    parts = {"train": [], "validation": [], "test": []}
    for cls in range(3):
        idx = rng.permutation(np.flatnonzero(y == cls))
        for name, sub in zip(parts, (idx[:30], idx[30:40], idx[40:])):
            parts[name].extend(sub.tolist())
    return {name: rng.permutation(idx).astype(np.int64) for name, idx in parts.items()}


@dataclass(frozen=True)
class Standardizer:
    mean: tuple
    scale: tuple

    def __post_init__(self):
        mean, scale = np.asarray(self.mean), np.asarray(self.scale)
        if (mean.shape != (4,) or scale.shape != (4,) or np.iscomplexobj(mean)
                or np.iscomplexobj(scale) or not np.all(np.isfinite(mean))
                or not np.all(np.isfinite(scale)) or np.any(scale <= 0)):
            raise ValueError("standardizer requires four finite means and positive scales")
        object.__setattr__(self, "mean", tuple(map(float, mean)))
        object.__setattr__(self, "scale", tuple(map(float, scale)))

    @classmethod
    def fit(cls, X_train):
        X = features(X_train)
        scale = X.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        return cls(tuple(X.mean(axis=0)), tuple(scale))

    def transform(self, X):
        return (features(X) - np.asarray(self.mean)) / np.asarray(self.scale)

    def to_dict(self):
        return {"mean": self.mean, "scale": self.scale, "ddof": 0}
