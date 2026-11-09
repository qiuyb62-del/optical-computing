"""Official MNIST IDX files and training-only, unwhitened PCA.

Download is opt-in. Checksums are pinned to torchvision v0.24.1's MNIST loader.
No torchvision, sklearn, PyTorch, or network is needed to replay saved features.
"""
from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import struct
from urllib.request import urlopen
import numpy as np
from .ai_data import integer, labels

RAW_DIR = Path(__file__).resolve().parents[1] / "data/mnist/raw"
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
RESOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def images_array(images):
    a = np.asarray(images)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[0] == 0 or a.shape[1:] != (28, 28):
        raise ValueError("images must be a nonempty uint8 array of shape (samples, 28, 28)")
    return a


def parse_idx(raw, *, count, images):
    """Reject truncated, overlong, wrongly shaped or wrongly labelled IDX data."""
    header_size = 16 if images else 8
    if len(raw) < header_size:
        raise ValueError("truncated IDX header")
    header = struct.unpack(">" + "I" * (header_size // 4), raw[:header_size])
    expected = (2051, count, 28, 28) if images else (2049, count)
    if header != expected or len(raw) != header_size + count * (784 if images else 1):
        raise ValueError("IDX shape or payload length mismatch")
    a = np.frombuffer(raw, dtype=np.uint8, offset=header_size).copy()
    return a.reshape(count, 28, 28) if images else labels(a, count, 10)


def load_mnist(raw_dir=RAW_DIR, *, download=False):
    raw_dir = Path(raw_dir)
    values, provenance = [], []
    for name, expected_md5 in RESOURCES.items():
        path = raw_dir / name
        if path.exists():
            raw = path.read_bytes()
        elif download:
            with urlopen(MIRROR + name, timeout=120) as response:
                raw = response.read()
        else:
            raise FileNotFoundError(f"Missing {path}; run verify_week06.py --download once")
        if hashlib.md5(raw).hexdigest() != expected_md5:
            raise ValueError(f"MNIST checksum mismatch: {name}")
        count = 60000 if name.startswith("train") else 10000
        values.append(parse_idx(gzip.decompress(raw), count=count, images="images" in name))
        if not path.exists():
            raw_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        provenance.append({"file": name, "url": MIRROR + name, "md5": expected_md5,
                           "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    return (*values, provenance)


def training_split(y, seed=20261109):
    """Exactly 1,000 validation images/class; remaining 50,000 are training."""
    y = labels(y, 60000, 10)
    rng = np.random.default_rng(integer(seed, "split seed"))
    train, validation = [], []
    for digit in range(10):
        ids = rng.permutation(np.flatnonzero(y == digit))
        if len(ids) <= 1000:
            raise ValueError("each digit needs more than 1000 training examples")
        validation.extend(ids[:1000])
        train.extend(ids[1000:])
    return {"train": rng.permutation(train), "validation": rng.permutation(validation)}


@dataclass(frozen=True, eq=False)
class PixelPCA:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def __post_init__(self):
        arrays = [np.asarray(v) for v in (self.mean, self.components, self.explained_variance_ratio)]
        m, c, r = arrays
        if (m.shape != (784,) or c.ndim != 2 or c.shape[1] != 784 or not 1 <= c.shape[0] <= 784
                or r.shape != (c.shape[0],) or any(np.iscomplexobj(v) or not np.isfinite(v).all() for v in arrays)
                or np.any(r < 0) or r.sum() > 1 + 1e-10
                or not np.allclose(c @ c.T, np.eye(len(c)), atol=1e-10, rtol=1e-10)):
            raise ValueError("invalid PCA mean, orthonormal components, or variance ratios")
        for key, value in zip(("mean", "components", "explained_variance_ratio"), arrays):
            a = np.array(value, dtype=np.float64, copy=True)
            a.setflags(write=False)
            object.__setattr__(self, key, a)

    @classmethod
    def fit(cls, training_images, n_components=16):
        images = images_array(training_images)
        n_components = integer(n_components, "n_components", 1)
        if len(images) < 2 or n_components > min(len(images) - 1, 784):
            raise ValueError("PCA needs at least n_components + 1 training samples")
        X = images.reshape(len(images), 784).astype(np.float64) / 255.0
        mean = X.mean(axis=0)
        X -= mean
        eigenvalues, eigenvectors = np.linalg.eigh((X.T @ X) / (len(X) - 1))
        eigenvalues = np.maximum(eigenvalues, 0)
        if eigenvalues.sum() == 0:
            raise ValueError("PCA cannot fit a constant dataset")
        components = eigenvectors[:, -n_components:][:, ::-1].T.copy()
        # Fix the arbitrary eigenvector sign for repeatable exported features.
        pivots = np.argmax(np.abs(components), axis=1)
        components *= np.where(components[np.arange(n_components), pivots] < 0, -1, 1)[:, None]
        ratios = eigenvalues[-n_components:][::-1] / eigenvalues.sum()
        return cls(mean, components, ratios)

    def transform(self, images, *, batch_size=4096):
        images = images_array(images)
        batch_size = integer(batch_size, "batch_size", 1)
        output = np.empty((len(images), len(self.components)), dtype=np.float64)
        for start in range(0, len(images), batch_size):
            X = images[start:start + batch_size].reshape(-1, 784).astype(np.float64) / 255.0
            output[start:start + len(X)] = (X - self.mean) @ self.components.T
        return output

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, components=self.components,
                            explained_variance_ratio=self.explained_variance_ratio)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["mean"], data["components"], data["explained_variance_ratio"])
