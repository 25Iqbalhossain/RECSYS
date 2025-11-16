from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


WEIGHTS_PATH = Path("data/artifacts/ranker_weights.json")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def train_logistic(X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 200, l2: float = 0.0) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=np.float32)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = _sigmoid(z)
        grad_w = (X.T @ (p - y)) / n + l2 * w
        grad_b = float(np.sum(p - y) / n)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, float(b)


def save_weights(feature_names: List[str], w: np.ndarray, b: float, path: Path = WEIGHTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"features": feature_names, "weights": [float(x) for x in w.tolist()], "bias": float(b)}
    path.write_text(json.dumps(payload, indent=2))


def load_weights(path: Path = WEIGHTS_PATH) -> Tuple[List[str], np.ndarray, float] | None:
    if not path.exists():
        return None
    obj = json.loads(path.read_text())
    names = list(obj.get("features", []))
    w = np.asarray(obj.get("weights", []), dtype=np.float32)
    b = float(obj.get("bias", 0.0))
    if w.size != len(names):
        return None
    return names, w, b


def score_with_weights(features: Dict[str, float], names: List[str], w: np.ndarray, b: float) -> float:
    x = np.asarray([float(features.get(k, 0.0)) for k in names], dtype=np.float32)
    return float(_sigmoid(x @ w + b))

