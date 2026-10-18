from __future__ import annotations

import hashlib
import math
import re


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def hash_embedding(text: str, dim: int = 64) -> list[float]:
    """
    Deterministic lightweight embedding for reproducible local experiments.
    This is not semantically strong but is stable and cheap for ablation runs.
    """
    vec = [0.0] * dim
    for token in TOKEN_RE.findall(text.lower()):
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h, "little") % dim
        sign = 1.0 if (h[0] & 1) == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))
