"""On-disk feature cache.

AST needs about 0.8 s per clip on a laptop CPU, so every feature vector is computed once and stored.
Keys are (featurizer id, clip, variant):
  - featurizer id changes whenever the features would change (e.g. another AST checkpoint),
  - variant is "orig" for the untouched clip and, from v0.2, a hash of the MR spec for transformed clips.
SQLite is used because it is in the standard library, survives interruption (each batch is one
transaction) and handles tens of thousands of rows without thousands of small files.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    featurizer TEXT NOT NULL,
    clip       TEXT NOT NULL,
    variant    TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vec        BLOB NOT NULL,
    PRIMARY KEY (featurizer, clip, variant)
)
"""


class FeatureCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute(_SCHEMA)
        self._db.commit()

    def get_all(self, featurizer: str, variant: str = "orig") -> dict[str, np.ndarray]:
        rows = self._db.execute(
            "SELECT clip, dim, vec FROM features WHERE featurizer = ? AND variant = ?", (featurizer, variant)
        )
        return {clip: np.frombuffer(vec, dtype=np.float32, count=dim).copy() for clip, dim, vec in rows}

    def missing(self, featurizer: str, clips: Iterable[str], variant: str = "orig") -> list[str]:
        have = {
            c for (c,) in self._db.execute(
                "SELECT clip FROM features WHERE featurizer = ? AND variant = ?", (featurizer, variant)
            )
        }
        return [c for c in clips if c not in have]

    def put_many(self, featurizer: str, vectors: Mapping[str, np.ndarray], variant: str = "orig") -> None:
        rows = []
        for clip, v in vectors.items():
            v = np.asarray(v, dtype=np.float32).ravel()
            rows.append((featurizer, clip, variant, v.size, v.tobytes()))
        with self._db:  # one transaction per batch
            self._db.executemany("INSERT OR REPLACE INTO features VALUES (?, ?, ?, ?, ?)", rows)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> FeatureCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
