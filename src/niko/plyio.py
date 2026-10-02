"""Minimal PLY reading (the vertex x, y, z of our own binary files and COLMAP / MegaSaM ones)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1",
              "uint8": "u1", "char": "i1", "int8": "i1", "short": "<i2", "ushort": "<u2", "int": "<i4",
              "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def read_ply_xyz(path: str | Path) -> np.ndarray:
    """x, y, z [N, 3] float64 of the vertex element (binary little-endian or ascii), finite rows only."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    end = raw.index(b"\n", end) + 1
    fmt, n, props, in_vertex = None, 0, [], False
    for line in raw[:end].decode("ascii").splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "format":
            fmt = w[1]
        elif w[0] == "element":
            in_vertex = w[1] == "vertex"
            if in_vertex:
                n = int(w[2])
        elif w[0] == "property" and in_vertex:
            props.append((w[2], _PLY_TYPES[w[1]]))
    names = [p[0] for p in props]
    if fmt == "ascii":
        rows = [r.split() for r in raw[end:].decode("ascii").split("\n")[:n]]
        a = np.array(rows, float) if n else np.zeros((0, len(names)))
        X = a[:, [names.index("x"), names.index("y"), names.index("z")]]
    elif fmt == "binary_little_endian":
        d = np.frombuffer(raw, dtype=np.dtype(props), count=n, offset=end)
        X = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    else:
        raise ValueError(f"unsupported PLY format {fmt}")
    return X[np.all(np.isfinite(X), 1)]
