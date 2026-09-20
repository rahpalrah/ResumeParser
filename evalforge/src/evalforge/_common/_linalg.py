# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Dependency-free numerical kernels.

Everything EvalForge needs mathematically is implemented here in pure Python so
the package installs and runs with the standard library alone. If NumPy happens
to be importable it is *not* required; these routines are written to be exact
and deterministic rather than maximally fast, which also makes evaluation runs
bit-for-bit reproducible across machines.
"""

import math
from typing import Iterable, List, Sequence, Tuple

__all__ = [
    "dot",
    "norm",
    "l2_normalize",
    "cosine",
    "add",
    "sub",
    "scale",
    "centroid",
    "mean",
    "variance",
    "stdev",
    "quantile",
    "softmax",
    "sigmoid",
    "matmul",
    "transpose",
    "identity",
    "jacobi_eigh",
    "gram_matrix",
    "normalized_laplacian",
    "wasserstein_1",
    "solve_linear",
    "clamp",
    "normal_ppf",
]

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]

_EPS = 1e-12


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return low if value < low else (high if value > high else value)


def dot(a: Vector, b: Vector) -> float:
    """Inner product of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    """Euclidean norm."""
    return math.sqrt(max(0.0, math.fsum(x * x for x in a)))


def l2_normalize(a: Vector) -> List[float]:
    """Return ``a`` scaled to unit length (zero vectors are returned as-is)."""
    n = norm(a)
    if n < _EPS:
        return list(a)
    return [x / n for x in a]


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity, clamped to ``[-1, 1]``."""
    na, nb = norm(a), norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    return clamp(dot(a, b) / (na * nb), -1.0, 1.0)


def add(a: Vector, b: Vector) -> List[float]:
    """Element-wise sum."""
    return [x + y for x, y in zip(a, b)]


def sub(a: Vector, b: Vector) -> List[float]:
    """Element-wise difference."""
    return [x - y for x, y in zip(a, b)]


def scale(a: Vector, k: float) -> List[float]:
    """Multiply every component by ``k``."""
    return [x * k for x in a]


def centroid(vectors: Sequence[Vector]) -> List[float]:
    """Arithmetic mean of a set of equal-length vectors."""
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            out[i] += v[i]
    return [x / len(vectors) for x in out]


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean; ``0.0`` for an empty sequence."""
    vals = list(values)
    if not vals:
        return 0.0
    return math.fsum(vals) / len(vals)


def variance(values: Iterable[float], ddof: int = 0) -> float:
    """Variance with ``ddof`` degrees of freedom removed."""
    vals = list(values)
    n = len(vals)
    if n - ddof <= 0:
        return 0.0
    mu = mean(vals)
    return math.fsum((v - mu) ** 2 for v in vals) / (n - ddof)


def stdev(values: Iterable[float], ddof: int = 0) -> float:
    """Standard deviation with ``ddof`` degrees of freedom removed."""
    return math.sqrt(variance(values, ddof))


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated empirical quantile of ``values`` at level ``q``."""
    if not values:
        return 0.0
    q = clamp(q, 0.0, 1.0)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def softmax(values: Sequence[float], temperature: float = 1.0) -> List[float]:
    """Numerically stable softmax."""
    if not values:
        return []
    t = max(temperature, _EPS)
    top = max(values)
    exps = [math.exp((v - top) / t) for v in values]
    total = math.fsum(exps)
    if total < _EPS:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]


def sigmoid(x: float) -> float:
    """Logistic function, overflow safe."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def transpose(m: Matrix) -> List[List[float]]:
    """Matrix transpose."""
    if not m:
        return []
    return [list(col) for col in zip(*m)]


def matmul(a: Matrix, b: Matrix) -> List[List[float]]:
    """Naive matrix product."""
    if not a or not b:
        return []
    bt = transpose(b)
    return [[dot(row, col) for col in bt] for row in a]


def identity(n: int) -> List[List[float]]:
    """``n x n`` identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def gram_matrix(vectors: Sequence[Vector]) -> List[List[float]]:
    """Pairwise cosine-similarity matrix."""
    n = len(vectors)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        out[i][i] = 1.0
        for j in range(i + 1, n):
            s = cosine(vectors[i], vectors[j])
            out[i][j] = s
            out[j][i] = s
    return out


def normalized_laplacian(weights: Matrix) -> List[List[float]]:
    """Symmetric normalized Laplacian ``L = I - D^-1/2 W D^-1/2``.

    Negative affinities are clipped to zero so ``W`` is a valid non-negative
    similarity matrix.
    """
    n = len(weights)
    w = [[max(0.0, weights[i][j]) if i != j else 0.0 for j in range(n)] for i in range(n)]
    deg = [math.fsum(row) for row in w]
    out = identity(n)
    for i in range(n):
        for j in range(n):
            if deg[i] > _EPS and deg[j] > _EPS:
                out[i][j] -= w[i][j] / math.sqrt(deg[i] * deg[j])
    return out


def jacobi_eigh(
    matrix: Matrix, max_sweeps: int = 100, tol: float = 1e-10
) -> Tuple[List[float], List[List[float]]]:
    """Eigendecomposition of a real symmetric matrix via cyclic Jacobi rotations.

    :return: ``(eigenvalues, eigenvectors)`` sorted ascending by eigenvalue.
        ``eigenvectors[k]`` is the eigenvector for ``eigenvalues[k]``.
    """
    n = len(matrix)
    if n == 0:
        return [], []
    a = [list(row) for row in matrix]
    v = identity(n)

    for _ in range(max_sweeps):
        off = math.sqrt(math.fsum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < tol:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq

    eigenvalues = [a[i][i] for i in range(n)]
    eigenvectors = transpose(v)
    order = sorted(range(n), key=lambda i: eigenvalues[i])
    return [eigenvalues[i] for i in order], [eigenvectors[i] for i in order]


def solve_linear(a: Matrix, b: Vector, ridge: float = 0.0) -> List[float]:
    """Solve ``(A + ridge*I) x = b`` by Gaussian elimination with partial pivoting."""
    n = len(a)
    if n == 0:
        return []
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for i in range(n):
        m[i][i] += ridge
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < _EPS:
            continue
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] / pv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > _EPS else 0.0 for i in range(n)]


def wasserstein_1(x: Sequence[float], y: Sequence[float], grid: int = 256) -> float:
    """1-D Wasserstein-1 distance between two empirical samples.

    Computed as the mean absolute difference of the two quantile functions,
    evaluated on a uniform grid of ``grid`` probability levels. Equivalent to
    the integral ``∫|F_x^-1(u) - F_y^-1(u)| du``.
    """
    if not x or not y:
        return 0.0
    xs, ys = sorted(x), sorted(y)
    total = 0.0
    for k in range(grid):
        u = (k + 0.5) / grid
        total += abs(quantile(xs, u) - quantile(ys, u))
    return total / grid


def normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Accurate to roughly 1.15e-9 in relative terms over the open unit interval,
    which is far beyond what threshold selection here requires.
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)

    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )
