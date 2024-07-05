from typing import Any, Callable, NamedTuple

import numpy as np


class Timing(NamedTuple):
    mintime: float
    maxtime: float
    mean: float
    std: float

    def __str__(self) -> str:
        return (
            f"Time (mean ± std):     {self.mean:8.3f} ms ± {self.std:8.3f} ms | "
            f"Range (min - max):     {self.mintime:8.3f} ms - {self.maxtime:8.3f} ms"
        )


def repeatit(
    stmt: str | Callable[[], Any],
    *,
    setup: str | Callable[[], Any] = "pass",
    repeat: int = 16,
    number: int = 1,
) -> Timing:
    import timeit

    r = timeit.repeat(stmt=stmt, setup=setup, repeat=repeat + 1, number=number)
    return Timing(
        mintime=np.min(r),
        maxtime=np.max(r),
        mean=np.mean(r),
        std=np.std(r, ddof=1),
    )


def scipy_interp_decomp(A: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    import scipy.linalg.interpolative as sli

    k, idx, proj = sli.interp_decomp(A, eps)
    interp = sli.reconstruct_interp_matrix(idx, proj)

    return k, idx, interp


def main(
    A: np.ndarray,
    eps: float,
) -> None:
    import pytential.linalg.utils as pli

    result_scipy = repeatit(lambda: scipy_interp_decomp(A, eps))
    result_cython = repeatit(lambda: pli.interp_decomp(A, rank=None, eps=eps))

    print(f"# Benchmark: Shape {A.shape} eps {eps:.8e}")
    print("## scipy / legacy Fotran vs scipy / cython")
    print(result_scipy)
    print(result_cython)


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    n = rng.integers(200, 300)
    m = rng.integers(200, 300)
    k = rng.integers(100, min(m, n))

    for p in range(2, 16):
        A = np.zeros((n, m))
        for _ in range(k):
            a = rng.random(n)
            b = rng.random(m)
            A += np.outer(a, b)

        main(A, eps=10 ** (-p))
