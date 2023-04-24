import numpy as np
import numpy.typing as npt
from .lup import lup_factorization
import sys
import random


def det(mat: npt.NDArray[np.floating]) -> None | np.floating:
    if mat.shape[0] != mat.shape[1]:
        return None

    l, u, p = lup_factorization(mat)

    sign = -1 if (np.diag(p).sum() - mat.shape[0]) % 2 else 1

    return sign * np.prod(np.diag(l)) * np.prod(np.diag(u))


# def rank(mat: npt.NDArray[np.floating]):
#     l, u, _ = lup_factorization(mat)

#     return min(np.diag(l).sum(), np.diag(u).sum())

def inv(mat: npt.NDArray[np.floating], tol: float = sys.float_info.epsilon) -> npt.NDArray[np.floating]:
    m, n = mat.shape

    if m != n:
        raise ValueError("Non-square matrices are not invertible")

    mat = np.hstack((mat, np.eye(m, n)))

    for i in range(m-1):
        if mat[i][i] == 0:
            h = mat[:, i].argmax()

            if h == i:
                continue

            if abs(mat[h][i]) < tol:
                raise ValueError(
                    f"Matrix is degenerate. Entry [{h}, {i}] for partial pivoting is too small")

            mat[[h, i]] = mat[[i, h]]

        factor = mat[i+1:n, i] / mat[i, i]
        mat[i+1:, :] -= mat[i, :] * factor[:, None]

    for i in range(m-1, 0, -1):
        factor = mat[:i, i] / mat[i, i]
        mat[:i, :] -= mat[i, :] * factor[:, None]

    factor = np.diag(mat)[:, None]
    mat /= factor

    return mat[:, n:]


def is_full(mat: npt.NDArray[np.floating]):
    return det(mat) != 0


def permute_equations(mat: np.ndarray, steps=10):
    n, _ = mat.shape

    for _ in range(steps):
        mat[random.randint(0, n - 1)] += random.random() * \
            mat[random.randint(0, n - 1)]

    return mat / np.linalg.norm(mat)


def rank(mat: np.ndarray) -> int:
    _, u, _ = lup_factorization(mat)
    return np.count_nonzero(u)
