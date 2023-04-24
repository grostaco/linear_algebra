import numpy as np
import numpy.typing as npt
from .lup import lup_factorization
import random


def det(mat: npt.NDArray[np.floating]) -> None | np.floating:
    if mat.shape[0] != mat.shape[1]:
        return None

    l, u, p = lup_factorization(mat)

    sign = -1 if (np.diag(p).sum() - mat.shape[0]) % 2 else 1

    return sign * np.prod(np.diag(l)) * np.prod(np.diag(u))


def rank(mat: np.ndarray, tol=1e-6) -> int:
    try:
        l, u, _ = lup_factorization(mat.copy())
    except ValueError:
        return 0
    return sum((abs(np.diag(u)) > tol))


def inv(mat: npt.NDArray[np.floating], tol: float = 1e-9) -> npt.NDArray[np.floating]:
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

        if abs(mat[i, i]) < tol:
            raise ValueError(f'Matrix is not invertible')

        factor = mat[i+1:n, i] / mat[i, i]
        mat[i+1:, :] -= mat[i, :] * factor[:, None]

    for i in range(m-1, 0, -1):
        if abs(mat[i, i]) < tol:
            raise ValueError(f'Matrix is not invertible')
        factor = mat[:i, i] / mat[i, i]
        mat[:i, :] -= mat[i, :] * factor[:, None]

    factor = np.diag(mat)[:, None]

    if (abs(factor) < tol).any():
        raise ValueError(f'Matrix is not invertible')

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


def back_substitution(mat, b):
    n = len(b) - 1
    x = np.zeros_like(b)
    x[n] = b[n] / mat[n, n]
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(n, i, -1):
            s = s - np.dot(mat[i, j], x[j])

        x[i] = s / mat[i, i]
        print(x[i])

    return x
