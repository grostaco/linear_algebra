import numpy as np
from linear_algebra.algebra.util import inv


def schur(mat: np.ndarray, depth=1):
    # p - :depth
    # q - depth:

    n, _ = mat.shape
    if n == depth:
        return mat

    A = mat[:depth, :depth]
    B = mat[:depth, depth:]
    C = mat[depth:, :depth]
    D = mat[depth:, depth:]

    mat[depth:, depth:] = D - C @ inv(A) @ B
    mat[depth:, :depth] = 0

    return schur(mat, depth + 1)
