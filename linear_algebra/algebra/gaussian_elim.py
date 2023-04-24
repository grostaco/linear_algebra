import numpy as np
import numpy.typing as npt
from typing import TypeVar
import sys

T = TypeVar('T', bound=np.floating)


def gaussian_elim(mat: npt.NDArray[T], tol: float = sys.float_info.epsilon) -> npt.NDArray[T]:
    mat = mat.copy()
    m, _ = mat.shape

    for i in range(m-1):
        # partial pivoting
        if mat[i][i] == 0:
            h = mat[:, i].argmax()

            if h == i:
                continue

            if abs(mat[h][i]) < tol:
                raise ValueError(
                    f"Matrix is degenerate. Entry [{h}, {i}] for partial pivoting is too small")

            mat[[h, i]] = mat[[i, h]]

        # reduction
        mat[i+1:, i] = mat[i+1:, i] / mat[i, i]
        mat[i+1:, i:] -= mat[i, i:] * mat[i + 1:, i][:, None]

    return mat
