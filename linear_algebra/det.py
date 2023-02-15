import numpy as np
import numpy.typing as npt
from .lup import lup_factorization


def det(mat: npt.NDArray[np.floating]) -> None | np.floating:
    if mat.shape[0] != mat.shape[1]:
        return None

    l, u, p = lup_factorization(mat)

    sign = -1 if (np.diag(p).sum() - mat.shape[0]) % 2 else 1

    return sign * np.prod(np.diag(l)) * np.prod(np.diag(u))
