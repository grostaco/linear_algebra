import numpy as np
from typing import Iterator
from .util import det


def cramer(mat: np.ndarray, aug: np.ndarray) -> Iterator[np.floating]:
    numerator = det(mat.copy())

    if numerator is None:
        raise ValueError("The matrix has no determinant")

    for i in range(mat.shape[0]):
        augmented = mat.copy()
        augmented[:, i] = aug

        if x := det(augmented):
            yield x/numerator
