import numpy as np
import random


def generate_equations(n: int, m: int, solution_type='unique'):
    if solution_type == 'unique':
        # Sanity check: make sure that it is full rank
        while True:
            mat = np.random.randn(n, m)
            x = np.random.randn(m, 1)
            b = mat @ x

            return mat, b, x

    if solution_type == 'infinite' or solution_type == 'none':
        while True:
            mat = np.random.randn(n, m)

            # generate dependency
            x = np.random.randn(m, 1)

            mat[-1, :] = 0

            permute_equations(mat, steps=n, norm=False)

            b = mat @ x
            if solution_type == 'none':
                # statistically improbable to be 0
                b[-1] = random.random()

            return mat, b, x

    raise ValueError(f'Unsupported solution type: {solution_type}')


def permute_equations(mat: np.ndarray, steps=10, norm: bool = True):
    n, _ = mat.shape

    for _ in range(steps):
        mat[random.randint(0, n - 1)] += random.random() * \
            mat[random.randint(0, n - 1)]

    normalizing_term = 1 if not norm else np.linalg.norm(mat)
    np.random.shuffle(mat)
    return mat / normalizing_term
