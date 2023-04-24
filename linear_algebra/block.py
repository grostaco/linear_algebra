from linear_algebra.util import inv 
import numpy as np

def block_elim(mat: np.array, depth=1) -> np.array:
    # p - :depth
    # q - depth:
    n, _ = mat.shape
    if depth == n:
        return mat
    A = mat[:depth, :depth]
    B = mat[:depth, depth:]
    C = mat[depth:, :depth]
    D = mat[depth:, depth:]
    
    mat[depth:, depth:] = D - C @ inv(A) @ B
    mat[depth:, :depth] = 0

    return block_elim(mat, depth+1)