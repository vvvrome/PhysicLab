import numpy as np

def reduccion(A, B):
    A = A.astype(np.float64, copy=True)
    B = B.astype(np.float64, copy=True)
    n = A.shape[0]

    for k in range(n-1):
        i_max = np.argmax(np.abs(A[k:n, k])) + k
        if np.isclose(A[i_max, k], 0):
            raise ValueError("Matrix is singular.")

        if i_max != k:
            A[[k, i_max], :] = A[[i_max, k], :]
            B[[k, i_max]] = B[[i_max, k]]

        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            B[i] -= factor * B[k]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (B[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        return x