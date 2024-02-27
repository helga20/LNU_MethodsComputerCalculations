import numpy as np


class TridiagonalMatrixAlgorithm():

    def __init__(self, A: np.ndarray, l: np.ndarray):
        if A.shape[0] < 3:
            raise ValueError('A matrix should have at least 3 rows')
        if A.shape[1] != 3:
            raise ValueError('A matrix should be (n, 3)')
        if len(l.shape) != 1:
            raise ValueError('Vector l should be (n, 1) shape')
        if A.shape[0] != l.shape[0]:
            raise ValueError('A matrix and l vector should be same length')
        self._A = A
        self._l = l

    def solve(self) -> np.ndarray:
        A = self._A.copy()
        l = self._l.copy()
        TridiagonalMatrixAlgorithm.direct(A, l)
        TridiagonalMatrixAlgorithm.reverse(A, l)
        return l

    @staticmethod
    def direct(A: np.ndarray, l: np.ndarray) -> None:
        n = A.shape[0]
        for i in range(n-1):
            if TridiagonalMatrixAlgorithm.checkRows(A, i, i+1):
                TridiagonalMatrixAlgorithm.swapRows(A, l, i, i+1)
            a = A[i, 1]
            b = A[i+1, 0]

            l[i] /= a
            for j in range(3):
                A[i, j] /= a

            if not np.isclose(b, 0):
                l[i+1] -= l[i] * b
                for j in range(2):
                    A[i+1, j] -= A[i, j+1] * b
                # A[i+1, 2] is changed at reverse step

    @staticmethod
    def reverse(A: np.ndarray, l: np.ndarray) -> None:
        n = A.shape[0]
        l[-1] /= A[-1, 1]
        A[-1, 1] = 1
        for i in reversed(range(n-1)):
            a = A[i, 2]
            if not np.isclose(a, 0):
                l[i] -= a* l[i+1]
                A[i,2 ] = 0

    @staticmethod
    def checkRows(A: np.ndarray, i: int, j:int) -> bool:
        a = A[i, 1]
        b = A[j, 0]
        if np.isclose(a, 0) and np.isclose(b, 0):
            raise ValueError("Problem has no solution")
        return np.isclose(a, 0)

    @staticmethod
    def swapRows(A, l, i, j):
        A[[i, j]] = A[[j, i]]
        l[[i, j]] = l[[j, i]]
