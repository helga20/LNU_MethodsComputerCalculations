from FiniteElementsMethod.Discretizer import Discretizer
from FiniteElementsMethod.ParabolicBoundaryProblem import ParabolicBoundaryProblem
import numpy as np
from typing import Union


class ParabolicDiscretizer:
    def __init__(self, pdeProblem: ParabolicBoundaryProblem):
        self._pdeProblem = pdeProblem
        self._nodesCoef = np.asarray([-1 / 2, 1 / 2])

        self._matCoefs = np.asarray(
            [
                # (0, h)
                [[[0, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [1, -1 / 2, 1 / 3]],
                 [[0, 0, 0], [-1, 1 / 2, 1 / 6]]],
                # (-h, 0) (0, h)
                [[[-1, -1 / 2, 1 / 6], [0, 0, 0]],
                 [[1, 1 / 2, 1 / 3], [1, -1 / 2, 1 / 3]],
                 [[0, 0, 0], [-1, 1 / 2, 1 / 6]]],
                # last row
                [[[-1, -1 / 2, 1 / 6], [0, 0, 0]],
                 [[1, 1 / 2, 1 / 3], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 0]]]
            ]
        )
        self._vecCoefs = np.asarray([
            [[[0], [1 / 2]]],
            [[[1 / 2], [1 / 2]]],
            [[[-1 / 2], [0]]]
        ])


    def buildMesh(self, n:int):
        nodes = np.linspace(0, 1, n+1)
        nodes_indices = np.arange(n+1)
        elements = np.column_stack((nodes_indices[:-1], nodes_indices[1:]))

        return nodes, elements

    def buildInitialSolution(self, mesh: Union[np.ndarray, np.ndarray]) -> np.ndarray:
        nodes, _ = mesh
        return self._pdeProblem.c(nodes.astype(float))

    def discretize(self, mesh: Union[np.ndarray, np.ndarray], t:float)-> Union[np.ndarray, np.ndarray, np.ndarray]:
        M, A = self.buildMatrix(mesh, t)
        l = self.buildVector(mesh, t)
        A, l = self.setBoundaryConditions(A, l, t)

        return M, A, l

    def postprocess(self, values: np.ndarray) -> np.ndarray:
        return values

    def buildMatrix(self, mesh: Union[np.ndarray, np.ndarray], t: float) -> Union[np.ndarray, np.ndarray]:
        problem = self._pdeProblem
        nodes, _ = mesh
        n = len(nodes)
        h = 1.0 / (n - 1)

        def fill_m_matrix_row(i):
            j = self._getCoefRowIndex(i, n)
            m = self._matCoefs[j][:, :, [2]]
            s = np.asarray([h])
            x = nodes[i] + h * self._nodesCoef
            c = np.asarray((problem.ro(x, t),)).T
            return (m * s * c).sum(axis=(1, 2))

        def fill_a_matrix_row(i):
            j = self._getCoefRowIndex(i, n)

            m = self._matCoefs[j]
            s = np.asarray([1 / h, 1, h])
            x = nodes[i] + h * self._nodesCoef



            c = np.asarray((problem.mu(x, t), [problem.beta(x, t), problem.beta(x, t)], problem.sigma(x, t))).T

            return (m * s * c).sum(axis=(1, 2))

        M = np.zeros((n, 3), dtype=np.float64)
        A = np.zeros((n, 3), dtype=np.float64)

        for i in range(n):
            M[i, :] = fill_m_matrix_row(1)
            A[i, :] = fill_a_matrix_row(1)

        return M, A


    def buildVector(self, mesh: Union[np.ndarray, np.ndarray], t: float) -> np.ndarray:
        problem = self._pdeProblem

        nodes, _ = mesh
        n = len(nodes)
        h = 1.0 / (n - 1)

        def fill_vector_row(i):
            j = self._getCoefRowIndex(i, n)
            m = self._vecCoefs[j]
            s = np.asarray([h])
            x = nodes[i] + h * self._nodesCoef
            c = np.asarray((problem.f(x, t),)).T
            return (m * s * c).sum(axis=(1, 2))

        l = np.zeros(n, dtype=np.float64)
        for i in range(n):
            l[i] = fill_vector_row(1)

        return l



    def setBoundaryConditions(self, A: np.ndarray, l: np.ndarray, t: float) -> Union[np.ndarray, np.ndarray]:
        problem = self._pdeProblem
        if problem.isDirichetCondition(problem._leftBC):
            A[0, :] = 0
            A[0, 1] = 1
            l[0] = problem.leftBC().gvalue(t)
        else:
            A[0, 1] -= problem.leftBC().qvalue(t)
            l[0] -= problem.leftBC().gvalue(t)
        if problem.isDirichetCondition(problem._rightBC):
            A[-1, :] = 0
            A[-1, 1] = 1
            l[-1] = problem.rightBC().gvalue(t)
        else:
            A[-1, 1] = - problem.rightBC().qvalue(t)
            l[-1] -= problem.rightBC().gvalue(t)

        return A, l

    def _getCoefRowIndex(self, i: int, n:int) -> int:
        if i == 0:
            return 0
        elif i == n-1:
            return 2
        else:
            return 1

    @staticmethod
    def mmul(a, b):
        r = np.zeros_like(b)
        r[0] = a[0, 1] * b[0] + a[0, 2] * b[1]
        for i in range(1, len(a) - 1):
            r[i] = a[i, 0] * b[i - 1] + a[i, 1] * b[i] + a[i, 2] * b[i + 1]
            r[-1] = a[-1, 0] * b[-2] + a[-1, 1] * b[-1]
        return r

    def check(self, A, M, l, dt, teta, t):
        if self._pdeProblem.isDirichetCondition(self._pdeProblem.leftBC()):
            A[1, 0] = 1
            A[0, 1] = 0
            M[0, 0] = (1-teta) * dt
            M[0, 1] = 0
            l[0] = self._pdeProblem.leftBC().uvalue(dt*t)
        if self._pdeProblem.isDirichetCondition(self._pdeProblem.rightBC()):
            A[1, 0] = 1
            A[0, 1] = 0
            M[0, 0] = (1-teta) * dt
            M[0, 1] = 0
            l[0] = self._pdeProblem.rightBC().uvalue(dt*t)




