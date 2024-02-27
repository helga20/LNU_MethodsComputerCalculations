from typing import Callable
from FiniteElementsMethod.BoundaryCondition import *


class BoundaryProblem:
    def __init__(self,
                 mu: Callable[[float], float],
                 beta: Callable[[float], float],
                 sigma: Callable[[float], float],
                 f: Callable[[float], float],
                 leftBoundaryCondition: BoundaryCondition,
                 rightBoundaryCondition: BoundaryCondition):

        self._mu = mu
        self._beta = beta
        self._sigma = sigma
        self._f = f
        self._leftBC = leftBoundaryCondition
        self._rightBC = rightBoundaryCondition

    def mu(self, x: float) -> float:
        return self._mu(x)

    def beta(self, x: float) -> float:
        return self._beta(x)

    def sigma(self, x: float) -> float:
        return self._sigma(x)

    def f(self, x: float) -> float:
        return self._f(x)

    def leftBC(self) -> BoundaryCondition:
        return self._leftBC

    def rightBC(self) -> BoundaryCondition:
        return self._rightBC

    @staticmethod
    def isDirichetCondition(bc: BoundaryCondition):
        return isinstance(bc, DirichletCondition)

