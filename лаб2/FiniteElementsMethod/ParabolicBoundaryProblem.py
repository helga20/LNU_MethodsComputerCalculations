from typing import Callable
from FiniteElementsMethod.BoundaryCondition import *
import numpy as np

class ParabolicBoundaryProblem:
    def __init__(self,
                 ro: Callable[[float, float], float],
                 mu: Callable[[[float], float], float],
                 beta: Callable[[float, float], float],
                 sigma: Callable[[float, float], float],
                 f: Callable[[float, float],  float],
                 c: Callable[[float], float],
                 leftBoundaryCondition: BoundaryCondition,
                 rightBoundaryCondition: BoundaryCondition):
        self._ro = ro
        self._mu = mu
        self._beta = beta
        self._sigma = sigma
        self._f = f
        self._c = c
        self._leftBC = leftBoundaryCondition
        self._rightBC = rightBoundaryCondition

    def ro(self, x: float, t: float=0) -> float:
        return self._ro(x, t)

    def mu(self, x: float, t: float=0) -> float:
        return self._mu(x, t)

    def beta(self, x: float, t: float=0) -> float:
        return self._beta(x, t)

    def sigma(self, x: float, t: float=0) -> float:
        return self._sigma(x, t)

    def f(self, x: float, t: float=0) -> float:
        return self._f(x, t)

    def c(self, x: np.ndarray) -> np.ndarray:
        return self._c(x.astype(float))

    def leftBC(self) -> BoundaryCondition:
        return self._leftBC

    def rightBC(self) -> BoundaryCondition:
        return self._rightBC

    @staticmethod
    def isDirichetCondition(bc: BoundaryCondition):
        return isinstance(bc, DirichletCondition)

