from typing import Callable


class BoundaryCondition:
    def __init__(self):
        pass

    def uvalue(self, x) -> float:
        return 0

    def gvalue(self, x) -> float:
        return 0

    def qvalue(self, x) -> float:
        return 0


class DirichletCondition(BoundaryCondition):
    def __init__(self, uvalue: Callable[[float], float]):
        self._uvalue = uvalue

    def uvalue(self, x) -> float:
        return self._uvalue(x)


class RobinCondition(BoundaryCondition):
    def __init__(self, g: Callable[[float], float], q: Callable[[float], float]):
        self._g = g
        self._q = q

    def gvalue(self, x) -> float:
        return self._g(x)

    def qvalue(self, x) -> float:
        return self._q(x)
