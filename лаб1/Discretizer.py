import numpy as np, pandas as pd
from typing import Union
from BoundaryProblem import BoundaryProblem
from BoundaryCondition import BoundaryCondition


class Discretizer:
    def __init__(self, problem: BoundaryProblem):
        self._pdeProblem = problem

    def do_discretization(self, n: int):
        x_elements = np.linspace(0, 1, n)
        K = Discretizer.build_matrix_K(self._pdeProblem, x_elements)
        l = Discretizer.build_free_elements_vector(self._pdeProblem, x_elements)
        return x_elements, K, l

    @staticmethod
    def build_matrix_K(problem: BoundaryProblem, x_elements: np.ndarray) -> np.ndarray:
        n = len(x_elements)
        h = 1.0 / (n - 1)
        mu = problem.mu
        beta = problem.beta
        sigma = problem.sigma
        K = np.zeros((n, 3), dtype=np.float64)

        if problem.isDirichetCondition(problem.leftBC()):
            K[0, 1] = 1
        else:
            x_i_plus = (x_elements[0] + x_elements[1]) / 2
            K[0, 1] = mu(x_i_plus) / h - beta(x_i_plus) / 2 + sigma(x_i_plus) * h / 3 - problem.leftBC().qvalue(x_i_plus)
            K[0, 2] = -mu(x_i_plus) / h + beta(x_i_plus) / 2 + sigma(x_i_plus) * h / 6

        for i in range(1, n-1):
            x_i_minus = (x_elements[i-1] + x_elements[i]) / 2
            x_i_plus = (x_elements[i] + x_elements[i+1]) / 2

            K[i, 0] = -mu(x_i_minus) / h - beta(x_i_minus) / 2 + sigma(x_i_minus) * h / 6     #нижня
            K[i, 1] = (mu(x_i_minus) / h + beta(x_i_minus) / 2 + sigma(x_i_minus) * h / 3) + \
                       (mu(x_i_plus) / h - beta(x_i_plus) / 2 + sigma(x_i_plus) * h / 3)
            K[i, 2] = -mu(x_i_plus) / h + beta(x_i_plus) / 2 + sigma(x_i_plus) * h / 6 #верхня

        if problem.isDirichetCondition(problem.rightBC()):
            K[-1, 1] = 1
        else:
            x_i_minus = x_elements[-1] - h / 2
            K[-1, 0] = -mu(x_i_minus) / h - beta(x_i_minus) / 2 + sigma(x_i_minus) * h/6
            K[-1, 1] = mu(x_i_minus) / h + beta(x_i_minus) / 2 + sigma(x_i_minus) * h / 3 - problem.rightBC().qvalue(x_i_minus)

        return K

    @staticmethod
    def build_free_elements_vector(problem: BoundaryProblem, nodes: np.ndarray) -> np.ndarray:
        n = len(nodes)
        h = 1.0 / (n-1)
        f = problem.f
        l = np.zeros(n, dtype=np.float64)

        if problem.isDirichetCondition(problem.leftBC()):
            l[0] = float(problem.leftBC().uvalue(nodes[0]))
        else:
            xr = nodes[0] + h/2
            l[0] = h * f(xr) / 2 - problem.leftBC().gvalue(xr)
        for i in range(1, n-1):
            xl = nodes[i] - h / 2
            xr = nodes[i] + h /2
            l[i] = h * (f(xl) + f(xr)) / 2

        if problem.isDirichetCondition(problem.rightBC()):
            l[-1] = problem.rightBC().uvalue(nodes[-1])
        else:
            xl = nodes[-1] - h /2
            l[-1] = h * f(xl) / 2 - float(problem.rightBC().gvalue(xl))
        return l