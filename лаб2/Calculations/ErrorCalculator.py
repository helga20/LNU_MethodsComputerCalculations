import numpy as np
from FiniteElementsMethod import BoundaryProblem
from FiniteElementsMethod.ParabolicBoundaryProblem import ParabolicBoundaryProblem


class ErrorCalculator:
    def __init__(self, problem: ParabolicBoundaryProblem):
        self._problem = problem
        self._beta_vector = np.asarray([-1, 1])
        self._sigma_vector = np.asarray([1, 1])

    def dirichlet_error(self, x_elements, alpha, t):
        n = len(x_elements)
        h = 1.0 / (n - 1)
        errors = np.zeros(n - 1, dtype=np.float64)

        for i in range(n-1):
            center_element = (x_elements[i] + x_elements[i + 1]) / 2
            alpha_vector = [alpha[i], alpha[i + 1]]

            mu = self._problem.mu(center_element)
            beta = self._problem.beta(center_element)
            sigma = self._problem.sigma(center_element)
            f = self._problem.f(center_element, t)

            errors[i] += beta * 1 / h * np.dot(self._beta_vector, alpha_vector)
            errors[i] += sigma * 1 / 2 * np.dot(self._sigma_vector, alpha_vector)
            errors[i] = pow((f - errors[i]), 2) * h ** 2
            denominator = ((6 * mu) / (5 * h)) * (10 + (h ** 2 * sigma / mu))
            errors[i] = errors[i] / denominator
        return np.sum(errors)

    def neumann_error(self, x_elements, alpha, t):
        n = len(x_elements)
        h = 1.0 / (n - 1)
        errors = np.zeros(n - 1, dtype=np.float64)

        for i in range(n - 1):
            center_element = (x_elements[i] + x_elements[i + 1]) / 2
            alpha_vector = [alpha[i], alpha[i + 1]]

            mu = self._problem.mu(center_element)
            beta = self._problem.beta(center_element)
            sigma = self._problem.sigma(center_element)
            f = self._problem.f(center_element, t)

            errors[i] += beta * 1 / h * np.dot(self._beta_vector, alpha_vector)
            errors[i] += sigma * 1 / 2 * np.dot(self._sigma_vector, alpha_vector)
            errors[i] = pow((f - errors[i]), 2) * h ** 2
            denominator = (4 * mu / (3 * h)) * (12 + ((h ** 2 * sigma) / mu))
            errors[i] = errors[i] / denominator

        return np.sum(errors)
