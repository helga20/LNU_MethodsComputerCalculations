import numpy as np
from BoundaryProblem import BoundaryProblem


class NormCalculator:
    def __init__(self, problem: BoundaryProblem):
        self._problem = problem
        self._mu_matrix = np.asarray([[1, -1], [-1, 1]])
        self._beta_matrix = np.asarray([[-1, -1], [1, 1]])
        self._sigma_matrix = np.asarray([[2, 1], [1, 2]])

    def energy_norm(self, x_elements, alpha_values):
        n = len(x_elements)
        h = 1.0 / (n - 1)
        norm = np.zeros(n - 1, dtype=np.float64)

        for i in range(n - 1):
            center_element = (x_elements[i] + x_elements[i + 1]) / 2
            alpha_vector = [alpha_values[i], alpha_values[i + 1]]

            mu = self._problem.mu(center_element)
            beta = self._problem.beta(center_element)
            sigma = self._problem.sigma(center_element)

            norm[i] += mu * 1 / h * np.dot(np.dot(alpha_vector, self._mu_matrix), alpha_vector)
            norm[i] += beta * 1 / 2 * np.dot(np.dot(alpha_vector, self._beta_matrix), alpha_vector)
            norm[i] += sigma * h / 6 * np.dot(np.dot(alpha_vector, self._sigma_matrix), alpha_vector)

            # norm[i] = norm[i] - (self._problem.leftBoundaryCondition.qvalue(alpha_values[-1]**2)) +\
            # (self._problem.rightBoundaryCondition.qvalue(alpha_values[0] ** 2))
        #     оце ще у формулі???
        return norm

    def sobolev_norm(self, x_elements, alpha_values):
        n = len(x_elements)
        h = 1.0 / (n - 1)
        norm = np.zeros(n - 1, dtype=np.float64)

        for i in range(n - 1):
            alpha_vector = [alpha_values[i], alpha_values[i + 1]]

            norm[i] += 1 / h * np.dot(np.dot(alpha_vector, self._mu_matrix), alpha_vector)
            norm[i] += h / 6 * np.dot(np.dot(alpha_vector, self._sigma_matrix), alpha_vector)

        return norm
