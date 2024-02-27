import numpy as np
from FiniteElementsMethod.BoundaryProblem import *


class Calculations:
    def __init__(self, problem: BoundaryProblem):
        self._problem = problem
        self._beta_array = np.asarray([-1, 1])
        self._sigma_array = np.asarray([1, 1])

    def r_calculate(self, x_elements, alpha):
        n = len(x_elements)
        h = 1.0 / (n - 1)
        r_value = np.zeros(n-1, dtype=np.float64)

        for i in range(n-1):
            center_element = (x_elements[i] + x_elements[i+1]) / 2
            alpha_vector = [alpha[i], alpha[i+1]]

            beta = self._problem.beta(center_element)
            sigma = self._problem.sigma(center_element)
            f = self._problem.f(center_element)

            r_value[i] += beta * 1 / h * np.dot(self._beta_array, alpha_vector)
            r_value[i] += sigma * 1 / 2 * np.dot(self._sigma_array, alpha_vector)
            r_value[i] = (f - r_value[i])
            r_value[i] = np.power(r_value[i], 2) * h * h

        return np.sum(r_value)

    def jump_calculate(self, x_elements, alpha):
        n = len(x_elements)
        h = 1.0 / (n - 1)
        jump = np.zeros(n-1, dtype=np.float64)
        to_add_left = 0.0
        to_add_right = 0.0
        for i in range(1, n - 1):
            center_element = (x_elements[i] + x_elements[i + 1]) / 2
            center_element_left = (x_elements[i-1] + x_elements[i]) / 2

            alpha_vector = [alpha[i], alpha[i + 1]]
            alpha_vector_left = [alpha[i-1], alpha[i]]

            mu_right = self._problem.mu(center_element)
            mu_left = self._problem.mu(center_element_left)

            jump[i] += mu_right * 1/h * np.dot(self._beta_array, alpha_vector)
            jump[i] -= mu_left * 1/2 * np.dot(self._sigma_array, alpha_vector_left)
            jump[i] = pow(jump[i], 2) * h

        if not BoundaryProblem.isDirichetCondition(self._problem.leftBC()):
            to_add_left = h ** 2 * pow((self._problem.rightBC().gvalue(x_elements[0]) - self._problem.rightBC().qvalue(x_elements[0]) * alpha[0]) - \
                     self._problem.mu(x_elements[0:2].mean()) * np.dot(alpha[0:2], self._beta_array), 2)

        if not BoundaryProblem.isDirichetCondition(self._problem.rightBC()):
            to_add_right = h ** 2 * pow((self._problem.rightBC().gvalue(x_elements[-1]) - self._problem.rightBC().qvalue(x_elements[-1]) * alpha[-1]) - \
                     self._problem.mu(x_elements[0:2].mean()) * np.dot(alpha[0:2], self._beta_array), 2)
        return np.sum(jump) + to_add_left + to_add_right



    @staticmethod
    def q_calculate(norm_array, i):

        chys = (norm_array[i] - norm_array[i - 1]) / (norm_array[i - 1] - norm_array[i - 2])
        denominator = (norm_array[i - 1] - norm_array[i - 2]) / (norm_array[i - 2] - norm_array[i - 3])

        q = np.log(chys) / np.log(denominator)

        return q

