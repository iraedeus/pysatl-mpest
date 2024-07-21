""" The module in which the L moments method is presented """
import numpy as np

from mpest import Distribution, Samples
from mpest.em.methods.abstract_method import AMethod
from mpest.em.methods.types import AExpectation, AMaximization
from mpest.mixture_distribution import MixtureDistribution
from mpest.problem import Problem
from mpest.utils import ResultWithError


class LMomentsMethod(AMethod[tuple]):
    """
    Class which represents L moments method

    :param e_step: The object of the internal subclass in which the E step is performed
    :param m_step: The object of the internal subclass in which the M step is performed
    """

    class IndicatorEStep(AExpectation):
        """
        Class which represents method for performing E step in L moments method.
        """

        def calc_indicators(self, problem: Problem):
            samples, mixture = problem.samples, problem.distributions
            priors = [mixture[i].prior_probability for i in range(len(mixture))]
            k, m = len(mixture), len(samples)

            z = np.zeros([k, m], dtype=float)

            for j in range(k):
                z_j = np.zeros([1, m])
                for i in range(m):
                    x_i = samples[i]
                    d_j = mixture[j]

                    denominator = np.sum(
                        [
                            priors[_j] * d.model.pdf(x_i, d.params)
                            for _j, d in enumerate(mixture)
                        ]
                    )
                    numerator = priors[j] * d_j.model.pdf(x_i, d_j.params)
                    z_ji = numerator / denominator
                    z_j[:, i] = z_ji

                z[j] = z_j

            return z

        def update_priors(self, problem: Problem, indicators):
            samples, mixture = problem.samples, problem.distributions
            k, m = len(mixture), len(samples)

            new_priors = []
            for j in range(k):
                p_j = np.sum([indicators[j][i] for i in range(m)])
                new_priors.append(p_j / m)

            return new_priors

        def step(self, problem: Problem) -> tuple:
            """
            A function that performs E step

            :param problem: Object of class Problem, which contains samples and mixture.
            :return: ???
            """

            indicators = self.calc_indicators(problem)
            new_priors = self.update_priors(problem, indicators)
            return problem, new_priors, indicators

    class MStep(AMaximization):
        """
        Class which calculate new params using matrix with indicator from E step.
        """

        def calculate_m1(self, problem: Problem, indicators):
            samples, mixture = problem.samples, problem.distributions
            ordered_samples = np.sort(samples)
            k, m = len(mixture), len(samples)

            moments = []
            for j in range(k):
                denominator = np.sum([indicators[j][i] for i in range(m)])
                numerator = np.sum(
                    [indicators[j][i] * ordered_samples[i] for i in range(m)]
                )
                moments.append(numerator / denominator)

            return moments

        def calculate_m2(self, problem: Problem, indicators, m1):
            samples, mixture = problem.samples, problem.distributions
            ordered_samples = np.sort(samples)
            k, m = len(mixture), len(samples)

            moments = []
            for j in range(k):
                sum_z_ji = np.sum([indicators[j][i] for i in range(m)])
                denominator = sum_z_ji**2 - sum_z_ji

                numerator = 0
                for i in range(m):
                    sum_z_less = np.sum([indicators[j][l] for l in range(i)])
                    numerator += indicators[j][i] * ordered_samples[i] * sum_z_less

                m2_j = ((2 * numerator) / denominator) - m1[j]
                moments.append(m2_j)

            return moments

        def step(self, e_result: tuple) -> ResultWithError[MixtureDistribution]:
            """
            A function that performs E step

            :param e_result: ???
            """

            problem, new_priors, indicators = e_result
            m1 = self.calculate_m1(problem, indicators)
            m2 = self.calculate_m2(problem, indicators, m1)

            mixture = problem.distributions
            new_distributions = []

            for j, d in enumerate(mixture):
                new_params = d.model.calc_params(m1[j], m2[j])
                new_d = Distribution(
                    d.model, d.model.params_convert_to_model(new_params)
                )
                new_distributions.append(new_d)

            new_mixture = MixtureDistribution.from_distributions(
                new_distributions, new_priors
            )
            return ResultWithError(new_mixture)

    def __init__(self, e_step: AExpectation, m_step: AMaximization) -> None:
        """
        Object constructor

        :param e_step: The object of the internal subclass in which the E step is performed
        :param m_step: The object of the internal subclass in which the M step is performed
        """

        self.e_step = e_step
        self.m_step = m_step

    @property
    def name(self) -> str:
        """name getter"""

        return "Likelihood"

    def step(self, problem: Problem) -> ResultWithError[MixtureDistribution]:
        """
        Function which performs E and M steps

        :param problem: Object of class Problem, which contains samples and mixture.
        """

        e_result = self.e_step.step(problem)
        return self.m_step.step(e_result)
