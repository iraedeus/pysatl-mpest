""" The module in which the L moments method is presented """
import numpy as np
from scipy.stats import dirichlet

from mpest.distribution import Distribution
from mpest.em.methods.abstract_steps import AExpectation, AMaximization
from mpest.mixture_distribution import MixtureDistribution
from mpest.problem import Problem
from mpest.utils import ResultWithError

EResult = tuple[Problem, list[float], np.ndarray]


class IndicatorEStep(AExpectation[EResult]):
    """
    Class which represents method for performing E step in L moments method.
    """

    def __init__(self):
        """
        Object constructor
        """

        self.indicators = np.array([])

    def init_indicators(self, problem: Problem) -> None:
        """
        A function that initializes a matrix with indicators

        :param problem: Object of class Problem, which contains samples and mixture.
        """

        k, m = len(problem.distributions), len(problem.samples)
        self.indicators = np.transpose(dirichlet.rvs([1 for _ in range(k)], m))

    def calc_indicators(self, problem: Problem):
        """
        A function that recalculates the matrix with indicators.

        :param problem: Object of class Problem, which contains samples and mixture.
        """

        samples, mixture = np.sort(problem.samples), problem.distributions
        priors = np.array([dist.prior_probability for dist in mixture])
        k, m = len(mixture), len(samples)

        z = np.zeros((k, m), dtype=float)

        pdf_values = np.zeros((k, m), dtype=float)
        for j, d_j in enumerate(mixture):
            pdf_values[j] = [d_j.model.pdf(samples[i], d_j.params) for i in range(m)]

        denominators = np.sum(priors[:, np.newaxis] * pdf_values, axis=0)

        for j in range(k):
            numerators = priors[j] * pdf_values[j]
            z[j] = numerators / denominators

        self.indicators = z

    def update_priors(self, problem: Problem) -> list[float]:
        """
        A function that recalculates the list with prior probabilities.

        :param problem: Object of class Problem, which contains samples and mixture.
        :return: List with prior probabilities
        """

        samples, mixture = problem.samples, problem.distributions
        k, m = len(mixture), len(samples)

        new_priors = []
        for j in range(k):
            p_j = np.sum([self.indicators[j][i] for i in range(m)])
            new_priors.append(p_j / m)

        return new_priors

    def step(self, problem: Problem) -> EResult:
        """
        A function that performs E step

        :param problem: Object of class Problem, which contains samples and mixture.
        :return: Tuple with problem, new_priors and indicators.
        """

        if not hasattr(self, "indicators"):
            self.init_indicators(problem)
        else:
            self.calc_indicators(problem)

        new_priors = self.update_priors(problem)
        new_problem = Problem(
            problem.samples,
            MixtureDistribution.from_distributions(problem.distributions, new_priors),
        )
        return new_problem, new_priors, self.indicators


class MStep(AMaximization[EResult]):
    """
    Class which calculate new params using matrix with indicator from E step.
    """

    def calculate_m1(self, problem: Problem, indicators: np.ndarray) -> list[float]:
        """
        A function that calculates the list of first moments of each distribution.

        :param problem: Object of class Problem, which contains samples and mixture.
        :param indicators: Matrix with indicators
        :return: List with first moments
        """

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

    def calculate_m2(
        self, problem: Problem, indicators: np.ndarray, m1: list[float]
    ) -> list[float]:
        """
        A function that calculates the list of second moments of each distribution.

        :param problem: Object of class Problem, which contains samples and mixture.
        :param indicators: Matrix with indicators
        :return: List with second moments
        """

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

    def step(self, e_result: EResult) -> ResultWithError[MixtureDistribution]:
        """
        A function that performs E step

        :param e_result: Tuple with problem, new_priors and indicators.
        """

        problem, new_priors, indicators = e_result
        m1 = self.calculate_m1(problem, indicators)
        m2 = self.calculate_m2(problem, indicators, m1)

        mixture = problem.distributions
        new_distributions = []

        for j, d in enumerate(mixture):
            moments = [m1[j], m2[j]]
            new_params = d.model.calc_params(moments)
            new_d = Distribution(d.model, d.model.params_convert_to_model(new_params))
            new_distributions.append(new_d)

        new_mixture = MixtureDistribution.from_distributions(
            new_distributions, new_priors
        )
        return ResultWithError(new_mixture)
