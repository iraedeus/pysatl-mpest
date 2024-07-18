"""TODO"""
from itertools import permutations

import numpy as np

from mpest import MixtureDistribution
from mpest.em import EM
from mpest.em.breakpointers import ParamDifferBreakpointer, StepCountBreakpointer
from mpest.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from mpest.em.methods.likelihood_method import LikelihoodMethod
from mpest.optimizers import ALL_OPTIMIZERS
from mpest.problem import Problem, Result
from mpest.utils import ResultWithError


def run_test(problem: Problem, deviation: float) -> list[Result]:
    """TODO"""
    result = []
    for optimizer in ALL_OPTIMIZERS:
        method = method = LikelihoodMethod(
            LikelihoodMethod.BayesEStep(), LikelihoodMethod.LikelihoodMStep(optimizer)
        )
        em_algo = EM(
            StepCountBreakpointer() + ParamDifferBreakpointer(deviation=deviation),
            FiniteChecker() + PriorProbabilityThresholdChecker(),
            method,
        )

        result.append(em_algo.solve(problem=problem))

    return result


def check_for_params_error_tolerance(
    results: list[ResultWithError[MixtureDistribution]],
    base_mixture: MixtureDistribution,
    expected_error: float,
) -> bool:
    """A function to check for compliance with an expected parameters error"""

    def absolute_diff_params(a: MixtureDistribution, b: MixtureDistribution):
        """Metric which checks absolute differ of params of gotten distribution mixtures"""

        a_p, b_p = ([d.params for d in ld] for ld in (a, b))

        return min(
            sum(np.sum(np.abs(x - y)) for x, y in zip(a_p, _b_p))
            for _b_p in permutations(b_p)
        )

    for result in results:
        assert result.error is None
        actual_error = absolute_diff_params(result.content, base_mixture)
        print(actual_error)
        if actual_error <= expected_error:
            return True
    return False


def check_for_priors_error_tolerance(
    results: list[ResultWithError[MixtureDistribution]],
    base_mixture: MixtureDistribution,
    expected_error: float,
) -> bool:
    """A function to check for compliance with an expected prior probabilities error"""

    def absolute_diff_priors(a: MixtureDistribution, b: MixtureDistribution):
        """
        Metric which checks absolute differ of prior probabilities of gotten distribution mixtures
        """

        a_p, b_p = ([d.prior_probability for d in ld] for ld in (a, b))

        return min(
            sum(np.sum(np.abs(x - y)) for x, y in zip(a_p, _b_p))
            for _b_p in permutations(b_p)
        )

    for result in results:
        assert result.error is None
        actual_error = absolute_diff_priors(result.content, base_mixture)
        print(actual_error)
        if actual_error <= expected_error:
            return True
    return False
