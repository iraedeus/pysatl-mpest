"""TODO"""
from itertools import permutations

import numpy as np

from mpest import MixtureDistribution
from mpest.problem import Problem, Result
from mpest.em.breakpointers import StepCountBreakpointer, ParamDifferBreakpointer
from mpest.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from mpest.optimizers import ALL_OPTIMIZERS
from mpest.em import EM
from mpest.utils import ResultWithError


def run_test(problem: Problem, deviation: float) -> list[Result]:
    """TODO"""
    result = []
    for optimizer in ALL_OPTIMIZERS:
        em_algo = EM(
            StepCountBreakpointer() + ParamDifferBreakpointer(deviation=deviation),
            FiniteChecker() + PriorProbabilityThresholdChecker(),
            optimizer,
        )

        result.append(em_algo.solve(problem=problem))

    return result


def absolute_diff_params(a: MixtureDistribution, b: MixtureDistribution):
    """Metric which checks absolute differ of gotten distribution mixtures"""

    a_p, b_p = ([d.params for d in ld] for ld in (a, b))

    return min(
        sum(np.sum(np.abs(x - y)) for x, y in zip(a_p, _b_p))
        for _b_p in permutations(b_p)
    )


def check_for_error_tolerance(
    results: list[ResultWithError[MixtureDistribution]],
    base_mixture: MixtureDistribution,
    expected_error: float,
) -> bool:
    """A function to check for compliance with an expected error"""

    for result in results:
        assert result.error is None
        actual_error = absolute_diff_params(result.content, base_mixture)
        if actual_error <= expected_error:
            return True
    return False
