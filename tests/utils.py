"""TODO"""

from mpest.em import EM
from mpest.em.breakpointers import ParamDifferBreakpointer, StepCountBreakpointer
from mpest.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from mpest.optimizers import ALL_OPTIMIZERS
from mpest.problem import Problem, Result


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
