"""TODO"""

from em_algo.problem import Problem, Result
from em_algo.em.breakpointers import StepCountBreakpointer, ParamDifferBreakpointer
from em_algo.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from em_algo.optimizers import ALL_OPTIMIZERS
from em_algo.em import EM


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
