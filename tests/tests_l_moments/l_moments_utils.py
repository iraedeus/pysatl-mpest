"""TODO"""

from mpest.em import EM
from mpest.em.breakpointers import ParamDifferBreakpointer, StepCountBreakpointer
from mpest.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from mpest.em.methods.l_moments_method import IndicatorEStep, MStep
from mpest.em.methods.method import Method
from mpest.problem import Problem, Result


def run_test(problem: Problem, deviation: float) -> Result:
    """TODO"""
    method = Method(IndicatorEStep(), MStep())
    em_algo = EM(
        StepCountBreakpointer() + ParamDifferBreakpointer(deviation=deviation),
        FiniteChecker() + PriorProbabilityThresholdChecker(),
        method,
    )

    return em_algo.solve(problem=problem)
