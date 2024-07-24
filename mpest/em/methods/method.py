from mpest.em.methods.abstract_steps import AExpectation, AMaximization
from mpest.mixture_distribution import MixtureDistribution
from mpest.problem import Problem
from mpest.utils import ResultWithError


class Method:
    def __init__(self, e_step: AExpectation, m_step: AMaximization):
        self.e_step = e_step
        self.m_step = m_step

    def step(self, problem: Problem) -> ResultWithError[MixtureDistribution]:
        e_result = self.e_step.step(problem)
        return self.m_step.step(e_result)
