""" The module in which the L moments method is presented """
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

        def step(self, problem: Problem) -> tuple:
            """
            A function that performs E step

            :param problem: Object of class Problem, which contains samples and mixture.
            :return: ???
            """

    class MStep(AMaximization):
        """
        Class which calculate new params using matrix with indicator from E step.
        """

        def step(self, e_result: tuple) -> ResultWithError[MixtureDistribution]:
            """
            A function that performs E step

            :param e_result: ???
            """

    def __init__(self, e_step: AExpectation, m_step: AMaximization) -> None:
        """
        Object constructor

        :param e_step: The object of the internal subclass in which the E step is performed
        :param m_step: The object of the internal subclass in which the M step is performed
        """

        self.e_step = e_step
        self.m_step = m_step

    def step(self, problem: Problem) -> ResultWithError[MixtureDistribution]:
        """
        Function which performs E and M steps

        :param problem: Object of class Problem, which contains samples and mixture.
        """
