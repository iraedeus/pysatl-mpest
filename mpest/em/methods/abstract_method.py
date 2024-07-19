"""Module which contains abstract model classes, which describe methods and E, M steps"""

from abc import ABC, abstractmethod
from typing import Any

from mpest.mixture_distribution import MixtureDistribution
from mpest.problem import Problem
from mpest.utils import ANamed, ResultWithError


class AExpectation(ABC):
    """
    Abstract class which represents E step for EM
    """

    @abstractmethod
    def step(self, problem: Problem) -> Any:
        """
        Function which performs E step

        :param problem: Object of class Problem, which contains samples and mixture.
        :return: Args, which used in m step. Depends on method
        """


class AMaximization(ABC):
    """
    Abstract class which represents E step for EM
    """

    @abstractmethod
    def step(self, e_result: Any) -> ResultWithError[MixtureDistribution]:
        """
        Function which performs M step

        :param e_result: Args, which got from e step. Depends on method
        :return: Object of class MixtureDistribution with new params of distributions
        """


class AMethod(ANamed, ABC):
    """
    Abstract class which represents method for EM

    :param e_step: The object of the internal subclass in which the E step is performed
    :param m_step: The object of the internal subclass in which the M step is performed
    """

    @abstractmethod
    def __init__(self, e_step: AExpectation, m_step: AMaximization) -> None:
        """
        Method object constructor

        :param e_step: The object of the internal subclass in which the E step is performed
        :param m_step: The object of the internal subclass in which the M step is performed
        """

        self.e_step = e_step
        self.m_step = m_step

    @abstractmethod
    def step(self, problem: Problem) -> ResultWithError[MixtureDistribution]:
        """
        A function that performs the E and M steps

        :param problem: Object of class Problem, which contains samples and mixture.
        :return: Object of class MixtureDistribution with new params of distributions
        """
