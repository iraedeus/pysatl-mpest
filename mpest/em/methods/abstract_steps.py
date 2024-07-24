"""Module which contains abstract model classes, which describe methods and E, M steps"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from mpest.mixture_distribution import MixtureDistribution
from mpest.problem import Problem
from mpest.utils import ResultWithError

X = TypeVar("X")
Y = TypeVar("Y")


class AExpectation(Generic[X], ABC):
    """
    Abstract class which represents E step for EM
    """

    @abstractmethod
    def step(self, problem: Problem) -> X:
        """
        Function which performs E step

        :param problem: Object of class Problem, which contains samples and mixture.
        :return: Args, which used in m step. Depends on method
        """


class AMaximization(Generic[Y], ABC):
    """
    Abstract class which represents M step for EM
    """

    @abstractmethod
    def step(self, e_result: Y) -> ResultWithError[MixtureDistribution]:
        """
        Function which performs M step

        :param e_result: Args, which got from e step. Depends on method
        :return: Object of class MixtureDistribution with new params of distributions
        """
