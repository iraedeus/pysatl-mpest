"""Module which represents problem, which can be solved by using this lib."""

from abc import ABC, abstractmethod

from em_algo.types import Samples
from em_algo.distribution_mixture import DistributionMixture
from em_algo.utils import ResultWithError


class Problem:
    """
    Class which represents the parameter estimation of distribution mixture problem.

    Described by samples and the initial approximation.
    Initial approximation is an distribution mixture.
    """

    def __init__(
        self,
        samples: Samples,
        distributions: DistributionMixture,
    ) -> None:
        self._samples = samples
        self._distributions = distributions

    @property
    def samples(self):
        """Samples getter"""
        return self._samples

    @property
    def distributions(self):
        """Distributions getter"""
        return self._distributions


Result = ResultWithError[DistributionMixture]


class ASolver(ABC):
    """
    Abstract class which represents solver for
    the parameter estimation of distributions mixture problem.
    """

    @abstractmethod
    def solve(self, problem: Problem) -> Result:
        """
        Method which solve the parameter estimation
        of distributions mixture problem.
        """
