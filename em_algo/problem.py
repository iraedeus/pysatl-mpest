"""TODO"""

from abc import ABC, abstractmethod

from em_algo.types import Samples
from em_algo.distribution_mixture import DistributionMixture
from em_algo.utils import ResultWithError


class Problem:
    """TODO"""

    def __init__(
        self,
        samples: Samples,
        distributions: DistributionMixture,
    ) -> None:
        self._samples = samples
        self._distributions = distributions

    @property
    def samples(self):
        """TODO"""
        return self._samples

    @property
    def distributions(self):
        """TODO"""
        return self._distributions


Result = ResultWithError[DistributionMixture]


class Solver(ABC):
    """TODO"""

    @abstractmethod
    def solve(self, problem: Problem) -> Result:
        """TODO"""
