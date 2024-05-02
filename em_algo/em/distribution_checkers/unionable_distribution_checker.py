"""TODO"""

from abc import ABC

from em_algo.distribution_mixture import DistributionInMixture
from em_algo.em import EM


class UnionDistributionChecker(EM.DistributionChecker):
    """TODO"""

    def __init__(self, breakpointers: list[EM.DistributionChecker] | None) -> None:
        super().__init__()
        self._distribution_checkers: list[EM.DistributionChecker] = (
            [] if breakpointers is None else breakpointers
        )

    @property
    def name(self):
        return " + ".join(
            distribution_checker.name
            for distribution_checker in self._distribution_checkers
        )

    def __add__(
        self,
        additional: "UnionDistributionChecker | EM.DistributionChecker",
    ):
        if isinstance(additional, UnionDistributionChecker):
            return UnionDistributionChecker(
                self._distribution_checkers + additional._distribution_checkers
            )
        return UnionDistributionChecker(self._distribution_checkers + [additional])

    def __radd__(
        self,
        additional: "UnionDistributionChecker | EM.DistributionChecker",
    ):
        return self + additional

    @staticmethod
    def union(
        first: "UnionDistributionChecker | EM.DistributionChecker",
        second: "UnionDistributionChecker | EM.DistributionChecker",
    ):
        """TODO"""
        if isinstance(first, UnionDistributionChecker):
            return first + second
        if isinstance(second, UnionDistributionChecker):
            return first + second
        return UnionDistributionChecker([first, second])

    def is_alive(
        self,
        step: int,
        distribution: DistributionInMixture,
    ) -> bool:
        for distribution_checker in self._distribution_checkers:
            if not distribution_checker.is_alive(step, distribution):
                return False
        return True


class UnionableDistributionChecker(EM.DistributionChecker, ABC):
    """TODO"""

    def union(self, additional: EM.DistributionChecker):
        """TODO"""
        return UnionDistributionChecker.union(self, additional)

    def __add__(self, additional: EM.DistributionChecker):
        """TODO"""
        return self.union(additional)

    def __radd__(self, additional: EM.DistributionChecker):
        """TODO"""
        return self.union(additional)
