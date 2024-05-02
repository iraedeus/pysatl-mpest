"""TODO"""

from abc import ABC

from em_algo.distribution_mixture import DistributionMixture
from em_algo.em import EM


class UnionBreakpointer(EM.Breakpointer):
    """TODO"""

    def __init__(self, breakpointers: list[EM.Breakpointer] | None) -> None:
        super().__init__()
        self._breakpointers: list[EM.Breakpointer] = (
            [] if breakpointers is None else breakpointers
        )

    @property
    def name(self):
        return " + ".join(breakpointer.name for breakpointer in self._breakpointers)

    def __add__(self, additional: "UnionBreakpointer | EM.Breakpointer"):
        if isinstance(additional, UnionBreakpointer):
            return UnionBreakpointer(self._breakpointers + additional._breakpointers)
        return UnionBreakpointer(self._breakpointers + [additional])

    def __radd__(self, additional: "UnionBreakpointer | EM.Breakpointer"):
        return self + additional

    @staticmethod
    def union(
        first: "UnionBreakpointer | EM.Breakpointer",
        second: "UnionBreakpointer | EM.Breakpointer",
    ):
        """TODO"""
        if isinstance(first, UnionBreakpointer):
            return first + second
        if isinstance(second, UnionBreakpointer):
            return first + second
        return UnionBreakpointer([first, second])

    def is_over(
        self,
        step: int,
        previous_step: DistributionMixture | None,
        current_step: DistributionMixture,
    ) -> bool:
        for breakpointer in self._breakpointers:
            if breakpointer.is_over(step, previous_step, current_step):
                return True
        return False


class UnionableBreakpointer(EM.Breakpointer, ABC):
    """TODO"""

    def union(self, another: EM.Breakpointer):
        """TODO"""
        return UnionBreakpointer.union(self, another)

    def __add__(self, another: EM.Breakpointer):
        return self.union(another)

    def __radd__(self, another: EM.Breakpointer):
        return self.union(another)
