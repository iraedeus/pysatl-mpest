"""TODO"""

from em_algo.distribution_mixture import DistributionMixture
from em_algo.em.breakpointers import UnionableBreakpointer


class StepCountBreakpointer(UnionableBreakpointer):
    """TODO"""

    def __init__(self, max_step: int | None = 16) -> None:
        self._max_step = max_step

    @property
    def max_step(self):
        """TODO"""
        return self._max_step

    @property
    def name(self):
        return f"StepCountBreakpointer(max_step={self.max_step})"

    def is_over(
        self,
        step: int,
        previous_step: DistributionMixture | None,
        current_step: DistributionMixture,
    ) -> bool:
        if (self.max_step is not None) and (step >= self.max_step):
            return True
        return False
