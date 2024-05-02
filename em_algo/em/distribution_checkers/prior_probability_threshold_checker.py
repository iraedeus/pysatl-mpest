"""TODO"""

from em_algo.em.distribution_checkers import UnionableDistributionChecker
from em_algo.distribution_mixture import DistributionInMixture


class PriorProbabilityThresholdChecker(UnionableDistributionChecker):
    """TODO"""

    def __init__(
        self,
        prior_probability_threshold: float | None = 0.0001,
        prior_probability_threshold_step: int | None = 3,
    ) -> None:
        self._prior_probability_threshold = prior_probability_threshold
        self._prior_probability_threshold_step = prior_probability_threshold_step

    @property
    def prior_probability_threshold(self):
        """TODO"""
        return self._prior_probability_threshold

    @property
    def prior_probability_threshold_step(self):
        """TODO"""
        return self._prior_probability_threshold_step

    @property
    def name(self):
        return (
            "PriorProbabilityThresholdChecker("
            + f"threshold={self.prior_probability_threshold}, "
            + f"start_step={self.prior_probability_threshold_step}"
            + ")"
        )

    def is_alive(
        self,
        step: int,
        distribution: DistributionInMixture,
    ) -> bool:
        if distribution.prior_probability is None:
            return False
        if self.prior_probability_threshold is None:
            return True
        if self.prior_probability_threshold_step is not None:
            if step < self.prior_probability_threshold_step:
                return True
        if distribution.prior_probability < self.prior_probability_threshold:
            return False
        return True
