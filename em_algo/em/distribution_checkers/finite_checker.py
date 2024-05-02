"""TODO"""

import numpy as np

from em_algo.em.distribution_checkers import UnionableDistributionChecker
from em_algo.distribution_mixture import DistributionInMixture


class FiniteChecker(UnionableDistributionChecker):
    """TODO"""

    @property
    def name(self):
        return "FiniteChecker"

    def is_alive(
        self,
        step: int,
        distribution: DistributionInMixture,
    ) -> bool:
        if (distribution.prior_probability is not None) and not np.isfinite(
            distribution.prior_probability
        ):
            return False
        return bool(np.all(np.isfinite(distribution.params)))
