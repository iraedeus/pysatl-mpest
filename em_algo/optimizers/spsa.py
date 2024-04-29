"""TODO"""

from typing import Callable

import spsa

from em_algo.types import Params
from em_algo.optimizers import AOptimizer


class SPSA(AOptimizer):
    """TODO"""

    @property
    def name(self):
        return "SPSA"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        return spsa.minimize(func, params)
