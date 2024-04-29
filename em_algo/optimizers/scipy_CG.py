"""TODO"""

from typing import Callable
from scipy.optimize import minimize

from em_algo.types import Params
from em_algo.optimizers import AOptimizer


class ScipyCG(AOptimizer):
    """TODO"""

    @property
    def name(self):
        return "ScipyCG"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        return minimize(func, params, method="CG").x
