"""Module which contains COBYLA optimizer"""

from typing import Callable
from scipy.optimize import minimize

from em_algo.types import Params
from em_algo.optimizers import AOptimizer


class ScipyCOBYLA(AOptimizer):
    """Class which represents SciPy COBYLA optimizer"""

    @property
    def name(self):
        return "ScipyCOBYLA"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        return minimize(func, params, method="COBYLA").x
