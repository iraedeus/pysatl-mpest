"""Module which contains truncated Newton (TNC) optimizer"""

from typing import Callable
from scipy.optimize import minimize

from em_algo.types import Params
from em_algo.optimizers import AOptimizer


class ScipyTNC(AOptimizer):
    """Class which represents SciPy truncated Newton (TNC) optimizer"""

    @property
    def name(self):
        return "ScipyTNC"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        return minimize(func, params, method="TNC").x
