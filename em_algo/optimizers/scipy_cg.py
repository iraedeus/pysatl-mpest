"""Module which contains CG (Conjugate Gradient iteration) optimizer"""

from typing import Callable
from scipy.optimize import minimize

from em_algo.types import Params
from em_algo.optimizers import AOptimizer


class ScipyCG(AOptimizer):
    """Class which represents SciPy CG (Conjugate Gradient iteration) optimizer"""

    @property
    def name(self):
        return "ScipyCG"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        return minimize(func, params, method="CG").x
