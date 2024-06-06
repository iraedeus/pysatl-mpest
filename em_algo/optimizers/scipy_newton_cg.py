"""Module which contains Newton-CG optimizer"""

from typing import Callable
import numpy as np
from scipy.optimize import minimize

from em_algo.types import Params
from em_algo.optimizers.abstract_optimizer import AOptimizerJacobian


class ScipyNewtonCG(AOptimizerJacobian):
    """Class which represents SciPy Newton-CG optimizer"""

    @property
    def name(self):
        return "ScipyNewtonCG"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
        jacobian: Callable[[Params], np.ndarray],
    ) -> Params:
        return minimize(func, params, jac=jacobian, method="Newton-CG").x
