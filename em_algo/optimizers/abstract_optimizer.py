"""TODO"""

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from em_algo.types import Params
from em_algo.utils import Named


class AOptimizer(Named, ABC):
    """TODO"""

    @abstractmethod
    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        """TODO"""


class AOptimizerJacobian(Named, ABC):
    """TODO"""

    @abstractmethod
    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
        jacobian: Callable[[Params], np.ndarray],
    ) -> Params:
        """TODO"""


TOptimizer = AOptimizer | AOptimizerJacobian
