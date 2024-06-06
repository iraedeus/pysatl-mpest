"""Module which represents optimizers and abstract classes for extending"""

from mpest.optimizers.abstract_optimizer import (
    AOptimizer,
    AOptimizerJacobian,
    TOptimizer,
)
from mpest.optimizers.scipy_newton_cg import ScipyNewtonCG
from mpest.optimizers.scipy_cg import ScipyCG
from mpest.optimizers.scipy_cobyla import ScipyCOBYLA
from mpest.optimizers.scipy_nelder_mead import ScipyNelderMead
from mpest.optimizers.scipy_slsqp import ScipySLSQP
from mpest.optimizers.scipy_tnc import ScipyTNC

ALL_OPTIMIZERS = [
    ScipyCG(),
    ScipyNewtonCG(),
    ScipyNelderMead(),
    ScipySLSQP(),
    ScipyTNC(),
    ScipyCOBYLA(),
]
