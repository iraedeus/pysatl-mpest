"""TODO"""

import random
import numpy as np

from examples.utils import Test, Clicker, run_tests, save_results
from examples.mono_test_generator import generate_mono_test
from examples.config import MAX_WORKERS

from em_algo.models import WeibullModelExp, GaussianModel, ExponentialModel, AModel

from em_algo.em import EM
from em_algo.em.breakpointers import StepCountBreakpointer, ParamDifferBreakpointer
from em_algo.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from em_algo.optimizers import ScipyNewtonCG

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    tests: list[Test] = []

    counter = Clicker()

    def _generate_test(
        model: type[AModel], o_borders: list[tuple[float, float]]
    ) -> list[Test]:
        return generate_mono_test(
            model_t=model,
            params_borders=o_borders,
            clicker=counter,
            ks=list(range(1, 6)),
            sizes=[50, 100, 200, 500, 1000],
            distributions_count=64,
            base_size=2048,
            tests_per_size=16,
            tests_per_cond=4,
            runs_per_test=1,
            solver=EM(
                StepCountBreakpointer(32) + ParamDifferBreakpointer(0.01),
                FiniteChecker() + PriorProbabilityThresholdChecker(0.001, 3),
                ScipyNewtonCG(),
            ),
        )

    tests += _generate_test(WeibullModelExp, [(0.25, 25), (0.25, 25)])
    tests += _generate_test(GaussianModel, [(-15, 15), (0.25, 25)])
    tests += _generate_test(ExponentialModel, [(0.25, 25)])

    results = run_tests(tests, workers_count=MAX_WORKERS, shuffled=True)

    save_results(results, "big_mono_test")
