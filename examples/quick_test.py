"""TODO"""

import random
import numpy as np

from examples.utils import Test, run_tests, save_results, Clicker
from examples.mono_test_generator import generate_mono_test
from examples.config import MAX_WORKERS

from em_algo.models import WeibullModelExp, GaussianModel, ExponentialModel, AModel
from em_algo.em import EM
from em_algo.em.breakpointers import StepCountBreakpointer, ParamDifferBreakpointer
from em_algo.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from em_algo.optimizers import (
    ScipyCG,
    ScipyNewtonCG,
    ScipyNelderMead,
    ScipySLSQP,
    ScipyTNC,
    ScipyCOBYLA,
)


def run_test():
    """TODO"""

    random.seed(42)
    np.random.seed(42)

    tests: list[Test] = []

    counter = Clicker()

    def _generate_test(
        model: type[AModel], params_borders: list[tuple[float, float]]
    ) -> list[Test]:
        test = generate_mono_test(
            model_t=model,
            solvers=[
                EM(
                    StepCountBreakpointer(16) + ParamDifferBreakpointer(0.01),
                    FiniteChecker() + PriorProbabilityThresholdChecker(0.001, 3),
                    optimizer,
                )
                for optimizer in [
                    ScipyCG(),
                    ScipyNewtonCG(),
                    ScipyNelderMead(),
                    ScipySLSQP(),
                    ScipyTNC(),
                    ScipyCOBYLA(),
                ]
            ],
            clicker=counter,
            params_borders=params_borders,
            ks=[1, 2, 3, 4, 5],
            sizes=[50, 100, 200, 500],
            distributions_count=1,
            base_size=1024,
            tests_per_size=1,
            tests_per_cond=1,
            runs_per_test=1,
        )
        return test

    tests += _generate_test(WeibullModelExp, [(0.25, 25), (0.25, 25)])
    tests += _generate_test(GaussianModel, [(-15, 15), (0.25, 25)])
    tests += _generate_test(ExponentialModel, [(0.25, 25)])

    results = run_tests(tests, MAX_WORKERS, True)

    save_results(results, "quick_test")


if __name__ == "__main__":
    run_test()
