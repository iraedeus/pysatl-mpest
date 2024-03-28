"""TODO"""

import random
import numpy as np

from test_utils import Test, run_tests, save_results

from em_algo.models import ExponentialModel, GaussianModel
from em_algo.distribution import Distribution
from em_algo.optimizer import ScipyNewtonCG

if __name__ == "__main__":
    models = [
        (ExponentialModel, ((0.25, 15),)),
        (GaussianModel, ((-15, 15), (0.25, 15))),
    ]
    k_list = [(1, 1), (1, 2), (2, 2), (2, 3)]
    sizes: list[int] = [50, 100, 200]
    MAX_STEP = 32
    DISTRIBUTION_COUNT = 32
    BASE_SIZE = 1024
    TESTS_PER_COND = 32
    RUNS_PER_TEST = 1

    tests = []

    for xk, yk in k_list:
        for _ in range(DISTRIBUTION_COUNT):
            PER_MODEL = BASE_SIZE // (xk + yk)
            base_distributions = []
            x = []
            for k, model in zip((xk, yk), models):
                for _ in range(k):
                    o = np.array(
                        [random.uniform(border[0], border[1]) for border in model[1]]
                    )
                    x += list(model[0].generate(o, PER_MODEL))
                    base_distributions.append(
                        Distribution(model[0], model[0].params_convert_to_model(o))
                    )
            base = np.array(x)
            for size in sizes:
                for _ in range(TESTS_PER_COND):
                    start_distributions = []
                    for k, model in zip((xk, yk), models):
                        for _ in range(k):
                            o = np.array(
                                [
                                    random.uniform(border[0], border[1])
                                    for border in model[1]
                                ]
                            )
                            start_distributions.append(
                                Distribution(
                                    model[0], model[0].params_convert_to_model(o)
                                )
                            )
                    tests.append(
                        Test(
                            len(tests),
                            start_distributions,
                            base_distributions,
                            base,
                            np.array(random.sample(x, size)),
                            xk + yk,
                            RUNS_PER_TEST,
                            0.01,
                            MAX_STEP,
                            0.001,
                            3,
                            ScipyNewtonCG,
                        )
                    )

    result = run_tests(tests, 14)

    save_results(result, "exponential_gauss")
