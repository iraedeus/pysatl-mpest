"""Module which contains experiment of moving one distribution against other"""

import random

import numpy as np

from examples.mono_test_generator import Clicker
from examples.utils import Test, run_tests, save_results
from examples.config import MAX_WORKERS
from em_algo.models import GaussianModel, WeibullModelExp
from em_algo import DistributionMixture, Distribution, Problem
from em_algo.em import EM
from em_algo.em.breakpointers import StepCountBreakpointer, ParamDifferBreakpointer
from em_algo.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from em_algo.optimizers import ScipyCG, ScipySLSQP, ScipyTNC, ScipyNewtonCG

# Gaussian

gaussian_start_params = [(0.0, 3.0), (-10.0, 3.0), (10.0, 3.0)]
weibull_start_params = [(0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (5.0, 1.0)]
sizes = [50, 100, 200, 500, 1000]
BASE_SIZE = 2048
TESTS_PER_COND = 8
TESTS_PER_SIZE = 8

clicker = Clicker()
gaussian = GaussianModel()
weibull = WeibullModelExp()

tests = []

for sp in gaussian_start_params:
    main_distr = list(gaussian.generate(np.array(sp), BASE_SIZE // 2))
    for second_sp in np.linspace(sp[0] - 5, sp[0] + 5, num=8, endpoint=True):
        x = main_distr + list(
            gaussian.generate(np.array((second_sp, 3.0)), BASE_SIZE // 2)
        )
        random.shuffle(x)

        start_params_borders = [
            (min(sp[0], second_sp) - 1.0, max(sp[0], second_sp) + 1.0),
            (1.0, 5.0),
        ]

        for size in sizes:
            for _ in range(TESTS_PER_COND):
                samples = random.sample(x, size)
                for _ in range(TESTS_PER_SIZE):
                    start_params = [
                        np.array(
                            [
                                random.uniform(border[0], border[1])
                                for border in start_params_borders
                            ]
                        )
                        for _ in range(2)
                    ]
                    tests.append(
                        Test(
                            clicker.click(),
                            np.array(x),
                            DistributionMixture.from_distributions(
                                [
                                    Distribution(
                                        gaussian,
                                        gaussian.params_convert_to_model(np.array(sp)),
                                    ),
                                    Distribution(
                                        gaussian,
                                        gaussian.params_convert_to_model(
                                            np.array((second_sp, 3.0))
                                        ),
                                    ),
                                ]
                            ),
                            Problem(
                                np.array(samples),
                                DistributionMixture.from_distributions(
                                    [
                                        Distribution(
                                            gaussian,
                                            gaussian.params_convert_to_model(params),
                                        )
                                        for params in start_params
                                    ]
                                ),
                            ),
                            [
                                EM(
                                    StepCountBreakpointer(16)
                                    + ParamDifferBreakpointer(0.01),
                                    FiniteChecker()
                                    + PriorProbabilityThresholdChecker(0.001, 3),
                                    optimizer,
                                )
                                for optimizer in [
                                    ScipyCG(),
                                    ScipyNewtonCG(),
                                    ScipySLSQP(),
                                    ScipyTNC(),
                                ]
                            ],
                            1,
                        )
                    )

for sp in weibull_start_params:
    main_distr = list(weibull.generate(np.array(sp), BASE_SIZE // 2))
    for second_sp in np.linspace(max(sp[0] - 5, 0.1), sp[0] + 5, num=8, endpoint=True):
        x = main_distr + list(
            weibull.generate(np.array((second_sp, 1.0)), BASE_SIZE // 2)
        )
        random.shuffle(x)

        start_params_borders = [
            (max(min(sp[0], second_sp) - 1.0, 0.1), max(sp[0], second_sp) + 1.0),
            (0.5, 2.0),
        ]

        for size in sizes:
            for _ in range(TESTS_PER_COND):
                samples = random.sample(x, size)
                for _ in range(TESTS_PER_SIZE):
                    start_params = [
                        np.array(
                            [
                                random.uniform(border[0], border[1])
                                for border in start_params_borders
                            ]
                        )
                        for _ in range(2)
                    ]
                    tests.append(
                        Test(
                            clicker.click(),
                            np.array(x),
                            DistributionMixture.from_distributions(
                                [
                                    Distribution(
                                        weibull,
                                        weibull.params_convert_to_model(np.array(sp)),
                                    ),
                                    Distribution(
                                        weibull,
                                        weibull.params_convert_to_model(
                                            np.array((second_sp, 3.0))
                                        ),
                                    ),
                                ]
                            ),
                            Problem(
                                np.array(samples),
                                DistributionMixture.from_distributions(
                                    [
                                        Distribution(
                                            weibull,
                                            weibull.params_convert_to_model(params),
                                        )
                                        for params in start_params
                                    ]
                                ),
                            ),
                            [
                                EM(
                                    StepCountBreakpointer(16)
                                    + ParamDifferBreakpointer(0.01),
                                    FiniteChecker()
                                    + PriorProbabilityThresholdChecker(0.001, 3),
                                    optimizer,
                                )
                                for optimizer in [
                                    ScipyCG(),
                                    ScipyNewtonCG(),
                                    ScipySLSQP(),
                                    ScipyTNC(),
                                ]
                            ],
                            1,
                        )
                    )

results = run_tests(
    tests,
    workers_count=MAX_WORKERS,
    shuffled=True,
    chunksize=64,
    # create_history=True,
    # remember_time=True,
)

save_results(results, "diff_test")
