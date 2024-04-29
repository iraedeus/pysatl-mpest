"""TODO"""

from typing import Iterable
import random
import numpy as np

from examples.utils import Clicker, Test

from em_algo.problem import Problem
from em_algo.models import AModel
from em_algo.em import EM
from em_algo.distribution import Distribution
from em_algo.distribution_mixture import DistributionMixture


def generate_mono_test(
    model_t: type[AModel],
    solver: EM,
    clicker: Clicker,
    params_borders: list[tuple[float, float]],
    start_params_borders: list[tuple[float, float]] | None = None,
    ks: Iterable[int] = (1, 2, 3, 4, 5),
    sizes: Iterable[int] = (50, 100, 200, 500),
    distributions_count: int = 1,
    base_size: int = 1024,
    tests_per_size: int = 8,
    tests_per_cond: int = 8,
    runs_per_test: int = 3,
):
    """TODO"""
    tests: list[Test] = []

    for k in ks:
        for _ in range(distributions_count):
            per_model = base_size // k
            true_params = []
            x = []
            models: list[AModel] = []

            for _ in range(k):
                params = np.array(
                    [random.uniform(border[0], border[1]) for border in params_borders]
                )
                model = model_t()
                x += list(model.generate(params, per_model))

                true_params.append(params)
                models.append(model)

            random.shuffle(x)
            all_samples = np.array(x)

            if start_params_borders is None:
                start_params_borders = params_borders

            for size in sizes:
                for _ in range(tests_per_cond):
                    samples = random.sample(x, size)
                    for _ in range(tests_per_size):
                        start_params = [
                            np.array(
                                [
                                    random.uniform(border[0], border[1])
                                    for border in start_params_borders
                                ]
                            )
                            for _ in range(k)
                        ]
                        tests.append(
                            Test(
                                clicker.click(),
                                all_samples,
                                DistributionMixture.from_distributions(
                                    [
                                        Distribution(
                                            model,
                                            model.params_convert_to_model(params),
                                        )
                                        for model, params in zip(models, true_params)
                                    ]
                                ),
                                Problem(
                                    np.array(samples),
                                    DistributionMixture.from_distributions(
                                        [
                                            Distribution(
                                                model,
                                                model.params_convert_to_model(params),
                                            )
                                            for model, params in zip(
                                                models, start_params
                                            )
                                        ]
                                    ),
                                ),
                                solver,
                                runs_per_test,
                            )
                        )
    return tests
