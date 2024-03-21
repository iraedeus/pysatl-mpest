import random
import time
from typing import NamedTuple
import sys
import pickle
import numpy as np
from tqdm.contrib.concurrent import process_map

# fmt: off

sys.path.insert(1, "../src")

from em import EM
from distribution import Distribution
import utils
from models import WeibullModelExp, GaussianModel, Model
from optimizer import Optimizer, ScipyNewtonCG

# fmt: on

MAX_WORKERS = 12


class Test(NamedTuple):
    number: int
    model: type[Model]

    base_data: utils.Samples
    data: utils.Samples
    k: int
    params: list[utils.Params]
    start_params: list[utils.Params]
    runs_per_test: int

    deviation: float
    max_step: int
    prior_probability_threshold: float
    prior_probability_threshold_step: int
    optimizer: type[Optimizer]


class TestResult(NamedTuple):
    test: Test
    result: EM.Result
    time: float


class Clicker():
    def __init__(self):
        self._counter = -1

    def get(self):
        self._counter += 1
        return self._counter


def run_test(test: Test) -> TestResult:
    times = []

    for _ in range(test.runs_per_test):
        start = time.perf_counter()
        result = EM.em_algo(
            test.data,
            [
                Distribution(
                    test.model,
                    test.model.params_convert_to_model(params)
                )
                for params in test.start_params
            ],
            test.k,
            deviation=test.deviation,
            max_step=test.max_step,
            prior_probability_threshold=test.prior_probability_threshold,
            prior_probability_threshold_step=test.prior_probability_threshold_step,
            optimizer=test.optimizer
        )

        stop = time.perf_counter()
        times.append(stop - start)

    return TestResult(test, result, float(np.mean(times)))


def generate_test(
    model: type[Model],
    o_borders_for_data: list[tuple[float, float]],
    clicker: Clicker,

    o_borders_for_start_params: list[tuple[float, float]] | None = None,

    k_list: list[int] = [1, 2, 3, 4, 5],
    sizes: list[int] = [50, 100, 200, 500],
    distribution_count: int = 1,
    base_size: int = 1024,
    tests_per_cond: int = 1,
    runs_per_test: int = 3,

    deviation: float = 0.01,
    max_step: int = 16,
    prior_probability_threshold: float = 0.001,
    prior_probability_threshold_step: int = 3,
    optimizer: type[Optimizer] = ScipyNewtonCG
) -> list[Test]:
    new_tests: list[Test] = []

    for k in k_list:
        for _ in range(distribution_count):
            per_model = base_size // k
            params = []
            x = []
            for _ in range(k):
                o = np.array([
                    random.uniform(border[0], border[1])
                    for border in o_borders_for_data
                ])
                x += list(model.generate(o, per_model))
                params.append(o)

            random.shuffle(x)
            base = np.array(x)

            if o_borders_for_start_params is not None:
                params_borders = o_borders_for_start_params
            else:
                params_borders = o_borders_for_data

            for size in sizes:
                for _ in range(tests_per_cond):
                    start_params = [
                        np.array([
                            random.uniform(border[0], border[1])
                            for border in params_borders
                        ])
                        for _ in range(k)
                    ]
                    new_tests.append(
                        Test(
                            clicker.get(),
                            model,

                            base,
                            np.array(random.sample(x, size)),
                            k,
                            params,
                            start_params,
                            runs_per_test,

                            deviation,
                            max_step,
                            prior_probability_threshold,
                            prior_probability_threshold_step,
                            optimizer
                        ))
    return new_tests


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    tests: list[Test] = []

    counter = Clicker()

    def _generate_test(model: type[Model], o_borders: list[tuple[float, float]]) -> list[Test]:
        return generate_test(
            model=model,
            o_borders_for_data=o_borders,
            clicker=counter,
            max_step=32
        )

    tests += _generate_test(WeibullModelExp, [(0.25, 25), (0.25, 25)])
    tests += _generate_test(GaussianModel, [(-15, 15), (0.25, 25)])

    random.shuffle(tests)

    results = process_map(run_test, tests, max_workers=MAX_WORKERS)
    results.sort(key=lambda t: t.test.number)

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
