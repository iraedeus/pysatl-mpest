import random
import time
import numpy as np
from typing import NamedTuple
import seaborn as sns
from scipy.stats import weibull_min, norm
import sys
import pickle
from tqdm.contrib.concurrent import process_map

# fmt: off

sys.path.insert(1, "../src")

from distribution import Distribution
from utils import *
import utils
from em import EM
from models import WeibullModelExp, GaussianModel, Model

# fmt: on

"""
k_list = list(range(1, 6))
sizes = [50, 100, 200, 500]
distribution_count = 32
base_size = 1024
tests_per_cond = 20
"""

k_list = list(range(1, 6))
sizes = [50, 100, 200, 500]
distribution_count = 1
base_size = 1024
tests_per_cond = 1

max_workers = 4


class Test(NamedTuple):
    number: int
    model: Model
    base_data: sample
    data: sample
    k: int
    params: list[utils.params]
    params_modified: list[utils.params]
    start_params: list[utils.params]


class TestResult(NamedTuple):
    test: Test
    result: EM.Result
    time: float


def run_test(test: Test) -> TestResult:
    times = []

    for _ in range(3):
        start = time.perf_counter()
        result = EM.em_algo(
            test.data,
            [Distribution(test.model, params) for params in test.start_params],
            test.k,
            max_step=16,
            prior_probability_threshold=0.001
        )
        stop = time.perf_counter()
        times.append(stop - start)

    return TestResult(test, result, float(np.mean(times)))


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    tests: list[Test] = []

    for k in k_list:
        for _ in range(distribution_count):
            per_model = base_size // k
            params = []
            params_modified = []
            x = []
            for _ in range(k):
                o = np.array([random.uniform(0.25, 25) for _ in range(2)])
                x += list(weibull_min.rvs(
                    o[0],
                    loc=0,
                    scale=o[1],
                    size=per_model
                ))
                params.append(o)
                params_modified.append(np.log(o))
            random.shuffle(x)
            base = np.array(x)
            for size in sizes:
                for _ in range(tests_per_cond):
                    tests.append(
                        Test(
                            len(tests),
                            WeibullModelExp,
                            base,
                            np.array(random.sample(x, size)),
                            k,
                            params,
                            params_modified,
                            [
                                np.array([
                                    random.uniform(np.log(0.25), np.log(25))
                                    for _ in range(2)
                                ])
                                for _ in range(k)
                            ]
                        ))

    for k in k_list:
        for _ in range(distribution_count):
            per_model = base_size // k
            params = []
            params_modified = []
            x = []
            for _ in range(k):
                o = np.array([random.uniform(0.25, 25) for _ in range(2)])
                x += list(norm.rvs(loc=o[0], scale=o[1], size=per_model))
                params.append(o)
                params_modified.append(np.array([o[0], np.log(o[1])]))
            random.shuffle(x)
            base = np.array(x)
            for size in sizes:
                for _ in range(tests_per_cond):
                    tests.append(
                        Test(
                            len(tests),
                            GaussianModel,
                            base,
                            np.array(random.sample(x, size)),
                            k,
                            params,
                            params_modified,
                            [
                                np.array([
                                    random.uniform(0.25, 25),
                                    random.uniform(np.log(0.25), np.log(25))
                                ])
                                for _ in range(k)
                            ]
                        ))

    random.shuffle(tests)

    results = process_map(run_test, tests, max_workers=max_workers)

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
