"""TODO"""

from typing import NamedTuple, Callable
from functools import partial

import random
import time
import pickle
import numpy as np
from tqdm.contrib.concurrent import process_map

from em_algo.types import Samples
from em_algo.distribution_mixture import DistributionMixture
from em_algo.problem import Problem, Result
from em_algo.em import EM

from examples.config import RESULTS_FOLDER

np.seterr(all="ignore")


class Test(NamedTuple):
    """TODO"""

    index: int
    all_data: Samples
    true_mixture: DistributionMixture

    problem: Problem
    solvers: list[EM]

    runs: int


class SingleSolverResult(NamedTuple):
    """TODO"""

    test: Test

    solver: EM
    result: Result
    steps: int
    time: float

    log: list[EM.Log.Item] = []


class TestResult(NamedTuple):
    """TODO"""

    test: Test
    results: list[SingleSolverResult]


class Clicker:
    """TODO"""

    def __init__(self) -> None:
        self._counter = -1

    def click(self):
        """TODO"""
        self._counter += 1
        return self._counter


def run_test(
    test: Test,
    create_history=False,
    remember_time=False,
) -> TestResult:
    """TODO"""

    times = []
    results = []

    for solver in test.solvers:
        for _ in range(test.runs):
            start = time.perf_counter()
            result = solver.solve_logged(
                test.problem,
                create_history=create_history,
                remember_time=remember_time,
            )
            stop = time.perf_counter()
            times.append(stop - start)

        results.append(
            SingleSolverResult(
                test,
                solver,
                result.result,
                result.log.steps,
                float(np.mean(times)),
                result.log.log,
            )
        )

    return TestResult(test, results)


def run_tests(
    tests: list[Test],
    workers_count: int,
    shuffled: bool = True,
    chunksize: int = 1,
    create_history=False,
    remember_time=False,
) -> list[TestResult]:
    """TODO"""

    if not shuffled:
        _tests = tests
    else:
        _tests = list(tests)
        random.shuffle(_tests)

    results: list[TestResult] = process_map(
        partial(run_test, create_history=create_history, remember_time=remember_time),
        _tests,
        max_workers=workers_count,
        chunksize=chunksize,
    )

    if shuffled:
        key: Callable[[TestResult], int] = lambda t: t.test.index
        results.sort(key=key)

    return results


def save_results(results: list[TestResult], name: str) -> None:
    """TODO"""

    with open(RESULTS_FOLDER / f"{name}.pkl", "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


def open_results(name: str) -> list[TestResult]:
    """TODO"""

    with open(RESULTS_FOLDER / f"{name}.pkl", "rb") as f:
        return pickle.load(f)
