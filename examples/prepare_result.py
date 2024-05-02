"""TODO"""

import itertools

import pandas as pd
import numpy as np


from tqdm.contrib.concurrent import process_map

from em_algo.types import Samples
from em_algo.distribution_mixture import DistributionMixture, DistributionInMixture
from examples.utils import SingleSolverResult, TestResult
from examples.config import MAX_WORKERS


def nll(samples: Samples, mixture: DistributionMixture) -> float:
    """TODO"""
    occur = sum(np.log(mixture.pdf(x)) for x in samples) / len(samples)
    if occur == -0.0:
        occur = 0.0
    return occur


def result_to_df(result: SingleSolverResult):
    """TODO"""

    distribution_mixture = DistributionMixture(
        [
            (
                d
                if (d.prior_probability is not None) and (d.prior_probability > 0.001)
                else DistributionInMixture(d.model, d.params, None)
            )
            for d in result.result.result
        ]
    )
    failed = all(d.prior_probability is None for d in result.result.result)

    return {
        "test_index": result.test.index,
        "optimizer": result.solver.optimizer.name,
        "k": len(result.test.true_mixture),
        "sample": result.test.problem.samples,
        "true_mixture": result.test.true_mixture,
        "result_mixture": distribution_mixture,
        "error": result.result.error,
        "log": result.log,
        "steps": result.steps,
        "time": result.time,
        "model": result.test.true_mixture[0].model.name,
        "size": len(result.test.problem.samples),
        "success": (result.steps < 16) and failed,
        "failed": failed,
        "occur": nll(result.test.all_data, distribution_mixture),
    }


def prepare(results: list[TestResult]):
    """TODO"""

    return pd.DataFrame(
        process_map(
            result_to_df,
            list(itertools.chain.from_iterable(result.results for result in results)),
            max_workers=MAX_WORKERS,
            chunksize=256,
        )
    )
