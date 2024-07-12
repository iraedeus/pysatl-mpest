"""Unit test module which tests mixture of two distributions parameter estimation"""

from itertools import permutations

import numpy as np
import pytest

from mpest.distribution import Distribution
from mpest.mixture_distribution import MixtureDistribution
from mpest.models import (
    AModelWithGenerator,
    ExponentialModel,
    GaussianModel,
    WeibullModelExp,
)
from mpest.problem import Problem
from mpest.utils import Factory
from tests.utils import run_test


@pytest.mark.parametrize(
    "model_factory, params, start_params, size, deviation, expected_error",
    [
        (
            Factory(WeibullModelExp),
            [(0.5, 1.0), (1.0, 0.5)],
            [(1.0, 1.0), (0.5, 0.5)],
            500,
            0.01,
            0.2,
        ),
        (
            Factory(WeibullModelExp),
            [(0.5, 0.5), (1.0, 1.0)],
            [(1.0, 0.5), (0.5, 1.0)],
            500,
            0.01,
            0.2,
        ),
        (
            Factory(GaussianModel),
            [(0.0, 5.0), (1.0, 1.0)],
            [(1.0, 5.0), (-1.0, 5.0)],
            500,
            0.01,
            0.2,
        ),
        (
            Factory(GaussianModel),
            [(0.0, 5.0), (5.0, 2.0)],
            [(1.0, 5.0), (4.0, 5.0)],
            500,
            0.01,
            0.2,
        ),
        (
            Factory(ExponentialModel),
            [(1.0,), (2.0,)],
            [(0.5,), (1.5,)],
            500,
            0.01,
            0.2,
        ),
        (
            Factory(ExponentialModel),
            [(2.0,), (5.0,)],
            [(1.0,), (7.0,)],
            500,
            0.01,
            0.2,
        ),
    ],
)
def test_two_same_distributions_simple(
    model_factory: Factory[AModelWithGenerator],
    params,
    start_params,
    size: int,
    deviation: float,
    expected_error: float,
):
    """Runs mixture of two distributions parameter estimation unit test"""

    # pylint: disable=too-many-arguments

    np.random.seed(42)

    models = [model_factory.construct() for _ in range(len(params))]

    params = [np.array(param) for param in params]
    start_params = [np.array(param) for param in start_params]

    x = []
    for model, param in zip(models, params):
        x += list(model.generate(param, size))

    np.random.shuffle(x)
    x = np.array(x)

    c_params = [
        model.params_convert_to_model(param) for model, param in zip(models, params)
    ]
    c_start_params = [
        model.params_convert_to_model(param) for model, param in zip(models, params)
    ]

    problem = Problem(
        samples=x,
        distributions=MixtureDistribution.from_distributions(
            [Distribution(model, param) for model, param in zip(models, c_start_params)]
        ),
    )

    for result in run_test(problem=problem, deviation=deviation):
        assert result.error is None

        def absolute_diff_params(
            a: MixtureDistribution,
            b: MixtureDistribution,
        ):
            """Metric which checks absolute differ of gotten distribution mixtures"""

            a_p, b_p = ([d.params for d in ld] for ld in (a, b))

            return min(
                sum(np.sum(np.abs(x - y)) for x, y in zip(a_p, _b_p))
                for _b_p in permutations(b_p)
            )

        assert (
            absolute_diff_params(
                result.content,
                MixtureDistribution.from_distributions(
                    [
                        Distribution(model, param)
                        for model, param in zip(models, c_params)
                    ]
                ),
            )
            <= expected_error
        )
