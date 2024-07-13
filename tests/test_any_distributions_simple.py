"""Unit test module which tests mixture of several different distribution parameter estimation"""

# pylint: disable=duplicate-code
# pylint: disable=too-many-arguments

import numpy as np
import pytest

from mpest import Distribution, MixtureDistribution, Problem
from mpest.models import (
    AModelWithGenerator,
    ExponentialModel,
    GaussianModel,
    WeibullModelExp,
)
from tests.utils import check_for_params_error_tolerance, run_test


def idfunc(vals):
    """Function for customizing pytest ids"""

    if isinstance(vals, list):
        if issubclass(type(vals[0]), AModelWithGenerator):
            return str([d.name for d in vals])
        return vals
    return f"{vals}"


@pytest.mark.parametrize(
    "models, params, start_params, size, deviation, expected_error",
    [
        (
            [WeibullModelExp(), GaussianModel()],
            [[0.5, 1.0], [5.0, 1.0]],
            [[1.5, 1.0], [1.0, 2.0]],
            1500,
            0.01,
            0.11,
        ),
        (
            [ExponentialModel(), GaussianModel()],
            [[0.5], [5.0, 1.0]],
            [[1.0], [0.0, 5.0]],
            1500,
            0.01,
            0.25,
        ),
        (
            [ExponentialModel(), WeibullModelExp()],
            [[0.5], [5.0, 1.0]],
            [[1.0], [3.0, 3.0]],
            1500,
            0.01,
            0.1,
        ),
        (
            [ExponentialModel(), GaussianModel(), WeibullModelExp()],
            [[1.0], [5.0, 1.0], [4.0, 1.0]],
            [[1.0], [1.0, 5.0], [1.0, 2.0]],
            1500,
            0.01,
            0.35,
        ),
    ],
    ids=idfunc,
)
def test(
    models,
    params,
    start_params,
    size,
    deviation,
    expected_error,
):
    """Runs mixture of several different distributions parameter estimation unit test"""

    np.random.seed(42)

    params = [np.array(param) for param in params]
    start_params = [np.array(param) for param in start_params]

    c_params = [
        model.params_convert_to_model(param) for model, param in zip(models, params)
    ]
    c_start_params = [
        model.params_convert_to_model(param)
        for model, param in zip(models, start_params)
    ]

    base_mixture = MixtureDistribution.from_distributions(
        [Distribution(model, param) for model, param in zip(models, c_params)],
    )

    x = base_mixture.generate(size)

    problem = Problem(
        samples=x,
        distributions=MixtureDistribution.from_distributions(
            [Distribution(model, param) for model, param in zip(models, c_start_params)]
        ),
    )

    results = run_test(problem=problem, deviation=deviation)
    assert check_for_params_error_tolerance(results, base_mixture, expected_error)
