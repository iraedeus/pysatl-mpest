"""TODO"""

import pytest
import numpy as np

from em_algo.models import WeibullModelExp, GaussianModel, ExponentialModel, AModel
from em_algo.em import EM
from em_algo.distribution import Distribution
from em_algo.utils import absolute_diff_params


@pytest.mark.parametrize(
    "model, params, start_params, size, deviation, expected_error",
    [
        (
            WeibullModelExp,
            [(0.5, 1.0), (1.0, 0.5)],
            [(1.0, 1.0), (0.5, 0.5)],
            500,
            0.01,
            0.2,
        ),
        (
            WeibullModelExp,
            [(0.5, 0.5), (1.0, 1.0)],
            [(1.0, 0.5), (0.5, 1.0)],
            500,
            0.01,
            0.2,
        ),
        (
            GaussianModel,
            [(0.0, 5.0), (1.0, 1.0)],
            [(1.0, 5.0), (-1.0, 5.0)],
            500,
            0.01,
            0.2,
        ),
        (
            GaussianModel,
            [(0.0, 5.0), (5.0, 2.0)],
            [(1.0, 5.0), (4.0, 5.0)],
            500,
            0.01,
            0.2,
        ),
        (
            ExponentialModel,
            [(1.0,), (2.0,)],
            [(0.5,), (1.5,)],
            500,
            0.01,
            0.2,
        ),
        (
            ExponentialModel,
            [(2.0,), (5.0,)],
            [(1.0,), (7.0,)],
            500,
            0.01,
            0.2,
        ),
    ],
)
def test_one_distribution(
    model: type[AModel],
    params,
    start_params,
    size: int,
    deviation: float,
    expected_error: float,
):
    """TODO"""

    np.random.seed(42)

    params = [np.array(param) for param in params]
    start_params = [np.array(param) for param in start_params]

    x = []
    for param in params:
        x += list(model.generate(param, size))

    np.random.shuffle(x)
    x = np.array(x)

    c_params = [model.params_convert_to_model(param) for param in params]
    c_start_params = [model.params_convert_to_model(param) for param in start_params]

    result = EM.em_algo(
        samples=x,
        distributions=[Distribution(model, param) for param in c_start_params],
        k=1,
        deviation=deviation,
    )

    assert result.error is None

    assert (
        absolute_diff_params(
            result.distributions,
            [Distribution(model, param) for param in c_params],
        )
        <= expected_error
    )
