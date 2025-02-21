import numpy as np
import pytest

from mpest import Distribution, Params, Samples
from mpest.models import AModel, AModelWithGenerator, ExponentialModel, GaussianModel, WeibullModelExp


class DummyModelWithGenerator(AModelWithGenerator):
    @property
    def name(self):
        return "Dummy"

    def params_convert_to_model(self, params: Params) -> Params:
        return np.array([1.0])

    def params_convert_from_model(self, params: Params) -> Params:
        return np.array([1.0])

    def pdf(self, x: float, params: Params) -> float:
        return 1.0

    def lpdf(self, x: float, params: Params) -> float:
        return 1.0

    def generate(self, params: Params, size: int = 1, normalized: bool = False) -> Samples:
        return np.array([1.0, 2.0, 3.0])


class DummyModelWithoutGenerator(AModel):
    @property
    def name(self):
        return "Dummy"

    def params_convert_to_model(self, params: Params) -> Params:
        return np.array([1.0])

    def params_convert_from_model(self, params: Params) -> Params:
        return np.array([1.0])

    def pdf(self, x: float, params: Params) -> float:
        return 1.0

    def lpdf(self, x: float, params: Params) -> float:
        return 1.0


@pytest.mark.parametrize(
    "model, params",
    [
        (ExponentialModel(), np.array([1.0])),
        (WeibullModelExp(), np.array([1.0, 2.0])),
        (GaussianModel(), np.array([0.0, 2.0])),
    ],
)
def test_init(model, params):
    dist = Distribution(model, params)

    assert isinstance(dist.model, type(model))
    assert dist.model == model
    assert (dist.params == params).all()


@pytest.mark.parametrize(
    "model, params", [(ExponentialModel, [1.0]), (WeibullModelExp, [1.0, 2.0]), (GaussianModel, [0.0, 2.0])]
)
def test_from_params(model, params):
    dist = Distribution.from_params(model, params)

    assert isinstance(dist, Distribution)
    assert isinstance(dist.model, model)
    assert isinstance(dist.params, np.ndarray)
    assert (dist.params == params).all()


@pytest.mark.parametrize(
    "model, params",
    [
        (DummyModelWithGenerator, [1.0]),
    ],
)
def test_has_generator_true(model, params):
    dist = Distribution.from_params(model, params)
    assert dist.has_generator


@pytest.mark.parametrize(
    "model, params",
    [
        (DummyModelWithoutGenerator, [1.0]),
    ],
)
def test_has_generator_false(model, params):
    dist = Distribution.from_params(model, params)
    assert not (dist.has_generator)


@pytest.mark.parametrize(
    "model, params",
    [
        (DummyModelWithoutGenerator, [1.0]),
    ],
)
def test_generate_without_generator(model, params):
    dist = Distribution.from_params(model, params)

    with pytest.raises(TypeError):
        dist.generate(1)


@pytest.mark.parametrize(
    "model, params",
    [
        (ExponentialModel(), np.array([1.0])),
        (WeibullModelExp(), np.array([1.0, 2.0])),
        (GaussianModel(), np.array([0.0, 2.0])),
    ],
)
def test_pdf(model, params):
    dist = Distribution(model, params)
    assert isinstance(dist.pdf(1), float)
    assert all(dist.pdf(x) > 0 for x in np.linspace(0.01, 10, 100))


@pytest.mark.parametrize(
    "model, params, mean",
    [
        (ExponentialModel(), np.array([1.0]), 1.0),
        (ExponentialModel(), np.array([2.0]), 0.5),
        (ExponentialModel(), np.array([10.0]), 0.1),
    ],
)
def test_exp_generate_with_generator(model, params, mean):
    np.random.seed(42)

    dist = Distribution(model, params)
    samples = dist.generate(1000)

    assert isinstance(samples, np.ndarray)
    assert all(isinstance(x, float) for x in samples)
    assert mean == pytest.approx(samples.mean(), 0.05)


@pytest.mark.parametrize(
    "model, params, mean",
    [
        (GaussianModel(), np.array([1.0, 1.0]), 1.0),
        (GaussianModel(), np.array([1.0, 2.5]), 1.0),
        (GaussianModel(), np.array([1.0, 0.1]), 1.0),
        (GaussianModel(), np.array([3.0, 1.0]), 3.0),
        (GaussianModel(), np.array([-3.0, 1.0]), -3.0),
    ],
)
def test_norm_generate_with_generator(model, params, mean):
    np.random.seed(42)

    dist = Distribution(model, params)
    samples = dist.generate(1000)

    assert isinstance(samples, np.ndarray)
    assert all(isinstance(x, float) for x in samples)
    assert mean == pytest.approx(samples.mean(), 0.05)


@pytest.mark.parametrize(
    "model, params, mean",
    [
        (WeibullModelExp(), np.array([0.5, 1.0]), 2.0),
        (WeibullModelExp(), np.array([2.5, 1.0]), 0.887),
        (WeibullModelExp(), np.array([1.0, 3.0]), 3.0),
        (WeibullModelExp(), np.array([0.5, 3.0]), 6.0),
    ],
)
def test_weib_generate_with_generator(model, params, mean):
    np.random.seed(42)

    dist = Distribution(model, params)
    samples = dist.generate(1000)

    assert isinstance(samples, np.ndarray)
    assert all(isinstance(x, float) for x in samples)
    assert mean == pytest.approx(samples.mean(), 0.2)
