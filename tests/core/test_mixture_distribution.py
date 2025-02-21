import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad

from mpest.core.distribution import Distribution
from mpest.core.mixture_distribution import DistributionInMixture, MixtureDistribution
from mpest.models import AModel, ExponentialModel, WeibullModelExp
from mpest.models.gaussian import GaussianModel
from tests.core.utils import generate_random_distribution


@pytest.mark.parametrize(
    "model, params, prior",
    [
        (ExponentialModel(), np.array([1.0]), 1.0),
        (WeibullModelExp(), np.array([1.0, 2.0]), 0.5),
        (GaussianModel(), np.array([0.0, 2.0]), 0.7),
    ],
)
def test_distribution_in_mixture_init(model, params, prior):
    dist_in_mix = DistributionInMixture(model, params, prior)
    assert dist_in_mix.prior_probability == prior
    assert np.array_equal(dist_in_mix.params, params)
    assert dist_in_mix.model == model


@pytest.mark.parametrize(
    "model, params, prior",
    [
        (ExponentialModel(), np.array([1.0]), 1.0),
        (WeibullModelExp(), np.array([1.0, 2.0]), 0.5),
        (GaussianModel(), np.array([0.0, 2.0]), 0.7),
    ],
)
def test_distribution_in_mixture_pdf(model, params, prior):
    dist = DistributionInMixture(model, params, prior)
    assert isinstance(dist.pdf(1), float)
    assert all(dist.pdf(x) > 0 for x in np.linspace(0.01, 10, 100))


@pytest.mark.parametrize(
    "model, params, prior",
    [
        (ExponentialModel(), np.array([1.0]), None),
        (WeibullModelExp(), np.array([1.0, 2.0]), None),
        (GaussianModel(), np.array([0.0, 2.0]), None),
    ],
)
def test_distribution_in_mixture_pdf_none_prior(model, params, prior):
    dist = DistributionInMixture(model, params, prior)
    assert all(dist.pdf(x) == 0 for x in np.linspace(0.01, 10, 10))


@settings(max_examples=1000)
@given(st.integers(2, 10))
def test_mixture_distribution_init(size):
    priors = np.random.dirichlet(np.ones(size))
    dists = [
        generate_random_distribution(params_range=(0.0, 5.0), prior_probability_range=(prior, prior), in_mixture=True)
        for prior in priors
    ]
    mixture = MixtureDistribution(dists)

    assert isinstance(mixture, MixtureDistribution)
    assert np.isclose(sum(d.prior_probability for d in mixture.distributions if d.prior_probability is not None), 1.0)
    assert all(isinstance(d, DistributionInMixture) for d in mixture.distributions)
    assert all(np.array_equal(d.params, dmx.params) for d, dmx in zip(dists, mixture.distributions))
    assert all(np.isclose(d.prior_probability, dmx.prior_probability) for d, dmx in zip(dists, mixture.distributions))


@settings(max_examples=1000)
@given(st.integers(2, 10))
def test_mixture_distribution_from_distributions(size):
    priors = list(np.random.dirichlet(np.ones(size)))
    dists = [generate_random_distribution(params_range=(0.0, 5.0)) for _ in range(size)]
    mixture = MixtureDistribution.from_distributions(dists, priors)

    assert isinstance(mixture, MixtureDistribution)
    assert np.isclose(sum(d.prior_probability for d in mixture.distributions if d.prior_probability is not None), 1.0)
    assert all(isinstance(d, DistributionInMixture) for d in mixture.distributions)
    assert all(np.array_equal(d.params, dmx.params) for d, dmx in zip(dists, mixture.distributions))
    assert all(np.isclose(prior, dmx.prior_probability) for prior, dmx in zip(priors, mixture.distributions))


@settings(max_examples=1000)
@given(st.integers(2, 10))
def test_mixture_distribution_len(size):
    priors = list(np.random.dirichlet(np.ones(size)))
    dists = [generate_random_distribution(params_range=(0.0, 5.0)) for _ in range(size)]
    dists_in = [generate_random_distribution(params_range=(0.0, 5.0), in_mixture=True) for _ in range(size)]
    mixture_init = MixtureDistribution(dists_in)
    mixture_from = MixtureDistribution.from_distributions(dists, priors)

    assert len(mixture_init.distributions) == len(dists)
    assert len(mixture_from.distributions) == len(dists_in)
    assert len(mixture_from.distributions) == len(mixture_init.distributions)


@settings(max_examples=1000)
@given(st.integers(2, 10))
def test_mixture_distribution_from_distributions_none_priors(size):
    dists = [generate_random_distribution(params_range=(0.0, 5.0), prior_probability_range=None) for _ in range(size)]
    mixture = MixtureDistribution.from_distributions(dists)

    assert isinstance(mixture, MixtureDistribution)
    assert len(mixture.distributions) == size
    assert np.isclose(sum(d.prior_probability for d in mixture.distributions if d.prior_probability is not None), 1.0)
    assert all(isinstance(d, DistributionInMixture) for d in mixture.distributions)
    assert all(np.array_equal(d.params, dmx.params) for d, dmx in zip(dists, mixture.distributions))
    assert all(np.isclose(dmx.prior_probability, 1 / size) for dmx in mixture.distributions)


def test_mixture_distribution_from_distributions_different_sizes():
    with pytest.raises(ValueError):
        MixtureDistribution.from_distributions([Distribution.from_params(GaussianModel, [0.0, 1.0])], [0.2, 0.8])


@settings(max_examples=1000)
@given(st.integers(2, 10))
def test_mixture_distribution_getitem(size):
    priors = list(np.random.dirichlet(np.ones(size)))
    dists = [generate_random_distribution(params_range=(0.0, 5.0)) for _ in range(size)]
    mixture = MixtureDistribution.from_distributions(dists, priors)

    assert all(np.allclose(dists[i].params, mixture[i].params) for i in range(size))
    assert all(np.allclose(priors[i], mixture[i].prior_probability) for i in range(size))

    with pytest.raises(IndexError):
        mixture[size]


@settings(max_examples=10000, deadline=None)
@given(st.integers(2, 10))
def test_mixture_distribution_pdf(size):
    priors = list(np.random.dirichlet(np.ones(size)))
    dists = [generate_random_distribution(params_range=(0.0, 5.0)) for _ in range(size)]
    mixture = MixtureDistribution.from_distributions(dists, priors)

    X = np.linspace(0.1, 10, 100)
    expected_pdfs = [sum(priors[i] * d.pdf(x) for i, d in enumerate(dists)) for x in X]
    actual_pdfs = [mixture.pdf(x) for x in X]

    assert np.allclose(expected_pdfs, actual_pdfs)
    assert all(x > 0 for x in actual_pdfs)
    assert np.isclose(quad(mixture.pdf, -np.inf, np.inf)[0], 1)


def test_mixture_distribution_has_generator():
    priors = list(np.random.dirichlet(np.ones(4)))
    dists = [generate_random_distribution(params_range=(0.0, 5.0)) for _ in range(4)]
    mixture = MixtureDistribution.from_distributions(dists, priors)

    assert mixture.has_generator


@settings(max_examples=1000)
@given(st.integers(2, 10))
def test_mixture_distribution_normalize(size):
    priors = list(np.random.uniform(1, 100, size))
    priors_sum = sum(priors)
    dists = [generate_random_distribution(params_range=(0.0, 5.0)) for _ in range(size)]
    mixture = MixtureDistribution.from_distributions(dists, priors)

    assert all(np.isclose(priors[i] / priors_sum, dmx.prior_probability) for i, dmx in enumerate(mixture))


@settings(max_examples=1000)
@given(st.integers(2, 10), st.integers(0, 100))
def test_mixture_distribution_generate_valid(mixture_size, samples_size):
    priors = list(np.random.uniform(1, 100, mixture_size))
    dists = [generate_random_distribution(params_range=(0.0, 5.0)) for _ in range(mixture_size)]
    mixture = MixtureDistribution.from_distributions(dists, priors)

    generated_samples = mixture.generate(size=samples_size)
    assert len(generated_samples) == samples_size


def test_mixture_distribution_generate_invalid():
    class MockModel(AModel):
        @property
        def name(self) -> str:
            return "Mock"

        def params_convert_from_model(self, params):
            return params

        def params_convert_to_model(self, params):
            return params

        def pdf(self, x, params):
            return 0

        def lpdf(self, x, params):
            return 0

    mock_model = MockModel()
    distributions = [Distribution(mock_model, np.array([1.0])), Distribution(mock_model, np.array([2.0]))]
    mixture = MixtureDistribution.from_distributions(distributions)
    with pytest.raises(TypeError):
        mixture.generate()

    gaussian_model = GaussianModel()
    distributions = [Distribution(gaussian_model, np.array([0.0, 1.0])), Distribution(mock_model, np.array([2.0]))]
    mixture = MixtureDistribution.from_distributions(distributions)
    with pytest.raises(TypeError):
        mixture.generate()
