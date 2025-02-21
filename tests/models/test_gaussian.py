import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.stats import kstest, norm

from mpest.models.gaussian import GaussianModel, LMomentsParameterMixin


class TestLMomentsParameterMixin:
    """
    Test cases for LMomentsParameterMixin.
    """

    def test_calc_mean(self):
        mixin = LMomentsParameterMixin()
        moments = [2.0]  # Example L-moment
        expected_mean = moments[0]
        assert mixin.calc_mean(moments) == pytest.approx(expected_mean)

    def test_calc_variance(self):
        mixin = LMomentsParameterMixin()
        moments = [1.0, 2.0]  # Example L-moments
        expected_variance = moments[1] * np.sqrt(np.pi)
        assert mixin.calc_variance(moments) == pytest.approx(expected_variance)


class TestGaussianModel:
    """
    Test cases for the GaussianModel.
    """

    def test_name(self):
        model = GaussianModel()
        assert model.name == "Gaussian"

    @given(mean=st.floats(min_value=-10, max_value=10), sd=st.floats(min_value=0.01, max_value=10))
    def test_params_convert_to_model(self, mean, sd):
        model = GaussianModel()
        params = np.array([mean, sd])
        converted_params = model.params_convert_to_model(params)
        assert np.allclose(converted_params, np.array([mean, np.log(sd)]))

    @given(mean=st.floats(min_value=-10, max_value=10), log_sd=st.floats(min_value=-5, max_value=5))
    def test_params_convert_from_model(self, mean, log_sd):
        model = GaussianModel()
        params = np.array([mean, log_sd])
        converted_params = model.params_convert_from_model(params)
        assert np.allclose(converted_params, np.array([mean, np.exp(log_sd)]))

    @settings(deadline=None, max_examples=50)
    @given(
        mean=st.floats(min_value=-10, max_value=10),
        sd=st.floats(min_value=0.01, max_value=10),
        size=st.integers(min_value=100, max_value=1000),
    )
    def test_generate_normalized(self, mean, sd, size):
        model = GaussianModel()
        params = np.array([mean, np.log(sd)])
        samples = model.generate(params, size=size, normalized=True)

        assert isinstance(samples, np.ndarray)
        assert samples.shape == (size,)

        # Perform Kolmogorov-Smirnov test
        D, p_value = kstest(samples, lambda x: norm.cdf(x, loc=mean, scale=sd))
        assert p_value > 0.05

    @settings(deadline=None, max_examples=50)
    @given(
        mean=st.floats(min_value=-10, max_value=10),
        sd=st.floats(min_value=0.01, max_value=10),
        size=st.integers(min_value=100, max_value=1000),
    )
    def test_generate_not_normalized(self, mean, sd, size):
        model = GaussianModel()
        params = np.array([mean, sd])
        samples = model.generate(params, size=size, normalized=False)

        assert isinstance(samples, np.ndarray)
        assert samples.shape == (size,)

        # Perform Kolmogorov-Smirnov test
        D, p_value = kstest(samples, lambda x: norm.cdf(x, loc=mean, scale=sd))
        assert p_value > 0.05

    @settings(deadline=None)
    @given(
        x=st.floats(min_value=-10, max_value=10),
        mean=st.floats(min_value=-10, max_value=10),
        sd=st.floats(min_value=0.01, max_value=10),
    )
    def test_pdf(self, x, mean, sd):
        model = GaussianModel()
        params = np.array([mean, np.log(sd)])
        pdf_value = model.pdf(x, params)
        expected_pdf = norm.pdf(x, loc=mean, scale=sd)
        assert pdf_value >= 0
        assert pdf_value == pytest.approx(expected_pdf)

    @settings(deadline=None)
    @given(
        x=st.floats(min_value=-10, max_value=10),
        mean=st.floats(min_value=-10, max_value=10),
        sd=st.floats(min_value=1.0, max_value=10),
    )
    def test_lpdf(self, x, mean, sd):
        model = GaussianModel()
        params = np.array([mean, np.log(sd)])
        lpdf_value = model.lpdf(x, params)
        pdf_value = model.pdf(x, params)

        if pdf_value < 0:
            assert lpdf_value == -np.inf
        else:
            expected_lpdf = norm.logpdf(x, loc=mean, scale=sd)
            assert np.isfinite(lpdf_value)
            assert lpdf_value == pytest.approx(expected_lpdf)

    @settings(deadline=None)
    @given(
        mean=st.floats(min_value=-10, max_value=10),
        sd=st.floats(min_value=1.0, max_value=10),
    )
    def test_pdf_integral(self, mean, sd):
        model = GaussianModel()
        params = np.array([mean, np.log(sd)])  # Pass log(sd) as the model expects

        def pdf(x):
            return model.pdf(x, params)

        integral, error = quad(pdf, -np.inf, np.inf)
        assert integral == pytest.approx(1.0, rel=0.01)

    @settings(deadline=None)
    @given(
        x=st.floats(min_value=-10, max_value=10),
        mean=st.floats(min_value=-10, max_value=10),
        sd=st.floats(min_value=0.01, max_value=10),
    )
    def test_ldm(self, x, mean, sd):
        model = GaussianModel()
        params = np.array([mean, np.log(sd)])  # Pass log(sd) as the model expects
        ldm_value = model.ldm(x, params)

        # Analytical derivative of log likelihood with respect to mean
        expected_ldm = (x - mean) / (sd**2)
        assert ldm_value == pytest.approx(expected_ldm)

    @settings(deadline=None)
    @given(
        x=st.floats(min_value=-10, max_value=10),
        mean=st.floats(min_value=-10, max_value=10),
        sd=st.floats(min_value=0.01, max_value=10),
    )
    def test_ldsd(self, x, mean, sd):
        model = GaussianModel()
        params = np.array([mean, np.log(sd)])  # Pass log(sd) as the model expects
        ldsd_value = model.ldsd(x, params)

        # Analytical derivative of log likelihood with respect to log(sd)
        expected_ldsd = ((x - mean) ** 2) / (sd**2) - 1
        assert ldsd_value == pytest.approx(expected_ldsd)

    @settings(deadline=None)
    @given(
        x=st.floats(min_value=-10, max_value=10),
        mean=st.floats(min_value=-10, max_value=10),
        sd=st.floats(min_value=0.01, max_value=10),
    )
    def test_ld_params(self, x, mean, sd):
        model = GaussianModel()
        params = np.array([mean, np.log(sd)])  # Pass log(sd) as the model expects
        ld_params = model.ld_params(x, params)

        # Analytical derivatives of log likelihood with respect to mean and log(sd)
        expected_ldm = (x - mean) / (sd**2)
        expected_ldsd = ((x - mean) ** 2) / (sd**2) - 1
        assert np.allclose(ld_params, np.array([expected_ldm, expected_ldsd]))

    def test_calc_params(self):
        model = GaussianModel()
        moments = [2.0, 1.0]  # Example L-moments
        calculated_params = model.calc_params(moments)
        expected_mean = moments[0]
        expected_variance = moments[1] * np.sqrt(np.pi)
        assert np.allclose(calculated_params, np.array([expected_mean, expected_variance]))
