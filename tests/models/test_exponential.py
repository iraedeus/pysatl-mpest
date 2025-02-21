import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.stats import expon, kstest

from mpest.models.exponential import ExponentialModel, LMomentsParameterMixin


class TestLMomentsParameterMixin:
    """
    Test cases for LMomentsParameterMixin.
    """

    def test_calc_lambda(self):
        mixin = LMomentsParameterMixin()
        moments = [2.0]
        expected_lambda = 1 / moments[0]
        assert mixin.calc_lambda(moments) == pytest.approx(expected_lambda)


class TestExponentialModel:
    """
    Test cases for the ExponentialModel.
    """

    def test_name(self):
        model = ExponentialModel()
        assert model.name == "Exponential"

    @given(lmbda=st.floats(min_value=0.01, max_value=10))
    def test_params_convert_to_model(self, lmbda):
        model = ExponentialModel()
        params = np.array([lmbda])
        converted_params = model.params_convert_to_model(params)
        assert np.allclose(converted_params, np.log(params))

    @given(lmbda=st.floats(min_value=0.01, max_value=10))
    def test_params_convert_from_model(self, lmbda):
        model = ExponentialModel()
        params = np.array([lmbda])
        converted_params = model.params_convert_from_model(params)
        assert np.allclose(converted_params, np.exp(params))

    @settings(deadline=None)
    @given(lmbda=st.floats(min_value=0.01, max_value=10), size=st.integers(min_value=100, max_value=1000))
    def test_generate_normalized(self, lmbda, size):
        model = ExponentialModel()
        params = np.array(np.log([lmbda]))
        samples = model.generate(params, size=size, normalized=True)

        assert isinstance(samples, np.ndarray)
        assert samples.shape == (size,)
        assert np.all(samples >= 0)

        # Perform Kolmogorov-Smirnov test
        D, p_value = kstest(samples, lambda x: expon.cdf(x, scale=1 / lmbda))
        assert p_value > 0.05

    @settings(deadline=None)
    @given(
        lmbda=st.floats(min_value=0.01, max_value=10),
        size=st.integers(min_value=100, max_value=1000),
    )
    def test_generate_not_normalized(self, lmbda, size):
        model = ExponentialModel()
        params = np.array([lmbda])
        samples = model.generate(params, size=size, normalized=False)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (size,)
        assert np.all(samples >= 0)

        # Perform Kolmogorov-Smirnov test
        D, p_value = kstest(samples, lambda x: expon.cdf(x, scale=1 / lmbda))
        assert p_value > 0.05

    @settings(deadline=None)
    @given(x=st.floats(min_value=-10, max_value=10), lmbda=st.floats(min_value=0.01, max_value=10))
    def test_pdf(self, x, lmbda):
        model = ExponentialModel()
        params = np.array(np.log([lmbda]))

        pdf_actual = model.pdf(x, params)
        if x < 0:
            assert pdf_actual == 0
        else:
            pdf_expected = expon.pdf(x, scale=1 / lmbda)
            assert pdf_actual >= 0
            assert pdf_actual == pytest.approx(pdf_expected)

    @settings(deadline=None)
    @given(lmbda=st.floats(min_value=0.01, max_value=10))
    def test_pdf_integral(self, lmbda):
        model = ExponentialModel()
        params = np.array(np.log([lmbda]))

        def pdf(x):
            return model.pdf(x, params)

        integral, error = quad(pdf, 0, np.inf)
        assert integral == pytest.approx(1.0, rel=0.01)

    @settings(deadline=None)
    @given(x=st.floats(min_value=-10, max_value=10), lmbda=st.floats(min_value=0.01, max_value=10))
    def test_lpdf(self, x, lmbda):
        model = ExponentialModel()
        params = np.array(np.log([lmbda]))

        lpdf_actual = model.lpdf(x, params)
        if x < 0:
            assert lpdf_actual == -np.inf
        else:
            lpdf_expected = expon.logpdf(x, scale=1 / lmbda)
            assert np.isfinite(lpdf_actual)
            assert lpdf_expected == pytest.approx(lpdf_actual)

    @settings(deadline=None)
    @given(x=st.floats(min_value=-10, max_value=10), lmbda=st.floats(min_value=0.01, max_value=10))
    def test_ldl(self, x, lmbda):
        model = ExponentialModel()
        params = np.array(np.log([lmbda]))
        ldl_actual = model.ldl(x, params)

        if x < 0:
            assert ldl_actual == -np.inf
        else:
            ldl_expected = 1 - np.exp(np.log(lmbda)) * x
            assert np.isfinite(ldl_actual)
            assert ldl_actual == pytest.approx(ldl_expected)

    @settings(deadline=None)
    @given(x=st.floats(min_value=-10, max_value=10), lmbda=st.floats(min_value=0.01, max_value=10))
    def test_ld_params(self, x, lmbda):
        model = ExponentialModel()
        params = np.array([lmbda])
        ld_params = model.ld_params(x, params)
        assert isinstance(ld_params, np.ndarray)
        assert ld_params.shape == (1,)
        if x < 0:
            assert ld_params[0] == -np.inf
        else:
            assert np.isfinite(ld_params[0])

    def test_calc_params(self):
        model = ExponentialModel()
        moments = [2.0]
        calculated_params = model.calc_params(moments)
        expected_lambda = 1 / moments[0]
        assert np.allclose(calculated_params, np.array([expected_lambda]))
