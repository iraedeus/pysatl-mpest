import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.stats import kstest, weibull_min

from mpest.models.weibull import LMomentsParameterMixin, WeibullModelExp


class TestLMomentsParameterMixin:
    """
    Test cases for LMomentsParameterMixin.
    """

    def test_calc_k(self):
        mixin = LMomentsParameterMixin()
        moments = [2.0, 1.0]
        m1, m2 = moments[0], moments[1]
        expected_k = -np.log(2) / np.log(1 - (m2 / m1))
        assert mixin.calc_k(moments) == pytest.approx(expected_k)

    def test_calc_lambda(self):
        mixin = LMomentsParameterMixin()
        moments = [2.0, 1.0]
        m1 = moments[0]
        k = mixin.calc_k(moments)
        expected_lambda = m1 / math.gamma(1 + 1 / k)
        assert mixin.calc_lambda(moments) == pytest.approx(expected_lambda)


class TestWeibullModelExp:
    """
    Test cases for the WeibullModelExp.
    """

    def test_name(self):
        model = WeibullModelExp()
        assert model.name == "WeibullExp"

    @given(k=st.floats(min_value=0.01, max_value=10), lmbda=st.floats(min_value=0.01, max_value=10))
    def test_params_convert_to_model(self, k, lmbda):
        model = WeibullModelExp()
        params = np.array([k, lmbda])
        converted_params = model.params_convert_to_model(params)
        assert np.allclose(converted_params, np.log(params))

    @given(log_k=st.floats(min_value=-5, max_value=5), log_lmbda=st.floats(min_value=-5, max_value=5))
    def test_params_convert_from_model(self, log_k, log_lmbda):
        model = WeibullModelExp()
        params = np.array([log_k, log_lmbda])
        converted_params = model.params_convert_from_model(params)
        assert np.allclose(converted_params, np.exp(params))

    @settings(deadline=None, max_examples=50)
    @given(
        k=st.floats(min_value=0.01, max_value=10),
        lmbda=st.floats(min_value=0.01, max_value=10),
        size=st.integers(min_value=100, max_value=1000),
    )
    def test_generate_normalized(self, k, lmbda, size):
        model = WeibullModelExp()
        params = np.array([np.log(k), np.log(lmbda)])
        samples = model.generate(params, size=size, normalized=True)

        assert isinstance(samples, np.ndarray)
        assert samples.shape == (size,)
        assert np.all(samples >= 0)

        # Perform Kolmogorov-Smirnov test
        D, p_value = kstest(samples, lambda x: weibull_min.cdf(x, k, loc=0, scale=lmbda))
        assert p_value > 0.1

    @settings(deadline=None, max_examples=50)
    @given(
        k=st.floats(min_value=0.01, max_value=10),
        lmbda=st.floats(min_value=0.01, max_value=10),
        size=st.integers(min_value=100, max_value=1000),
    )
    def test_generate_not_normalized(self, k, lmbda, size):
        model = WeibullModelExp()
        params = np.array([k, lmbda])
        samples = model.generate(params, size=size, normalized=False)

        assert isinstance(samples, np.ndarray)
        assert samples.shape == (size,)
        assert np.all(samples >= 0)

        # Perform Kolmogorov-Smirnov test
        D, p_value = kstest(samples, lambda x: weibull_min.cdf(x, k, loc=0, scale=lmbda))
        assert p_value > 0.1

    @settings(deadline=None)
    @given(
        x=st.floats(min_value=-10, max_value=10),
        k=st.floats(min_value=0.01, max_value=10),
        lmbda=st.floats(min_value=0.01, max_value=10),
    )
    def test_pdf(self, x, k, lmbda):
        model = WeibullModelExp()
        params = np.array([np.log(k), np.log(lmbda)])
        pdf_value = model.pdf(x, params)
        expected_pdf = weibull_min.pdf(x, k, loc=0, scale=lmbda)

        if x < 0:
            assert pdf_value == 0
        else:
            assert pdf_value >= 0
            assert pdf_value == pytest.approx(expected_pdf)

    @settings(deadline=None)
    @given(
        x=st.floats(min_value=-10, max_value=10).filter(lambda x: x > 1 or x < -1),
        k=st.floats(min_value=0.01, max_value=10),
        lmbda=st.floats(min_value=0.01, max_value=10),
    )
    def test_lpdf(self, x, k, lmbda):
        model = WeibullModelExp()
        params = np.array([np.log(k), np.log(lmbda)])
        lpdf_value = model.lpdf(x, params)
        if x < 0:
            assert lpdf_value == -np.inf
        else:
            expected_lpdf = weibull_min.logpdf(x, k, loc=0, scale=lmbda)
            assert np.isfinite(lpdf_value)
            assert lpdf_value == pytest.approx(expected_lpdf)

    @settings(deadline=None)
    @given(
        k=st.floats(min_value=0.5, max_value=10),
        lmbda=st.floats(min_value=0.5, max_value=10),
    )
    def test_pdf_integral(self, k, lmbda):
        model = WeibullModelExp()
        params = np.array([np.log(k), np.log(lmbda)])

        def pdf(x):
            return model.pdf(x, params)

        integral, error = quad(pdf, 0, np.inf)
        assert integral == pytest.approx(1.0, rel=0.01)

    @settings(deadline=None)
    @given(
        x=st.floats(min_value=-10, max_value=10).filter(lambda x: x > 1 or x < -1),
        k=st.floats(min_value=0.5, max_value=10),
        lmbda=st.floats(min_value=0.5, max_value=10),
    )
    def test_ldk(self, x, k, lmbda):
        model = WeibullModelExp()
        params = np.array([np.log(k), np.log(lmbda)])
        ldk_value = model.ldk(x, params)

        ek, elm = np.exp(params)
        xlm = x / elm
        ldk_expected = 1.0 - ek * ((xlm**ek) - 1.0) * np.log(x / elm)

        if x < 0:
            assert ldk_value == -np.inf
        else:
            assert ldk_value == pytest.approx(ldk_expected, rel=0.01)

    @settings(deadline=None)
    @given(
        x=st.floats(min_value=-10, max_value=10).filter(lambda x: x > 1 or x < -1),
        k=st.floats(min_value=0.01, max_value=10),
        lmbda=st.floats(min_value=0.01, max_value=10),
    )
    def test_ldl(self, x, k, lmbda):
        model = WeibullModelExp()
        params = np.array([np.log(k), np.log(lmbda)])
        ldl_value = model.ldl(x, params)

        expected_ldl = k * ((x / lmbda) ** k - 1)
        if x < 0:
            assert ldl_value == -np.inf
        else:
            assert ldl_value == pytest.approx(expected_ldl, rel=0.01)

    @settings(deadline=None)
    @given(
        x=st.floats(min_value=0.01, max_value=10),
        k=st.floats(min_value=0.01, max_value=10),
        lmbda=st.floats(min_value=0.01, max_value=10),
    )
    def test_ld_params(self, x, k, lmbda):
        model = WeibullModelExp()
        params = np.array([np.log(k), np.log(lmbda)])
        ld_params = model.ld_params(x, params)

        ek, elm = np.exp(params)
        xlm = x / elm
        expected_ldk = 1.0 - ek * ((xlm**ek) - 1.0) * np.log(x / elm)
        expected_ldl = k * ((x / lmbda) ** k - 1)
        assert np.allclose(ld_params, np.array([expected_ldk, expected_ldl]), rtol=0.01)

    def test_calc_params(self):
        model = WeibullModelExp()
        moments = [2.0, 1.0]  # Example L-moments
        calculated_params = model.calc_params(moments)
        m1, m2 = moments[0], moments[1]
        expected_k = -np.log(2) / np.log(1 - (m2 / m1))
        expected_lambda = m1 / math.gamma(1 + 1 / expected_k)

        assert np.allclose(calculated_params, np.array([expected_k, expected_lambda]))
