"""Module which represents distribution."""

import numpy as np

from mpest.models import AModel
from mpest.types import Params


class Distribution:
    """Class which represents all needed data about distribution."""

    def __init__(
        self,
        model: AModel,
        params: Params,
    ) -> None:
        self._model = model
        self._params = params

    @classmethod
    def from_params(
        cls,
        model: type[AModel],
        params: list[float],
    ) -> "Distribution":
        """User friendly Distribution initializer"""

        model_obj = model()
        return cls(model_obj, model_obj.params_convert_to_model(np.array(params)))

    @property
    def model(self):
        """Model of distribution getter."""
        return self._model

    @property
    def params(self):
        """Params getter."""
        return self._params

    def pdf(self, x: float):
        """Probability density function for distribution."""
        return self.model.pdf(x, self.params)
