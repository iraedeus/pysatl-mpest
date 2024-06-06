"""Module which represents distribution."""

from em_algo.models import AModel
from em_algo.types import Params


class Distribution:
    """Class which represents all needed data about distribution."""

    def __init__(
        self,
        model: AModel,
        params: Params,
    ) -> None:
        self._model = model
        self._params = params

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
