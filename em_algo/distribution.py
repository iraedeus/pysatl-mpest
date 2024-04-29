"""TODO"""

from em_algo.models import AModel
from em_algo.types import Params


class Distribution:
    """Describes all needed data about distribution"""

    def __init__(
        self,
        model: AModel,
        params: Params,
    ) -> None:
        self._model = model
        self._params = params

    @property
    def model(self):
        """TODO"""
        return self._model

    @property
    def params(self):
        """TODO"""
        return self._params
