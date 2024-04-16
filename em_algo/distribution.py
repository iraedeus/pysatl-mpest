"""TODO"""

from typing import NamedTuple
from em_algo.models import AModel
from em_algo.types import Params


class Distribution(NamedTuple):
    """Describes all needed data about distribution"""

    model: type[AModel]
    params: Params
    prior_probability: float | None = None
