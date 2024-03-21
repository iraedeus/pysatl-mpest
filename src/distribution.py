from typing import NamedTuple
from models import Model
from utils import Params


# Describes all needed data about distribution
class Distribution(NamedTuple):
    model: type[Model]
    params: Params
    prior_probability: float | None = None
