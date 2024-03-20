from models import Model
from utils import *
from typing import NamedTuple


# Describes all needed data about distribution
class Distribution(NamedTuple):
    model: Model
    params: params
    prior_probability: float | None = None
