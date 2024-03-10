from models import Model
from utils import *
from typing import NamedTuple


class Distribution(NamedTuple):
    model: Model
    params: params
    prior_probability: float | None = None
