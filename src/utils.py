import numpy as np
from collections import namedtuple


sample = np.ndarray
params = np.ndarray

distribution_data = namedtuple(
    "distribution_data",
    [
        "model",
        "params",
        "prior_probability"
    ]
)
