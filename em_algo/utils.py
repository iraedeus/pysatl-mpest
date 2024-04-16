"""TODO"""

from itertools import permutations
import numpy as np

from em_algo.distribution import Distribution


def absolute_diff_params(
    a: list[Distribution],
    b: list[Distribution],
):
    """TODO"""

    a_p, b_p = ([d.params for d in ld] for ld in (a, b))

    return min(
        sum(np.sum(np.abs(x - y)) for x, y in zip(a_p, _b_p))
        for _b_p in permutations(b_p)
    )
