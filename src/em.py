import numpy as np
import warnings
from optimizer import Optimizer, ScipyNewtonCG
from utils import *
from distribution import Distribution
from typing import NamedTuple

warnings.filterwarnings("ignore")


class EM:
    def __init__(
        self,
        deviation: float = 0.01,
        max_step: int | None = None,
        optimizer: Optimizer = ScipyNewtonCG    # type: ignore
    ):
        self.deviation = deviation
        self.max_step = max_step
        self.optimizer = optimizer
        self.result = None

    class Result(NamedTuple):
        distributions: list[Distribution]
        steps: int
        error: Exception | None = None

    @staticmethod
    def em_algo(
        X: sample,
        distributions: list[Distribution],
        k: int,
        deviation: float = 0.01,
        max_step: int | None = 50,
        optimizer: Optimizer = ScipyNewtonCG    # type: ignore
    ) -> "EM.Result":
        def end_cond(
            prev: list[Distribution] | None,
            curr: list[Distribution],
            step: int
        ) -> bool:
            if prev is None:
                return True
            if (max_step is not None) and (step > max_step):
                return False

            prev_o = np.array([d.params for d in prev])
            prev_w = np.array([d.prior_probability for d in prev])
            curr_o = np.array([d.params for d in curr])
            curr_w = np.array([d.prior_probability for d in curr])

            return not (
                np.all(np.abs(prev_o - curr_o) < deviation)
                and np.all(np.abs(prev_w - curr_w) < deviation)
            )

        step = 0
        curr = [Distribution(model, o, 1 / k) for model, o, _ in distributions]
        prev = None

        while end_cond(prev, curr, step):
            prev = list(curr)

            # E part
            p_xij = []
            cX = []
            for x in X:
                p = np.array([model.p(x, o) for model, o, _ in curr])
                if np.any(p):
                    p_xij.append(p)
                    cX.append(x)

            if not cX:
                return EM.Result(curr, step, Exception("All models can't match"))

            m = len(p_xij)
            h = np.zeros([k, m])
            curr_w = np.array([w for _, _, w in curr])
            for i, p in enumerate(p_xij):
                wp = curr_w * p
                swp = np.sum(wp)
                if not swp:
                    return EM.Result(curr, step, Exception("Error in E step"))
                h[:, i] = wp / swp

            # M part
            new_w = np.sum(h, axis=1) / m

            for j, ch in enumerate(h[:]):
                model, o, _ = curr[j]

                if np.isnan(new_w[j]):
                    curr[j] = Distribution(model, o, new_w[j])
                    continue

                new_o = optimizer.minimize(
                    lambda o: -np.sum(ch * [model.lp(x, o) for x in cX]),
                    o,
                    jacobian=lambda o: -np.sum(
                        ch * np.swapaxes([model.ldO(x, o) for x in cX], 0, 1),
                        axis=1
                    )
                )
                curr[j] = Distribution(model, new_o, new_w[j])

            step += 1

        return EM.Result(curr, step)

    def fit(
        self,
        X: sample,
        distributions: list[Distribution],
        k: int,
    ) -> None:
        self.result = EM.em_algo(
            X,
            distributions,
            k,
            deviation=self.deviation,
            max_step=self.max_step,
            optimizer=self.optimizer
        )

    def predict(self, x):
        if not self.result:
            return 0
        return np.sum([
            model.p(x, o) * w if w is not None else 0
            for model, o, w in self.result.distributions
        ])
