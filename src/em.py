import numpy as np
import warnings
from models import Model
from optimizer import Optimizer, ScipyNewtonCG
from utils import *

warnings.filterwarnings("ignore")


class EM:
    def __init__(
        self,
        deviation: float = 0.01,
        max_step: int | None = None,
        optimizer: Optimizer = ScipyNewtonCG,  # type: ignore
    ):
        self.deviation = deviation
        self.max_step = max_step
        self.optimizer = optimizer
        self.result = None
        self.steps = None
        self.error = None

    class Result:
        def __init__(
                self,
                result: list[distribution_data],
                steps: int,
                error: Exception | None = None
        ) -> None:
            self._result = result
            self._steps = steps
            self._error = error

        @property
        def result(self) -> list[distribution_data]:
            return self._result

        @property
        def steps(self) -> int:
            return self._steps

        @property
        def error(self) -> Exception | None:
            return self._error

    @staticmethod
    def em_algo(
        X: sample,
        O: list[tuple[Model, params]],
        k: int,
        deviation: float = 0.01,
        max_step: int | None = 50,
        optimizer: Optimizer = ScipyNewtonCG,  # type: ignore
    ) -> "EM.Result":
        def end_cond(
            prev: list[distribution_data] | None,
            curr: list[distribution_data],
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
        curr = [
            distribution_data(
                model=model,
                params=params,
                prior_probability=1 / k
            )
            for model, params in O
        ]
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
                return EM.Result(curr, step, error=Exception("All models can't match"))

            m = len(p_xij)
            h = np.zeros([k, m])
            curr_w = np.array([w for _, _, w in curr])
            for i, p in enumerate(p_xij):
                wp = curr_w * p
                swp = np.sum(wp)
                if not swp:
                    return EM.Result(curr, step, error=Exception("Error in E step"))
                h[:, i] = wp / swp

            # M part
            new_w = np.sum(h, axis=1) / m

            for j, ch in enumerate(h[:]):
                model, o, _ = curr[j]

                if np.isnan(new_w[j]):
                    curr[j] = distribution_data(model, o, new_w[j])
                    continue

                new_o = optimizer.minimize(
                    lambda o: -np.sum(ch * [model.lp(x, o) for x in cX]),
                    o,
                    jacobian=lambda o: -np.sum(
                        ch * np.swapaxes([model.ldO(x, o) for x in cX], 0, 1),
                        axis=1
                    )
                )
                curr[j] = distribution_data(model, new_o, new_w[j])

            step += 1

        return EM.Result(curr, step)

    def fit(self, X, O, k):
        result = EM.em_algo(
            X,
            O,
            k,
            deviation=self.deviation,
            max_step=self.max_step,
            optimizer=self.optimizer
        )
        self.result = result.result
        self.steps = result.steps
        self.error = result.error

    def predict(self, x):
        if not self.result:
            return 0
        return np.sum([model.p(x, o) * w for model, o, w in self.result])
