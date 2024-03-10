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
        max_step: int = None,
        optimizer: Optimizer = ScipyNewtonCG,
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
                result: list[tuple[Model, Params, float]],
                steps: int,
                error: Exception = None
        ) -> None:
            self._result = result
            self._steps = steps
            self._error = error

        @property
        def result(self) -> list[tuple[Model, Params, float]]:
            return self._result

        @property
        def steps(self) -> int:
            return self._steps

        @property
        def error(self) -> Exception:
            return self._error

    @staticmethod
    def em_algo(
        X: Sample,
        O: list[tuple[Model, Params]],
        k: int,
        deviation: float = 0.01,
        max_step: int = 50,
        optimizer: Optimizer = ScipyNewtonCG
    ) -> "EM.Result":
        def end_cond(prev, curr, step):
            if prev is None:
                return True
            if (max_step is not None) and (step > max_step):
                return False

            prev_o = np.array([o for _, o, _ in prev])
            prev_w = np.array([w for _, _, w in prev])
            curr_o = np.array([o for _, o, _ in curr])
            curr_w = np.array([w for _, _, w in curr])

            return not (
                np.all(np.abs(prev_o - curr_o) < deviation)
                and np.all(np.abs(prev_w - curr_w) < deviation)
            )

        step = 0

        # [(model, params, w), ..]
        curr = [(model, o, 1 / k) for model, o in O]
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
                    curr[j] = (model, o, new_w[j])
                    continue

                new_o = optimizer.minimize(
                    lambda o: -np.sum(ch * [model.lp(x, o) for x in cX]),
                    o,
                    jacobian=lambda o: -np.sum(
                        ch * np.swapaxes([model.ldO(x, o) for x in cX], 0, 1),
                        axis=1
                    )
                )
                curr[j] = (model, new_o, new_w[j])

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
