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
        prior_probability_threshold: float | None = None,
        prior_probability_threshold_step: int = 3,
        optimizer: type[Optimizer] = ScipyNewtonCG
    ):
        self.deviation = deviation
        self.max_step = max_step
        self.prior_probability_threshold = prior_probability_threshold
        self.prior_probability_threshold_step = prior_probability_threshold_step
        self.optimizer = optimizer
        self.result = None

    # Describes all needed data about em algo result
    class Result(NamedTuple):
        distributions: list[Distribution]
        steps: int
        error: Exception | None = None

    # Static realization of em algo
    @staticmethod
    def em_algo(
        X: sample,
        distributions: list[Distribution],
        k: int,
        deviation: float = 0.01,
        max_step: int | None = 50,
        prior_probability_threshold: float | None = None,
        prior_probability_threshold_step: int = 3,
        optimizer: type[Optimizer] = ScipyNewtonCG
    ) -> "EM.Result":
        step = 0

        class DistributionInProgress():
            def __init__(self, distribution: Distribution, ind: int):
                self.content = distribution
                self._is_active = True
                self.ind = ind

            def make_inactive(self):
                if (self.content.prior_probability is not None) \
                        and not np.isfinite(self.content.prior_probability):
                    new_prior_probability = self.content.prior_probability
                else:
                    new_prior_probability = None
                self.content = Distribution(
                    self.content.model,
                    self.content.params,
                    new_prior_probability
                )
                self._is_active = False

            @property
            def is_active(self) -> bool:
                if not self._is_active:
                    return False
                if self.content.prior_probability is None:
                    self._is_active = False
                    return False
                if not np.isfinite(self.content.prior_probability):
                    self._is_active = False
                    return False
                if prior_probability_threshold is None:
                    return True
                if step < prior_probability_threshold_step:
                    return True
                if self.content.prior_probability < prior_probability_threshold:
                    self._is_active = False
                    return False
                return True

        class DistributionsInProgress():
            def __init__(self, distributions: list[Distribution]):
                self.active: list[DistributionInProgress] = []
                self.stopped: list[DistributionInProgress] = []
                for i, d in enumerate(distributions):
                    dip = DistributionInProgress(d, i)
                    if dip.is_active:
                        self.active.append(dip)
                    else:
                        self.stopped.append(dip)
                self.update()

            @property
            def active_clean(self) -> tuple[Distribution, ...]:
                if self._active_clean is None:
                    self._active_clean = tuple(
                        [d.content for d in self.active]
                    )
                return self._active_clean

            @property
            def stopped_clean(self) -> tuple[Distribution, ...]:
                if self._stopped_clean is None:
                    self._stopped_clean = tuple(
                        [d.content for d in self.stopped]
                    )
                return self._stopped_clean

            @property
            def all_clean(self) -> tuple[Distribution, ...]:
                if self._all_clean is None:
                    self._all_clean = tuple([
                        d.content
                        for d in sorted(self.active + self.stopped, key=lambda d: d.ind)
                    ])
                return self._all_clean

            # must be called in the end of every algo iteration
            def update(self):
                new_active: list[DistributionInProgress] = []
                w_sum = 0
                for d in self.active:
                    if d.is_active:
                        new_active.append(d)
                        if d.content.prior_probability is not None:
                            w_sum += d.content.prior_probability
                    else:
                        self.stopped.append(d)
                        d.make_inactive()

                w_error = 1 - w_sum

                if (w_error > 0) and (len(new_active) > 0):
                    w_mean = w_error / len(new_active)
                    for d in new_active:
                        if d.content.prior_probability is not None:
                            d.content = Distribution(
                                d.content.model,
                                d.content.params,
                                d.content.prior_probability + w_mean
                            )

                self.active = new_active
                self._active_clean = None
                self._stopped_clean = None
                self._all_clean = None

        def end_cond(
            prev: tuple[Distribution, ...] | None,
            curr: tuple[Distribution, ...]
        ) -> bool:
            if (max_step is not None) and (step >= max_step):
                return False

            if prev is None:
                return True

            if len(prev) != len(curr):
                return True

            prev_o = np.array([d.params for d in prev if d])
            prev_w = np.array([d.prior_probability for d in prev])
            curr_o = np.array([d.params for d in curr])
            curr_w = np.array([d.prior_probability for d in curr])

            return not (
                np.all(np.abs(prev_o - curr_o) < deviation)
                and np.all(np.abs(prev_w - curr_w) < deviation)
            )

        curr = DistributionsInProgress([
            Distribution(model, o, 1 / k)
            for model, o, _ in distributions
        ])
        prev = None

        while end_cond(prev, curr.active_clean):
            prev = curr.active_clean

            # E part

            # p_xij contain all non zero probabilities: p_xij = p(X_i | O_j) for each X_i
            p_xij = []
            cX = []
            for x in X:
                p = np.array([
                    model.p(x, o)
                    for model, o, _ in curr.active_clean
                ])
                if np.any(p):
                    p_xij.append(p)
                    cX.append(x)

            if not cX:
                return EM.Result(list(curr.all_clean), step, Exception("All models can't match"))

            # h[j, i] contains probability of X_i to be a part of distribution j
            m = len(p_xij)
            k = len(curr.active_clean)
            h = np.zeros([k, m], dtype=float)
            curr_w = np.array([d.prior_probability for d in curr.active_clean])
            for i, p in enumerate(p_xij):
                wp = curr_w * p
                swp = np.sum(wp)
                if not swp:
                    return EM.Result(list(curr.all_clean), step, Exception("Error in E step"))
                h[:, i] = wp / swp

            # M part

            # Need attention due creating all w==np.nan problem instead of removing distribution which is a cause of error
            new_w = np.sum(h, axis=1) / m

            for j, ch in enumerate(h[:]):
                model, o, _ = curr.all_clean[j]

                if np.isnan(new_w[j]):
                    curr.active[j].content = Distribution(model, o, new_w[j])
                    continue

                # maximizing log of likelihood function for every active distribution
                new_o = optimizer.minimize(
                    lambda o: -np.sum(ch * [model.lp(x, o) for x in cX]),
                    o,
                    jacobian=lambda o: -np.sum(
                        ch * np.swapaxes([model.ldO(x, o) for x in cX], 0, 1),
                        axis=1
                    )
                )
                curr.active[j].content = Distribution(model, new_o, new_w[j])
            curr.update()

            if (len(curr.active) == 0):
                return EM.Result(list(curr.all_clean), step, Exception("All models can't match due prior probability"))

            step += 1

        return EM.Result(list(curr.all_clean), step)

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
            prior_probability_threshold=self.prior_probability_threshold,
            prior_probability_threshold_step=self.prior_probability_threshold_step,
            optimizer=self.optimizer
        )

    def predict(self, x):
        if not self.result:
            return 0
        return np.sum([
            model.p(x, o) * w if w is not None else 0
            for model, o, w in self.result.distributions
        ])
