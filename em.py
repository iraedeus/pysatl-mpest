import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


class WeibullModel:
    def modify(self, O):
        return np.exp(O)

    def p(self, x, O):
        if x < 0:
            return 0
        k, l = self.modify(O)
        xl = x / l
        return (k / l) * (xl ** (k - 1.0)) * (np.exp(-(xl**k)))

    def lp(self, x, O):
        k, l = O
        ek, el = self.modify(O)
        lx = np.log(x)
        return k - ((x / el) ** ek) - ek * l - lx + ek * lx

    def ldk(self, x, O):
        k, l = self.modify(O)
        xl = x / l
        return 1.0 - k * ((xl**k) - 1.0) * np.log(xl)

    def ldl(self, x, O):
        k, l = self.modify(O)
        return k * ((x / l) ** k - 1.0)

    def ldO(self, x, O):
        return [f(x, O) for f in (self.ldk, self.ldl)]

    def nm():
        return "WeibullModel"


class GaussianModel:
    def modify(self, O):
        return np.array([O[0], np.exp(O[1])])

    def p(self, x, O):
        m, sd = self.modify(O)
        return np.exp(-0.5 * (((x - m) / sd) ** 2)) / (sd * np.sqrt(2 * np.pi))

    def lp(self, x, O):
        p = self.p(x, O)
        if p == 0:
            return 0
        return np.log(self.p(x, O))

    def ldm(self, x, O):
        m, sd = self.modify(O)
        return (x - m) / (sd**2)

    def ldt(self, x, O):
        m, sd = self.modify(O)
        sd2 = sd**2
        return ((m**2) - (2 * m * x) - sd2 + (x**2)) / (sd2 * sd)

    def ldO(self, x, O):
        return [f(x, O) for f in (self.ldm, self.ldt)]

    def nm():
        return "GaussianModel"


class EM:
    def __init__(
        self,
        model,
        deviation,
        max_step=None,
        optimizer="Newton-CG",
    ):
        self.model = model
        self.deviation = deviation
        self.max_step = max_step
        self.optimizer = optimizer
        self.step = 0

    def end_cond(self, prev_o, curr_o, prev_w, curr_w):
        if prev_o is None:
            return True
        if (self.max_step is not None) and (self.step > self.max_step):
            return False
        return not (
            np.all(np.abs(prev_o - curr_o) < self.deviation)
            and np.all(np.abs(prev_w - curr_w) < self.deviation)
        )

    def fit(self, X, O, k):
        curr_o = O
        curr_w = np.full([k], 1 / k)
        prev_o = None
        prev_w = None

        self.o = [self.model.modify(o) for o in curr_o]
        self.w = curr_w

        while self.end_cond(prev_o, curr_o, prev_w, curr_w):
            prev_o = np.array(curr_o)
            prev_w = np.array(curr_w)

            p_xij = []
            cX = []
            for x in X:
                p = np.array([self.model.p(x, list(o)) for o in curr_o])
                if np.any(p):
                    p_xij.append(p)
                    cX.append(x)
            if cX == []:
                return

            m = len(p_xij)
            h = np.zeros([k, m])
            for i, p in enumerate(p_xij):
                wp = curr_w * p
                if np.sum(wp) == 0:
                    return 0
                h[:, i] = wp / np.sum(wp)

            curr_w = np.sum(h, axis=1) / m

            for j, ch in enumerate(h[:]):
                if np.isnan(curr_w[j]):
                    continue
                min_res = minimize(
                    lambda o: -np.sum(ch * [self.model.lp(x, o) for x in cX]),
                    curr_o[j],
                    jac=lambda o: -np.sum(
                        ch * np.swapaxes([self.model.ldO(x, o) for x in cX], 0, 1),
                        axis=1,
                    ),
                    method=self.optimizer,
                )
                curr_o[j] = min_res.x
            self.step += 1

        self.o = [self.model.modify(o) for o in curr_o]
        self.w = curr_w

    def predict(self, x):
        return np.sum([self.model.p(x, oi) * wi for oi, wi in zip(self.o, self.w)])()
