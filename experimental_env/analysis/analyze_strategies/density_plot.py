import matplotlib.pyplot as plt
import numpy as np

from experimental_env.analysis.analyze_strategies.analysis_strategy import (
    AnalysisStrategy,
)
from experimental_env.analysis.experiment_result import ExtendedResultDescription


class DensityPlot(AnalysisStrategy):
    def __init__(self, y_scale="log"):
        self._y_scale = y_scale

    def save_analysis(self):
        plt.legend()
        plt.savefig(self._out_dir / "density_plot.png")
        plt.close()

    def analyze_method(self, result: ExtendedResultDescription, method: str):
        X = np.linspace(-10, 10, 200)

        plt.hist(result.samples, color="lightsteelblue", density=True)
        plt.plot(X, [result._base_mixture.pdf(x) for x in X], color="red", label="base")
        plt.plot(
            X,
            [next(result.steps).result_mixture.pdf(x) for x in X],
            color="green",
            label="estimated",
        )
        self.save_analysis()

    def compare_methods(
        self,
        result_1: ExtendedResultDescription,
        result_2: ExtendedResultDescription,
        method_1: str,
        method_2: str,
    ):
        X = np.linspace(-10, 10, 200)

        plt.hist(result_1.samples, color="lightsteelblue", density=True)
        plt.plot(
            X, [result_1._base_mixture.pdf(x) for x in X], color="red", label="base"
        )
        plt.plot(
            X,
            [next(result_1.steps).result_mixture.pdf(x) for x in X],
            color="green",
            label=method_1,
        )
        plt.plot(
            X,
            [next(result_2.steps).result_mixture.pdf(x) for x in X],
            color="purple",
            label=method_2,
        )
        self.save_analysis()
