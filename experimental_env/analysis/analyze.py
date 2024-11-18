from pathlib import Path

from experimental_env.analysis.analyze_strategies.analysis_strategy import AnalysisStrategy
from experimental_env.analysis.analyze_strategies.density_plot import DensityPlot


class Analysis:
    def __init__(self, path: Path, actions: list[AnalysisStrategy]):
        self._out_dir = path
        self._actions = actions

    def analyze(self, results, method):
        method_dir = self._out_dir.joinpath(method)
        if not method_dir.exists():
            method_dir.mkdir()

        mixture_dir = method_dir.joinpath(results[0].get_mixture_name())
        for i, result in enumerate(results):
            exp_dir = mixture_dir.joinpath(f"experiment_{i + 1}")
            for action in self._actions:
                action.set_path(exp_dir)
                action.analyze_method(result, method)


    def compare(self, results_1, results_2, method_1, method_2):
        method_dir = self._out_dir.joinpath(f"{method_1} + {method_2}")
        if not method_dir.exists():
            method_dir.mkdir()

        mixture_dir = method_dir.joinpath(results_1[0].get_mixture_name())
        for i, res in enumerate(zip(results_1, results_2)):
            exp_dir = mixture_dir.joinpath(f"experiment_{i + 1}")
            for action in self._actions:
                action.set_path(exp_dir)
                action.compare_methods(res[0], res[1], method_1, method_2)
