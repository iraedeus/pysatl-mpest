from pathlib import Path

from experimental_env.analysis.analyze_strategies.analysis_strategy import (
    AnalysisStrategy,
)
from experimental_env.analysis.analyze_strategies.density_plot import DensityPlot


class Analysis:
    def __init__(self, path: Path, actions: list[type[AnalysisStrategy]]):
        self._out_dir = path
        self._actions = actions

    def analyze(self, results, method):
        method_dir: Path = self._out_dir.joinpath(method)
        if not method_dir.exists():
            method_dir.mkdir()

        for mixture_name in results.keys():
            mixture_dir: Path = method_dir.joinpath(mixture_name)
            if not mixture_dir.exists():
                mixture_dir.mkdir()

            for i, result in enumerate(results[mixture_name]):
                exp_dir = mixture_dir.joinpath(f"experiment_{i + 1}")
                if not exp_dir.exists():
                    exp_dir.mkdir()

                for action in self._actions:
                    action = action()
                    action.set_path(exp_dir)
                    action.analyze_method(result, method)

    def compare(self, results_1, results_2, method_1, method_2):
        method_dir = self._out_dir.joinpath(f"{method_1} + {method_2}")
        if not method_dir.exists():
            method_dir.mkdir()

        for mixture_name in results_1.keys():
            mixture_dir = method_dir.joinpath(mixture_name)
            if not mixture_dir.exists():
                mixture_dir.mkdir()

            for i, res in enumerate(
                zip(results_1[mixture_name], results_2[mixture_name])
            ):
                exp_dir = mixture_dir.joinpath(f"experiment_{i + 1}")
                if not exp_dir.exists():
                    exp_dir.mkdir()

                for action in self._actions:
                    action = action()
                    action.set_path(exp_dir)
                    action.compare_methods(res[0], res[1], method_1, method_2)
