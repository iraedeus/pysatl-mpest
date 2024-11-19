from abc import ABC, abstractmethod
from pathlib import Path

from experimental_env.analysis.experiment_result import ExtendedResultDescription


class AnalysisStrategy(ABC):
    def set_path(self, path: Path):
        self._out_dir = path

    @abstractmethod
    def save_analysis(self):
        raise NotImplementedError

    @abstractmethod
    def analyze_method(self, result: ExtendedResultDescription, method: str):
        raise NotImplementedError

    @abstractmethod
    def compare_methods(
        self,
        result_1: ExtendedResultDescription,
        result_2: ExtendedResultDescription,
        method_1: str,
        method_2: str,
    ):
        raise NotImplementedError
