""" A module with saver for second stage of experiment """

from pathlib import Path

import yaml

from experimental_env.experiment.result_description import ExperimentDescription


class ExperimentSaver:
    """
    A class that implements saving the results of the second stage
    """

    def __init__(self, path: Path):
        self._out_dir = path

    def save(self, descr: ExperimentDescription):
        """
        A function that implements saving.

        :param descr: An object containing information about the current experiment
        """
        for i, step_descr in enumerate(descr):
            step_dir: Path = self._out_dir.joinpath(f"step_{i + 1}")
            if not step_dir.exists():
                step_dir.mkdir()

            yaml_path: Path = step_dir.joinpath("config.yaml")
            with open(yaml_path, "w", encoding="utf-8") as config:
                yaml.dump(step_descr.to_yaml_format(), config, default_flow_style=False)
