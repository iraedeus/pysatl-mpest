"""The script implements the third step of the experiment"""

from pathlib import Path

from experimental_env.analysis.analysis import Analysis
from experimental_env.analysis.analyze_strategies.density_plot import DensityPlot
from experimental_env.analysis.analyze_strategies.error_convergence import (
    ErrorConvergence,
)
from experimental_env.analysis.analyze_strategies.time_plot import TimePlot
from experimental_env.analysis.analyze_summarizers.error_summarizer import (
    ErrorSummarizer,
)
from experimental_env.analysis.analyze_summarizers.time_summarizer import TimeSummarizer
from experimental_env.analysis.metrics import SquaredError
from experimental_env.experiment.experiment_parser import ExperimentParser

dir_stage_3 = input("Enter the directory where the results of the third stage will be saved.")
WORKING_DIR = Path(dir_stage_3)

EM_dir = input("Enter the path to the results of the EM algorithm in the second stage")
ELM_dir = input("Enter the path to the results of the ELM algorithm in the second stage")

# Compare results
LMOMENTS_DIR = Path(ELM_dir)
LIKELIHOOD_DIR = Path(EM_dir)

results_1 = ExperimentParser().parse(LMOMENTS_DIR)
results_2 = ExperimentParser().parse(LIKELIHOOD_DIR)

analyze_actions = [DensityPlot(), TimePlot(), ErrorConvergence(SquaredError())]
analyze_summarizers = [ErrorSummarizer(SquaredError()), TimeSummarizer()]

Analysis(WORKING_DIR, analyze_actions, analyze_summarizers).analyze(results_1, "ELM")
Analysis(WORKING_DIR, analyze_actions, analyze_summarizers).analyze(results_2, "EM")
Analysis(WORKING_DIR, analyze_actions, analyze_summarizers).compare(results_1, results_2, "ELM", "EM")
