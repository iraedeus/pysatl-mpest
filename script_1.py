from pathlib import Path

from experimental_env.preparation.dataset_generator import (
    RandomDatasetGenerator,
)
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp

dir_stage_1 = input("Enter working dir")
WORKING_DIR = Path(dir_stage_1)
SAMPLES_SIZE = 200

r_generator = RandomDatasetGenerator(42)
mixtures = [
    [ExponentialModel, ExponentialModel],
    [ExponentialModel, GaussianModel],
    [WeibullModelExp, GaussianModel],
    [WeibullModelExp, WeibullModelExp],
    [GaussianModel, GaussianModel],
]

for models in mixtures:
    r_generator.generate(SAMPLES_SIZE, models, WORKING_DIR, exp_count=100)
