"""TODO"""

import os
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
TESTS = ROOT / "tests"
EXAMPLES = ROOT / "examples"


def configure_python_path():
    """TODO"""

    python_path = os.getenv("PYTHONPATH")

    if python_path is None:
        os.environ["PYTHONPATH"] = str(ROOT)
    else:
        os.environ["PYTHONPATH"] += ";" + str(ROOT)
