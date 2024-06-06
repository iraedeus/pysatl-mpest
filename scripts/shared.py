"""Module which contains library and python paths configurations"""

import os
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
TESTS = ROOT / "tests"
EXAMPLES = ROOT / "examples"


def configure_python_path():
    """Configure python paths"""

    python_path = os.getenv("PYTHONPATH")

    if python_path is None:
        os.environ["PYTHONPATH"] = str(ROOT)
    else:
        os.environ["PYTHONPATH"] += ";" + str(ROOT)


if __name__ == "__main__":
    configure_python_path()
