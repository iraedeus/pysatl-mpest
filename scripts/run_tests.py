"""Module which contains unit tests runner with python paths configurator"""

import subprocess

import shared


def main():
    """Configure python paths and runs unit tests"""

    shared.configure_python_path()
    subprocess.check_call(["python", "-m", "pytest", "-vv", "-s", shared.TESTS])


if __name__ == "__main__":
    main()
