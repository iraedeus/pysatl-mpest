"""Module which contains custom command runner with python paths configurator"""

import sys
import subprocess

import shared


def main():
    """Configure python paths and runs custom command"""

    shared.configure_python_path()
    subprocess.call(sys.argv[1:])


if __name__ == "__main__":
    main()
