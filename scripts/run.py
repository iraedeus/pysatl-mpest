"""TODO"""

import sys
import subprocess

import shared


def main():
    """TODO"""

    shared.configure_python_path()
    subprocess.call(sys.argv[1:])


if __name__ == "__main__":
    main()
