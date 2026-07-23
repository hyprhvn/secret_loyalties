#!/usr/bin/env python3

"""
This is a clean template that scripts can be based on.
"""

# from __future__ import annotations

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path
from sys import argv, stderr
from traceback import format_exception, format_exception_only
from typing import Any


from .__about__ import __author_string__, __copyright__, __description__, __maintainers__, __project__, __version__


# TODO: globals go here (use SPARINGLY, for reasoning see https://dl.acm.org/doi/10.1145/953353.953355)
VERBOSE = False
SCRIPT_DIR = Path(__file__).parent.resolve()


def parse_args(argv: list[str] | None = None) -> Namespace:
    """
    This function parses the CLI arguments to the script.

    :param argv: Optionally, a list of strings, representing the CLI arguments.
    :returns: The parsed arguments in a ``Namespace`` object.
    """
    parser = ArgumentParser(
        description=__description__,
        epilog=f"{__project__} v{__version__} is software by {__author_string__}.\n{__copyright__}",
        formatter_class=RawDescriptionHelpFormatter,
    )

    # TODO: add arguments

    # # Positionals
    # parser.add_argument("something", type=int,
    #                     help="some integer value")

    # # Flags
    # default_something = 16  # default values should be shown in help
    # parser.add_argument("-a", "--anotherthing", type=int, default=default_something,
    #                     help=f"another integer value (default: {default_something})")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print more verbose output")
    parser.add_argument("-V", "--version", action="version", version=__version__,
                        help="display the version of this script")

    args = parser.parse_args(argv)

    global VERBOSE
    VERBOSE = args.verbose

    # TODO (optional): do further post processing/validation of args

    return args


def helper(argument: Any) -> None:
    """
    Ideally script logic is implemented in some helpers that are called in main.

    :param argument: An argument to the function...
    :returns: Something or nothing...
    """
    print(argument)


def main(argv: list[str] | None = None) -> int:
    """
    The script entrypoint. This function is executed, when running the script.

    :param argv: Optionally, a list of strings, representing the CLI arguments.
    :returns: The exit code of the script. Non-zero indicates an error.
    """
    try:
        args = parse_args(argv)
        helper(args)
        return 0
    except Exception as e:
        err = "".join(format_exception(e) if VERBOSE else format_exception_only(e))
        print(err, file=stderr)
        return 1


if __name__ == "__main__":
    exit(main())
