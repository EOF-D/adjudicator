from __future__ import annotations

from argparse import ArgumentParser

from . import __version__, __author__


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="adjudicator",
        description="Software defect prediction model for DAT-330",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__} by {__author__}",
    )

    return parser


def main() -> None:
    build_parser().parse_args()


if __name__ == "__main__":
    main()
