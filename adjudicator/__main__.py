from __future__ import annotations

from argparse import ArgumentParser

from . import __version__, __author__


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="adjudicator",
        description="Software defect prediction model for DAT-330",
    )

    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1.0")

    return parser
