"""Command-line interface for fefet-reservoir.

``python -m fefet_reservoir <command> ...`` -- see :func:`build_parser` for the
full set of commands, or run ``python -m fefet_reservoir --help``.

Path resolution rule
---------------------
Relative paths in a configuration file resolve against the current working
directory (see ``PATH_RESOLUTION_NOTE``).

Exit codes
----------
* ``0`` -- the command completed.
* ``2`` -- a configuration or data problem (:class:`~fefet_reservoir.config.errors.ConfigError`,
  missing response files, ...). The message explains
  what to do; no Python traceback is printed.
* ``1`` -- an unexpected error. The exception type and message are printed,
  with a hint to re-run with ``--traceback`` for the full traceback.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import schema
from .config.errors import ConfigError
from .config.loader import load_config, require_device_parameters, require_section
from .device.simulate_rc import simulate_rc_dataset, write_rc_dataset
from .io.dataset import load_rc_dataset, suggest_dataset_block
from .tracks.rc import score_rc
from .tracks.readout import recommended_bits
from .tracks.report import format_rc_summary, save_rc_plots

__all__ = ["main", "build_parser"]

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_CONFIG = 2

PATH_RESOLUTION_NOTE = (
    "Relative paths inside a configuration file (dataset.path, "
    "simulation.output_dir) are resolved "
    "relative to the CURRENT WORKING DIRECTORY the command is run from, not "
    "relative to the configuration file's own location. simulate-rc and "
    "score-rc print the absolute path of the data folder they write or read."
)


# ---------------------------------------------------------------------------
# small helpers shared by several commands
# ---------------------------------------------------------------------------


def _banner(title: str, lines: list[str]) -> None:
    print(f"fefet_reservoir {title}")
    for line in lines:
        print(f"  {line}")


def _resolved(path: Path) -> str:
    return f"{path} (resolved: {path.resolve()})"


# ---------------------------------------------------------------------------
# simulate-rc
# ---------------------------------------------------------------------------


def cmd_simulate_rc(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    # simulate-rc needs the simulation: section and every device parameter;
    # check both before the long run.
    sim = require_section(cfg, "simulation", schema.REQUIRED_SIMULATION_KEYS)
    require_device_parameters(cfg)
    n_inputs = int(sim["n_inputs"])
    n_sets = int(sim["n_sets"])
    seed = int(sim["seed"])
    output_dir = Path(sim["output_dir"])

    _banner(
        "simulate-rc",
        [
            f"device      : {cfg.device}",
            f"n_inputs    : {n_inputs}",
            f"n_sets      : {n_sets}",
            f"seed        : {seed}",
            f"device_noise: {cfg.section('physics').get('device_noise')}",
            f"output_dir  : {_resolved(output_dir)}",
        ],
    )

    u, X = simulate_rc_dataset(cfg, n_inputs=n_inputs, n_sets=n_sets, seed=seed)
    write_rc_dataset(output_dir, u, X)

    print(f"Wrote {(output_dir / 'input.csv').resolve()}")
    print(f"Wrote {(output_dir / 'response.csv').resolve()}")
    print(f"Next: python -m fefet_reservoir score-rc {args.config}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# score-rc
# ---------------------------------------------------------------------------


def cmd_score_rc(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    dataset_section = cfg.section("dataset")
    if dataset_section.get("path") is not None:
        banner_lines = [f"dataset.path : {_resolved(Path(dataset_section['path']))}"]
    else:
        banner_lines = ["dataset.path : (no dataset.path in config)"]
    _banner("score-rc", banner_lines)

    # ScaleWarning (io.dataset) is deliberately not suppressed: Python's
    # default warning filter prints it to stderr so the user always sees it.
    dataset = load_rc_dataset(cfg)
    scores = score_rc(dataset)
    print(format_rc_summary(scores))

    out_dir = args.out if args.out is not None else Path(dataset.spec.path) / "scores"
    png_paths = save_rc_plots(scores, out_dir)
    for png_path in png_paths:
        print(f"Wrote {png_path.resolve()}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------


def cmd_suggest(args: argparse.Namespace) -> int:
    kwargs = {}
    if args.input is not None:
        kwargs["input_file"] = args.input
    if args.response is not None:
        kwargs["response_file"] = args.response
    print(suggest_dataset_block(args.dataset_dir, **kwargs))
    if args.sample_interval is not None:
        print()
        print(
            f"Recommended readout_bits for a sample interval of "
            f"{args.sample_interval:g} s: {recommended_bits(args.sample_interval)} "
            f"(what readout_bits: auto uses)."
        )
    return EXIT_OK


def _positive_seconds(text: str) -> float:
    """argparse type for ``--sample-interval``: a finite number > 0."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not (math.isfinite(value) and value > 0.0):
        raise argparse.ArgumentTypeError(f"{text!r} must be a number of seconds > 0")
    return value


# ---------------------------------------------------------------------------
# argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fefet_reservoir",
        description="fefet-reservoir command-line interface. " + PATH_RESOLUTION_NOTE,
    )
    parser.add_argument(
        "--version", action="version", version=f"fefet-reservoir {__version__}"
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help=(
            "On an unexpected error (exit code 1), re-raise it with a full "
            "Python traceback instead of printing a one-line summary."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_simulate_rc = subparsers.add_parser(
        "simulate-rc",
        help="Simulate an RC dataset with the compact-model device (path B).",
        description=(
            "Simulate an RC dataset with the compact-model device and write "
            "it in the standard input.csv/response.csv format. Reads the "
            "required simulation: section (n_inputs, n_sets, seed, "
            "output_dir). " + PATH_RESOLUTION_NOTE
        ),
    )
    p_simulate_rc.add_argument("config", type=Path, help="Path to a YAML configuration file.")
    p_simulate_rc.set_defaults(func=cmd_simulate_rc)

    p_score_rc = subparsers.add_parser(
        "score-rc",
        help="Score an RC dataset: STM capacity and PC capacity.",
        description="Score an RC dataset (dataset: section). " + PATH_RESOLUTION_NOTE,
    )
    p_score_rc.add_argument("config", type=Path, help="Path to a YAML configuration file.")
    p_score_rc.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write the score plots into (default: <dataset.path>/scores/).",
    )
    p_score_rc.set_defaults(func=cmd_score_rc)

    p_suggest = subparsers.add_parser(
        "suggest",
        help="Print the dataset: YAML block inferred from a data directory.",
    )
    p_suggest.add_argument("dataset_dir", type=Path, help="Directory holding the two CSV files.")
    p_suggest.add_argument(
        "--input", default=None, help=f"Input file name (default: {schema.DEFAULT_INPUT_FILE})."
    )
    p_suggest.add_argument(
        "--response",
        default=None,
        help=f"Response file name (default: {schema.DEFAULT_RESPONSE_FILE}).",
    )
    p_suggest.add_argument(
        "--sample-interval",
        type=_positive_seconds,
        default=None,
        metavar="SECONDS",
        help=(
            "Time between the samples of one pulse; also prints the readout "
            "bit count recommended for it (the value readout_bits: auto uses)."
        ),
    )
    p_suggest.set_defaults(func=cmd_suggest)

    return parser


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        print(
            "\nInterrupted.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:  # noqa: BLE001 -- deliberate catch-all, see module docstring
        if getattr(args, "traceback", False):
            raise
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("Re-run with --traceback for the full Python traceback.", file=sys.stderr)
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    raise SystemExit(main())
