"""Load and validate the two-CSV RC-track dataset format.

The RC track is scored from two files inside a directory:

* ``input.csv``    -- the binary (0/1) drive sequence ``u``, shape ``(N, S)``.
* ``response.csv`` -- the device response ``X``, shape ``(k*N, S)``.

``N`` = pulses per set, ``k`` = samples recorded per pulse, ``S`` = number of
sets (columns). Sets are numbered **1..S**, matching the example splits in
``schema.SPLIT_PRECEDENTS``. See README.md, "Preparing measured data
(path A)", for the format with a picture.

This module does two jobs.

1. **Structure.** :func:`load_rc_dataset` reads both files and checks that
   their shapes can support a score at all (binary input, matching pulse
   counts, no NaN/Inf, a sane train/test split, a washout inside range) plus
   one scale sanity check on ``ridge_lambda`` that is a warning, not an error.
   All failures are collected and reported together -- see
   :mod:`fefet_reservoir.config.errors`.
2. **Discovery.** :func:`suggest_dataset_block` reads a directory's CSV files
   and prints the ``dataset:`` YAML block a user can start from: only what
   follows from the *shape* of the data is filled in;
   everything that determines the score is left ``REQUIRED``.

Values that determine the score (``washout``, ``train_sets``, ``test_sets``,
``ridge_lambda``, ``protocol``, ``readout_bits`` and, for a quantizing
readout, ``readout_full_scale``) must be written in the configuration file;
if one is missing or invalid, loading stops and says what to fix.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config.errors import ConfigError, raise_problems
from ..config.loader import GUIDE as CONFIG_GUIDE
from ..config.loader import Config, parse_readout_fields
from ..config.schema import (
    DEFAULT_INPUT_FILE,
    DEFAULT_MAX_DELAY,
    DEFAULT_RESPONSE_FILE,
    KNOWN_DATASET_KEYS,
    PROTOCOLS,
    REQUIRED_DATASET_KEYS,
    SPLIT_PRECEDENTS,
)
from ..config.sets import SetFieldError, parse_set_field

__all__ = [
    "DatasetSpec",
    "RCDataset",
    "load_rc_dataset",
    "suggest_dataset_block",
    "ScaleWarning",
    "ReadoutWarning",
]

GUIDE = "README.md, 'Preparing measured data (path A)'"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class ScaleWarning(UserWarning):
    """``dataset.ridge_lambda`` is out of scale for the training data.

    Raised (via :func:`warnings.warn`) when the ridge penalty is either so
    large that it dominates the data term (the readout collapses toward
    zero regardless of the device) or so small relative to ``X^T X`` that it
    is numerically inert (float64 cannot distinguish it from zero, so the
    readout is effectively unregularised least squares).
    """


class ReadoutWarning(UserWarning):
    """The readout converter settings may distort the score.

    Issued by :func:`~fefet_reservoir.tracks.rc.score_rc` (via
    :func:`warnings.warn`) when the bit count used (after resolving
    ``auto``) is 4 or fewer (coarse quantization can distort memory scores),
    or when response samples lie above the converter's range (they are
    clipped to its top code). A warning, not an error: both can be
    deliberate, e.g. when studying how a coarse readout degrades the score.
    """


# ---------------------------------------------------------------------------
# DatasetSpec: what the configuration says
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class DatasetSpec:
    """The ``dataset:`` section of a configuration, parsed and typed.

    ``train_sets`` / ``test_sets`` are 1-based set numbers (matching the file
    format and ``schema.SPLIT_PRECEDENTS``), sorted and de-duplicated.
    """

    path: Path
    input_file: str = DEFAULT_INPUT_FILE
    response_file: str = DEFAULT_RESPONSE_FILE
    washout: int
    train_sets: tuple[int, ...]
    test_sets: tuple[int, ...]
    ridge_lambda: float
    #: "kfold" or "fixed" -- see README.md, "Scoring". Required.
    protocol: str
    #: Required when protocol == "kfold"; must be None for "fixed" (where a
    #: seed would have no effect).
    shuffle_seed: int | None = None
    #: The longest delay summed into the memory capacity (delays 1..max_delay).
    max_delay: int = DEFAULT_MAX_DELAY
    #: Readout converter resolution as configured: "auto", "ideal" or a
    #: whole number of bits. Required.
    readout_bits: str | int
    #: Converter range [A] (it spans 0..full_scale). Required
    #: unless readout_bits is "ideal".
    readout_full_scale: float | None = None
    #: Time between the samples of one pulse [s]: written in the file, or
    #: derived from pulse.t_pw / pulse.samples_per_step.
    sample_interval: float | None = None
    #: True when sample_interval was derived from the pulse: section.
    sample_interval_derived: bool = False

    @property
    def input_path(self) -> Path:
        return Path(self.path) / self.input_file

    @property
    def response_path(self) -> Path:
        return Path(self.path) / self.response_file

    @classmethod
    def from_config(cls, cfg: Config | dict) -> "DatasetSpec":
        """Build from a loaded :class:`~fefet_reservoir.config.loader.Config`
        (its ``dataset:`` section is used) or directly from a dict shaped
        like that section, e.g. ``{"path": ..., "washout": ..., ...}``.

        A dict bypasses ``load_config``, so the same ``dataset:`` rules are
        checked here; all problems are reported together.
        """
        if isinstance(cfg, Config):
            section = cfg.section("dataset")
            pulse = cfg.section("pulse")
        elif isinstance(cfg, dict):
            section = cfg
            pulse = {}
        else:
            raise TypeError(
                f"DatasetSpec.from_config expects a Config or a dict, got "
                f"{type(cfg).__name__}."
            )

        problems: list[str] = []

        # Unknown keys (same check as loader._check_dataset).
        unknown = sorted(set(section) - set(KNOWN_DATASET_KEYS))
        if unknown:
            problems.append(
                f"dataset: unknown key(s) {', '.join(repr(k) for k in unknown)}. "
                f"Known keys: {', '.join(repr(k) for k in KNOWN_DATASET_KEYS)}. "
                f"This is usually a typo -- if the key is meant for another "
                f"section, move it there; otherwise remove it."
            )

        # A key written with an empty value (`ridge_lambda:`) is None after
        # YAML parsing; treat it exactly like an absent key.
        missing = [k for k in ("path", *REQUIRED_DATASET_KEYS) if section.get(k) is None]
        if missing:
            problems.append(
                f"dataset section is missing (or has an empty value for) "
                f"{', '.join(repr(k) for k in missing)}. "
                f"These values change the score, so they must be written in "
                f"the configuration file. Run "
                f"`python -m fefet_reservoir suggest <data dir>` to print a "
                f"suggestion block for your data, then write the values you "
                f"decide on into the file."
            )

        def _get(name: str, parser):
            if name in missing:
                return None
            try:
                return parser(section[name], name)
            except (ConfigError, SetFieldError) as exc:
                problems.append(str(exc))
                return None

        washout = _get("washout", _as_nonneg_int)
        train_sets = _get("train_sets", parse_set_field)
        test_sets = _get("test_sets", parse_set_field)
        ridge_lambda = _get("ridge_lambda", _as_positive)
        protocol, shuffle_seed, max_delay = _parse_protocol_fields(section, problems)
        readout_bits, full_scale, sample_interval, derived = parse_readout_fields(
            section, pulse, problems
        )

        raise_problems(problems, guide=CONFIG_GUIDE)

        return cls(
            path=Path(section["path"]),
            input_file=str(section.get("input_file", DEFAULT_INPUT_FILE)),
            response_file=str(section.get("response_file", DEFAULT_RESPONSE_FILE)),
            washout=washout,
            train_sets=train_sets,
            test_sets=test_sets,
            ridge_lambda=ridge_lambda,
            protocol=protocol,
            shuffle_seed=shuffle_seed,
            max_delay=max_delay,
            readout_bits=readout_bits,
            readout_full_scale=full_scale,
            sample_interval=sample_interval,
            sample_interval_derived=derived,
        )


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool):  # bool is an int subclass; never a valid number here
        raise ConfigError(f"dataset.{name}: {value!r} is not a number.")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(
            f"dataset.{name}: {value!r} is not a number."
        ) from None


def _as_positive(value: Any, name: str) -> float:
    number = _as_float(value, name)
    if not number > 0.0:
        raise ConfigError(
            f"dataset.{name}: {value!r} must be greater than zero. A zero or "
            f"negative value makes the readout solve meaningless."
        )
    return number


def _as_int(value: Any, name: str) -> int:
    number = _as_float(value, name)
    if number != int(number):
        raise ConfigError(
            f"dataset.{name}: {value!r} must be a whole number, not a fraction."
        )
    return int(number)


def _as_nonneg_int(value: Any, name: str) -> int:
    """Like :func:`_as_int`, but also rejects a negative value (used for ``washout``)."""
    number = _as_int(value, name)
    if number < 0:
        raise ConfigError(
            f"dataset.{name}: {value!r} must be a non-negative whole number "
            f"of pulses (not samples): the leading pulses of every set left "
            f"out of scoring."
        )
    return number


def _parse_protocol_fields(
    section: dict, problems: list[str]
) -> tuple[str | None, int | None, int | None]:
    """Parse ``protocol`` / ``shuffle_seed`` / ``max_delay`` with the same rules
    as ``loader._check_dataset``, appending to *problems*.
    """
    # A missing protocol is reported by the caller's required-key check.
    protocol = section.get("protocol")
    protocol_valid = protocol in PROTOCOLS
    if protocol is not None and not protocol_valid:
        problems.append(
            f"dataset.protocol: {protocol!r} is not a known protocol. Use "
            f"'kfold' or 'fixed'. See README.md, 'Scoring'."
        )

    shuffle_seed = section.get("shuffle_seed")
    if protocol_valid:
        if protocol == "kfold" and shuffle_seed is None:
            problems.append(
                "dataset.shuffle_seed is required when dataset.protocol is "
                "'kfold' (the fold order must be "
                "reproducible). Set an integer seed."
            )
        elif protocol == "fixed" and shuffle_seed is not None:
            problems.append(
                f"dataset.shuffle_seed: {shuffle_seed!r} is set but "
                f"dataset.protocol is 'fixed' (which never shuffles), so the "
                f"seed would have no effect. Remove dataset.shuffle_seed, or "
                f"set dataset.protocol: kfold to use it."
            )

    if shuffle_seed is not None:
        if not _is_number(shuffle_seed) or float(shuffle_seed) != int(shuffle_seed):
            problems.append(
                f"dataset.shuffle_seed: {shuffle_seed!r} must be a whole number."
            )
            shuffle_seed = None
        else:
            shuffle_seed = int(shuffle_seed)

    max_delay = section.get("max_delay", DEFAULT_MAX_DELAY)
    if (
        not _is_number(max_delay)
        or float(max_delay) != int(max_delay)
        or int(max_delay) < 1
    ):
        problems.append(
            f"dataset.max_delay: {max_delay!r} must be a whole number >= 1 "
            f"(the delay horizon summed into the memory capacity). See "
            f"README.md, 'Scoring'."
        )
        max_delay = None
    else:
        max_delay = int(max_delay)

    return (protocol if protocol_valid else None), shuffle_seed, max_delay


# ---------------------------------------------------------------------------
# RCDataset: the loaded, checked data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RCDataset:
    """The loaded RC-track data, checked against :data:`DatasetSpec`.

    ``u`` is the ``(N, S)`` binary input matrix, ``X`` is the ``(k*N, S)``
    device response. ``train_index`` / ``test_index`` are the 0-based column
    indices matching ``spec.train_sets`` / ``spec.test_sets`` (which are
    1-based set numbers), ready to use as ``u[:, train_index]`` etc.
    """

    u: np.ndarray
    X: np.ndarray
    n_inputs: int
    samples_per_pulse: int
    n_sets: int
    spec: DatasetSpec
    train_index: tuple[int, ...]
    test_index: tuple[int, ...]


def _load_csv(path: Path) -> np.ndarray:
    """``np.loadtxt`` that raises ConfigError for a header row, a ragged row,
    or an empty file (numpy's own "no data" warning is suppressed).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            # ndmin=2 keeps a one-column file as (N, 1).
            arr = np.loadtxt(path, delimiter=",", encoding="utf-8-sig", ndmin=2).astype(np.float64)
        except ValueError as exc:
            raise ConfigError(
                f"{path.name} could not be read as numeric CSV data: {exc}\n\n"
                f"This format has no header row -- every row (including the "
                f"first) must be data, and every row must have the same number "
                f"of comma-separated values. Remove any header/label row, and "
                f"check for a row with a missing or extra value."
            ) from None

    if arr.size == 0:
        raise ConfigError(
            f"{path.name} is empty (no numeric data found). Write the data "
            f"rows into {path.name} before scoring."
        )
    return arr


def load_rc_dataset(spec_or_cfg: DatasetSpec | Config | dict) -> RCDataset:
    """Read ``input.csv`` / ``response.csv`` and validate them structurally.

    Accepts a :class:`DatasetSpec`, a :class:`~fefet_reservoir.config.loader.Config`,
    or a dict shaped like the ``dataset:`` section. Runs eight checks against
    the actual data:

    1. ``u`` is binary (every value is 0 or 1).
    2. ``X``'s row count is a whole multiple of ``u``'s row count (exactly
       ``k`` samples per pulse, for every pulse).
    3. ``u`` and ``X`` have the same number of columns (sets). This is an
       error, never a silent truncation.
    4. At least two sets exist (one to train, one to test).
    5. Neither file contains NaN or Inf.
    6. Every set number in ``train_sets`` / ``test_sets`` is within ``1..S``,
       the two are disjoint, and both are non-empty.
    7. ``washout`` is within ``0 <= washout < N`` (pulse indices, not samples).
    8. Warning only (:class:`ScaleWarning`): ``ridge_lambda`` is compared
       with the scale of ``X^T X`` over the training data.

    Checks that depend on an earlier one (e.g. the washout range needs a
    non-empty ``u`` to know ``N``) are skipped, not crashed, if their
    prerequisite already failed -- they are reported together with everything
    else found, not one at a time.
    """
    spec = spec_or_cfg if isinstance(spec_or_cfg, DatasetSpec) else DatasetSpec.from_config(spec_or_cfg)

    u_path = spec.input_path
    x_path = spec.response_path
    missing_files = [p for p in (u_path, x_path) if not p.is_file()]
    if missing_files:
        raise ConfigError(
            "Dataset file(s) not found:\n"
            + "\n".join(f"  - {p}" for p in missing_files)
            + f"\n\nLooked inside {spec.path} for '{spec.input_file}' and "
            f"'{spec.response_file}'. Check dataset.path and the file names "
            f"in the configuration."
        )

    u = _load_csv(u_path)
    X = _load_csv(x_path)

    problems: list[str] = []
    N, S_u = u.shape
    kN, S_x = X.shape

    _check_binary(u, u_path, problems)

    n_ok = N > 0
    divisible = False
    if n_ok:
        divisible = _check_divisibility(kN, N, x_path, u_path, problems)
    else:
        problems.append(
            f"{u_path.name} has 0 rows: it must contain at least one pulse "
            f"(one row per pulse index)."
        )

    cols_ok = _check_columns(S_u, S_x, u_path, x_path, problems)
    S = S_u

    _check_min_sets(S, problems)
    _check_finite(u, u_path.name, problems)
    finite_x = _check_finite(X, x_path.name, problems)
    split_ok = _check_split(spec.train_sets, spec.test_sets, S, problems)

    if n_ok:
        _check_washout(spec.washout, N, problems)

    # Check 8 needs the training columns to exist in X (cols_ok) and the
    # split to be valid (split_ok); otherwise X[:, train_cols] would raise.
    if n_ok and cols_ok and finite_x and split_ok:
        train_cols = [n - 1 for n in spec.train_sets]
        _check_scale(spec, X[:, train_cols], N)

    raise_problems(problems, source=spec.path, guide=GUIDE)

    k = (kN // N) if (n_ok and divisible) else 0
    train_index = tuple(sorted(n - 1 for n in spec.train_sets))
    test_index = tuple(sorted(n - 1 for n in spec.test_sets))

    return RCDataset(
        u=u,
        X=X,
        n_inputs=N,
        samples_per_pulse=k,
        n_sets=S,
        spec=spec,
        train_index=train_index,
        test_index=test_index,
    )


# ---------------------------------------------------------------------------
# the eight checks
# ---------------------------------------------------------------------------


def _check_binary(u: np.ndarray, u_path: Path, problems: list[str]) -> bool:
    """Check 1: every value of ``u`` is 0 or 1 (0.0/1.0 float ok)."""
    bad_mask = ~np.isin(u, (0.0, 1.0))
    n_bad = int(np.count_nonzero(bad_mask))
    if n_bad == 0:
        return True
    rows, cols = np.nonzero(bad_mask)
    examples = ", ".join(
        f"(row {r + 1}, col {c + 1}) = {u[r, c]!r}" for r, c in list(zip(rows, cols))[:5]
    )
    more = "" if n_bad <= 5 else f", and {n_bad - 5} more"
    problems.append(
        f"{u_path.name} must contain only binary values (0 or 1): found "
        f"{n_bad} value(s) outside {{0, 1}}. This is the digital gate-drive "
        f"sequence; a non-binary value usually means the wrong instrument "
        f"column was exported. First offending value(s): "
        f"{examples}{more} (row/col are 1-based)."
    )
    return False


def _check_divisibility(kN: int, N: int, x_path: Path, u_path: Path, problems: list[str]) -> bool:
    """Check 2: ``X.shape[0] % N == 0`` -- exactly k samples per pulse."""
    if kN % N == 0:
        return True
    problems.append(
        f"{x_path.name} has {kN} rows, which is not a whole multiple of "
        f"{u_path.name}'s {N} rows (pulses per set). {x_path.name} must "
        f"contain exactly k samples for every pulse, stacked in pulse order "
        f"(the k samples of pulse 1, then the k samples of pulse 2, ...), so "
        f"its row count must equal N times a whole number k. "
        f"{kN} / {N} = {kN / N:.4f}, not a whole number. This usually means "
        f"the two files were not stitched consistently."
    )
    return False


def _check_columns(S_u: int, S_x: int, u_path: Path, x_path: Path, problems: list[str]) -> bool:
    """Check 3: same number of columns (sets) in both files. An error, never
    a silent truncation."""
    if S_u == S_x:
        return True
    if S_u > S_x:
        problems.append(
            f"{u_path.name} has {S_u} columns (sets) but {x_path.name} has "
            f"only {S_x}. Fix the files so they match. If "
            f"{x_path.name} (with {S_x} sets) reflects what was actually "
            f"measured, keep only the first {S_x} columns of {u_path.name} "
            f"and re-run.\n"
            f"     {u_path.name}: {S_u} columns ({u_path})\n"
            f"     {x_path.name}: {S_x} columns ({x_path})"
        )
    else:
        problems.append(
            f"{x_path.name} has {S_x} columns (sets) but {u_path.name} has "
            f"only {S_u}. Every column of {x_path.name} must correspond to a "
            f"column of {u_path.name} (and vice versa) -- check that the two "
            f"files were generated together and cover the same sets.\n"
            f"     {u_path.name}: {S_u} columns ({u_path})\n"
            f"     {x_path.name}: {S_x} columns ({x_path})"
        )
    return False


def _check_min_sets(S: int, problems: list[str]) -> bool:
    """Check 4: at least one training set and one test set can exist."""
    if S >= 2:
        return True
    problems.append(
        f"There {'is' if S == 1 else 'are'} only {S} set(s) (column(s)) in "
        f"the data, but scoring needs at least one training set and one test "
        f"set (S >= 2). Add more sets, or combine this data with sets "
        f"measured elsewhere."
    )
    return False


def _check_finite(arr: np.ndarray, name: str, problems: list[str]) -> bool:
    """Check 5: no NaN / Inf in either file."""
    bad_mask = ~np.isfinite(arr)
    n_bad = int(np.count_nonzero(bad_mask))
    if n_bad == 0:
        return True
    first_r, first_c = np.argwhere(bad_mask)[0]
    problems.append(
        f"{name} contains {n_bad} non-finite value(s) (NaN or Inf). These "
        f"usually come from an instrument overflow, a divide-by-zero in "
        f"post-processing, or a bad export. First at (row {first_r + 1}, "
        f"col {first_c + 1}) = {arr[first_r, first_c]!r}. Locate and fix the "
        f"source measurement."
    )
    return False


def _check_split(
    train_sets: tuple[int, ...], test_sets: tuple[int, ...], S: int, problems: list[str]
) -> bool:
    """Check 6: split sanity -- in range, disjoint, both non-empty."""
    ok = True
    out_of_range = sorted({n for n in (*train_sets, *test_sets) if not (1 <= n <= S)})
    if out_of_range:
        problems.append(
            f"dataset.train_sets / dataset.test_sets reference set number(s) "
            f"{out_of_range} but the data only has {S} set(s) (valid range "
            f"1..{S}; sets are numbered starting at 1). Fix the set numbers, "
            f"or check that dataset.path points at the data you meant."
        )
        ok = False

    overlap = sorted(set(train_sets) & set(test_sets))
    if overlap:
        problems.append(
            f"dataset.train_sets and dataset.test_sets both contain set "
            f"number(s) {overlap}. A set cannot be used for both training and "
            f"testing at once -- remove it from one of the two lists."
        )
        ok = False

    if not train_sets:
        problems.append(
            "dataset.train_sets is empty. At least one set must be reserved for training."
        )
        ok = False
    if not test_sets:
        problems.append(
            "dataset.test_sets is empty. At least one set must be reserved for testing."
        )
        ok = False

    return ok


def _check_washout(washout: int, N: int, problems: list[str]) -> bool:
    """Check 7: washout is a pulse index within range."""
    if 0 <= washout < N:
        return True
    problems.append(
        f"dataset.washout: {washout} is out of range. It must satisfy "
        f"0 <= washout < {N} (the data has {N} pulses per set). washout is "
        f"counted in PULSE INDICES, not samples: it is the number of leading "
        f"pulses discarded as the device's warm-up transient before scoring "
        f"begins."
    )
    return False


def _check_scale(spec: DatasetSpec, X_train: np.ndarray, N: int) -> None:
    """Check 8 (warning): ridge_lambda sane relative to the scale of X^T X.

    ``X_train`` must contain only the training columns -- the test sets play
    no part in fitting the readout, so they must not influence this check.
    """
    if not spec.ridge_lambda > 0.0:
        return  # rejected earlier by _as_positive; never reach the band logic
    xtx_scale = len(spec.train_sets) * N * float(np.mean(X_train**2))
    if xtx_scale <= 0.0:
        return
    if spec.ridge_lambda > 0.1 * xtx_scale:
        warnings.warn(
            f"dataset.ridge_lambda ({spec.ridge_lambda:.3g}) is more than "
            f"10% of the typical diagonal magnitude of X^T X over the "
            f"training data ({xtx_scale:.3g}). The ridge penalty dominates "
            f"the data term, so the fitted readout will be shrunk toward "
            f"zero regardless of what the device actually does. Lower "
            f"ridge_lambda, or confirm this is intentional.",
            ScaleWarning,
            stacklevel=2,
        )
    elif spec.ridge_lambda < 1e-16 * xtx_scale:
        warnings.warn(
            f"dataset.ridge_lambda ({spec.ridge_lambda:.3g}) is below "
            f"float64 resolution relative to X^T X over the training data "
            f"({xtx_scale:.3g}): the ratio is under 1e-16. The penalty is "
            f"numerically inert -- you are effectively running unregularised "
            f"least squares. This may be intentional (a tiny ridge_lambda "
            f"can serve as a conditioning term rather than real "
            f"regularisation), but confirm it.",
            ScaleWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# suggestion block
# ---------------------------------------------------------------------------


def suggest_dataset_block(
    path: str | Path,
    *,
    input_file: str = DEFAULT_INPUT_FILE,
    response_file: str = DEFAULT_RESPONSE_FILE,
) -> str:
    """Infer N, k, S from the CSV files in *path* and format a ``dataset:``
    YAML block to paste into a configuration file.

    Only what follows from the *shape* of the data is filled in (N, k, S, and
    a structural suggestion for washout -- half the sequence). Everything
    that determines the score -- train_sets, test_sets, ridge_lambda, protocol,
    readout_full_scale -- is left marked ``REQUIRED``.
    ``readout_bits`` is suggested as ``auto``, which itself follows from the
    sample interval, not from a guess.
    """
    path = Path(path)
    u_path = path / input_file
    x_path = path / response_file
    missing_files = [p for p in (u_path, x_path) if not p.is_file()]
    if missing_files:
        raise ConfigError(
            "Cannot suggest a dataset block: file(s) not found:\n"
            + "\n".join(f"  - {p}" for p in missing_files)
        )

    u = _load_csv(u_path)
    X = _load_csv(x_path)

    N, S = u.shape
    kN, S_x = X.shape
    if S != S_x:
        raise ConfigError(
            f"Cannot suggest a dataset block: {input_file} has {S} column(s) "
            f"(sets) but {response_file} has {S_x}. Fix the column mismatch "
            f"first -- see {GUIDE}."
        )
    if N == 0 or kN % N != 0:
        raise ConfigError(
            f"Cannot suggest a dataset block: {response_file} has {kN} rows, "
            f"which is not a whole multiple of {input_file}'s {N} rows "
            f"(pulses per set). Fix the two files first -- see {GUIDE}."
        )
    k = kN // N

    lines = [
        "dataset:",
        f"  path: {path}",
        f"  # Inferred from the data: N = {N} pulses per set, k = {k} samples per pulse, S = {S} sets",
        f"  washout: {N // 2}            # structural suggestion: half the sequence -- adjust to your device",
        "  train_sets: <REQUIRED -- set numbers (1..S) used for training; see the example splits below>",
        "  test_sets:  <REQUIRED -- set numbers (1..S) held out for testing>",
        "  ridge_lambda: <REQUIRED>   # readout ridge regularisation strength",
        "  protocol: <REQUIRED -- kfold (shuffled, one fold per set) or fixed (one split as written)>",
        "  # shuffle_seed: 0   # required with protocol: kfold (a whole number); leave it out with fixed",
        "  # Readout converter. auto picks the bit count from sample_interval (time between",
        "  # the samples of one pulse): >= 1 ms -> 16 bit, 1 us..1 ms -> 12, 100 ns..1 us -> 10,",
        "  # < 100 ns -> 8. Use ideal for no quantization, or write a whole number of bits.",
        "  readout_bits: auto",
        "  readout_full_scale: <REQUIRED>   # converter range [A]: it spans 0..full_scale",
        "  # Set readout_full_scale a little above the largest drain current you expect (about",
        "  # 1.2 x the expected maximum); a range much wider than the signal wastes resolution",
        "  # (each factor of 2 costs one bit). Base it on the expected operating range rather than on the scored data.",
        "  sample_interval: <REQUIRED for measured data>   # [s]; derived from pulse.t_pw / pulse.samples_per_step when the file has a pulse: section",
        "  # Simulated data: physics.device_noise sets the device noise as a fraction of the",
        "  # read current (0 = no device noise; e.g. 0.002 = 0.2 %).",
        "",
        "Example splits -- a starting point, not a standard. By number of sets:",
        "",
        "  sets   train      test",
    ]
    matched = False
    for sets, train, test in SPLIT_PRECEDENTS:
        lines.append(f"  {sets:<6} {train:<10} {test}")
        if sets == S:
            matched = True
    lines.append("")
    if matched:
        lines.append(
            f"This dataset has {S} sets, matching an example split above -- "
            f"shown for reference only; the fields above are still REQUIRED."
        )
    else:
        lines.append(f"No example split is listed for {S} sets. Choose your own split.")
    return "\n".join(lines)
