"""Read a YAML configuration, check it, and fill in derived values.

This module does three things.

1. **Number coercion.** PyYAML implements YAML 1.1, which reads ``1e-6`` as a
   *string* and only ``1.0e-6`` as a number. Configuration values here are
   mostly in scientific notation, so every string that looks like a number is
   converted to ``float`` on the way in.
2. **Consistency checks.** Combinations that cannot work together (a device
   kind with an impossible phase ratio, a dataset section missing a
   score-determining value)
   are caught *when the file is read*, with a message that says what to do.
3. **Derived values.** Anything that follows from other values
   (``dt = t_pw / samples_per_step`` and friends) is computed here and must
   not be written in the YAML.

Usage::

    from fefet_reservoir.config import load_config

    cfg = load_config("configs/example_simulated.yaml")
    cfg["pulse"]["t_pw"]        # a value from the file
    cfg.derived["dt"]           # a value computed from the file
    cfg.section("dataset")      # a whole section, or {} if absent

Device parameters (``material:`` / ``mosfet:``) are needed only by
``simulate-rc`` (see :func:`require_device_parameters`); a scoring-only file may
omit them and the ``device:`` line.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError, raise_problems
from .schema import (
    DEFAULT_MAX_DELAY,
    DERIVED_KEYS,
    DEVICES,
    KNOWN_DATASET_KEYS,
    KNOWN_DOMAINS_KEYS,
    KNOWN_MATERIAL_KEYS,
    KNOWN_MOSFET_KEYS,
    KNOWN_PHYSICS_KEYS,
    KNOWN_PULSE_KEYS,
    KNOWN_SIMULATION_KEYS,
    PROTOCOLS,
    POSITIVE_DEVICE_PARAMETER_KEYS,
    PULSE_SHAPES,
    READOUT_BITS_MAX,
    READOUT_BITS_MIN,
    READOUT_BITS_MODES,
    REQUIRED_DATASET_KEYS,
    REQUIRED_DEVICE_PARAMETER_KEYS,
    REQUIRED_SIMULATION_KEYS,
    REQUIRED_SIMULATION_PHYSICS_KEYS,
    VIN_SAMPLING,
)
from .sets import SetFieldError, parse_set_field

__all__ = [
    "Config",
    "load_config",
    "coerce_numbers",
    "compute_derived",
    "validate",
    "get_by_path",
    "require_section",
    "require_device_parameters",
    "parse_readout_fields",
]

GUIDE = "README.md, 'Configuration files'"

#: A string that looks like a number. Underscores are deliberately excluded:
#: Python's ``float("3_3")`` returns 33.0, which is never what a label meant.
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


# ---------------------------------------------------------------------------
# 1) number coercion
# ---------------------------------------------------------------------------


#: Path-valued keys whose value is never coerced to a number (e.g. a folder
#: named ``"2024"``).
_PATH_LIKE_KEYS = frozenset(
    {"path", "input_file", "response_file", "output_dir"}
)


def coerce_numbers(obj: Any, *, skip_keys: frozenset[str] = _PATH_LIKE_KEYS) -> Any:
    """Convert every number-looking string in a nested dict/list to ``float``.

    Leaves genuine strings (``"fefet"``, ``"1us"``, ``"3_3"``) untouched, and
    never converts the value of a key in *skip_keys* (path-valued keys by
    default -- see ``_PATH_LIKE_KEYS``).
    """
    if isinstance(obj, dict):
        return {
            k: (v if k in skip_keys else coerce_numbers(v, skip_keys=skip_keys))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [coerce_numbers(v, skip_keys=skip_keys) for v in obj]
    if isinstance(obj, str) and _NUMERIC_RE.match(obj.strip()):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# 2) derived values
# ---------------------------------------------------------------------------


def _dt_from_pulse(pulse: dict) -> float | None:
    """``dt = t_pw / samples_per_step``, or ``None`` if either is absent."""
    t_pw = pulse.get("t_pw")
    vn = pulse.get("samples_per_step")
    if t_pw is None or vn is None:
        return None
    return float(t_pw) / float(vn)


def compute_derived(data: dict) -> dict:
    """Values that must not be in the YAML because they follow from others.

    * ``n_tetra`` / ``n_ortho`` : domain count split by phase (``count x t_ratio``)
    * ``dt``         : ``t_pw / samples_per_step``
    * ``t_slope``    : ``t_pw / 1000`` (rise time of a square pulse)
    * ``signal_ref`` : ``(signal_on + signal_off) / 2`` (triangular reference)
    """
    derived: dict[str, Any] = {}

    t_ratio = data.get("t_ratio")
    domains = data.get("domains") or {}
    count = domains.get("count")
    if t_ratio is not None and count is not None:
        n_tetra = int(round(float(count) * float(t_ratio)))
        derived["n_tetra"] = n_tetra
        derived["n_ortho"] = int(count) - n_tetra

    pulse = data.get("pulse") or {}
    t_pw = pulse.get("t_pw")
    if t_pw is not None:
        dt = _dt_from_pulse(pulse)
        if dt is not None:
            derived["dt"] = dt
        derived["t_slope"] = float(t_pw) / 1000.0
    if "signal_on" in pulse and "signal_off" in pulse:
        derived["signal_ref"] = (float(pulse["signal_on"]) + float(pulse["signal_off"])) / 2.0

    return derived


# ---------------------------------------------------------------------------
# 3) consistency checks
# ---------------------------------------------------------------------------


def _find_derived_keys(obj: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            where = f"{path}.{key}" if path else str(key)
            if key in DERIVED_KEYS:
                found.append(where)
            found.extend(_find_derived_keys(value, where))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found.extend(_find_derived_keys(value, f"{path}[{i}]"))
    return found


def _check_no_derived_fields(data: dict, problems: list[str]) -> None:
    pulse = data.get("pulse") or {}
    allow_dt = "t_pw" not in pulse  # without a pulse, dt is a primary input
    for where in _find_derived_keys(data):
        name = where.rsplit(".", 1)[-1]
        if name == "dt" and allow_dt:
            continue
        problems.append(
            f"'{where}' must not be written in the configuration file. It is "
            f"computed from other values (dt = t_pw / samples_per_step, "
            f"t_slope = t_pw / 1000, signal_ref = (signal_on + signal_off) / 2, "
            f"n_tetra / n_ortho = domain count x t_ratio); remove this line."
        )


def _check_device(data: dict, problems: list[str]) -> None:
    device = data.get("device")
    # A non-numeric t_ratio is reported even when device is not checked below.
    t_ratio = data.get("t_ratio")
    t_ratio_ok = t_ratio is None or _is_number(t_ratio)
    if not t_ratio_ok:
        problems.append(f"t_ratio: {t_ratio!r} is not a number.")

    # Only the simulator needs the device kind: a scoring-only file (no
    # `simulation:` section) may leave the `device:` line out entirely.
    if device is None and data.get("simulation") is None:
        return
    if device is None:
        problems.append(
            "device: is missing -- use fefet (ferroelectric) or afefet "
            "(antiferroelectric). A file with a simulation: section needs a "
            "device: line."
        )
        return
    if device not in DEVICES:
        problems.append(
            f"device: {device!r} is not a known device. "
            f"Use 'fefet' (ferroelectric) or 'afefet' (antiferroelectric)."
        )
        return

    if t_ratio is None or not t_ratio_ok:
        return
    t_ratio = float(t_ratio)
    if device == "fefet" and t_ratio != 0.0:
        problems.append(
            f"device: fefet but t_ratio: {t_ratio}. A ferroelectric device has "
            f"no tetragonal (antiferroelectric) phase, so t_ratio must be 0.0. "
            f"To model a mixed device, set device: afefet and adjust t_ratio."
        )
    elif device == "afefet" and not 0.0 < t_ratio <= 1.0:
        problems.append(
            f"device: afefet but t_ratio: {t_ratio}. It must be greater than 0 "
            f"and at most 1.0."
        )


def _check_domains(data: dict, problems: list[str]) -> None:
    """Check ``domains.count`` (a whole number >= 1)."""
    domains = _device_section(data, "domains", KNOWN_DOMAINS_KEYS, problems)
    count = domains.get("count")
    if count is not None and not _is_number(count):
        problems.append(f"domains.count: {count!r} is not a number.")
    elif count is not None and (
        not _is_finite_number(count) or float(count) != int(count) or int(count) < 1
    ):
        problems.append(
            f"domains.count: {count!r} must be a whole number >= 1 (the "
            f"number of switching domains in the ferroelectric layer)."
        )


def _device_section(data: dict, name: str, known: tuple[str, ...], problems: list[str]) -> dict:
    """Return section *name* (``{}`` if absent), reporting a non-mapping
    value and any key not in *known*. Names that must not be written at all
    (``DERIVED_KEYS``, e.g. ``pulse.dt``) are left to
    :func:`_check_no_derived_fields`, which explains why."""
    section = data.get(name) or {}
    if not isinstance(section, dict):
        problems.append(f"{name}: must be a mapping of 'name: value' lines.")
        return {}
    unknown = sorted(set(section) - set(known) - set(DERIVED_KEYS))
    if unknown:
        problems.append(
            f"{name}: unknown key(s) {', '.join(repr(k) for k in unknown)}. "
            f"Known keys: {', '.join(repr(k) for k in known)}. "
            f"This is usually a typo -- if the key is meant for another "
            f"section, move it there; otherwise remove it."
        )
    return section


def _check_material_mosfet(data: dict, problems: list[str]) -> None:
    """``material:`` / ``mosfet:``: unknown keys, values that are not
    numbers, and values that must be > 0 (``POSITIVE_DEVICE_PARAMETER_KEYS``)."""
    for name, known in (("material", KNOWN_MATERIAL_KEYS), ("mosfet", KNOWN_MOSFET_KEYS)):
        section = _device_section(data, name, known, problems)
        for key in known:
            value = section.get(key)
            if value is None:
                continue
            if not _is_finite_number(value):
                problems.append(f"{name}.{key}: {value!r} is not a number.")
            elif f"{name}.{key}" in POSITIVE_DEVICE_PARAMETER_KEYS and float(value) <= 0.0:
                problems.append(
                    f"{name}.{key}: {value!r} must be a number > 0 (a "
                    f"physical size or material constant cannot be zero or "
                    f"negative)."
                )


def _check_physics(data: dict, problems: list[str]) -> None:
    physics = data.get("physics") or {}
    if not isinstance(physics, dict):
        problems.append("physics: must be a mapping of 'name: value' lines.")
        return
    unknown = sorted(set(physics) - set(KNOWN_PHYSICS_KEYS))
    if unknown:
        problems.append(
            f"physics: unknown key(s) {', '.join(repr(k) for k in unknown)}. "
            f"Known keys: {', '.join(repr(k) for k in KNOWN_PHYSICS_KEYS)}. "
            f"This is usually a typo -- if the key is meant for another "
            f"section, move it there; otherwise remove it."
        )

    # Same rule as REQUIRED_SIMULATION_KEYS: required as soon as the file
    # has a `simulation:` section, i.e. whenever it can drive simulate-rc.
    if data.get("simulation") is not None:
        missing = [k for k in REQUIRED_SIMULATION_PHYSICS_KEYS if physics.get(k) is None]
        if missing:
            problems.append(
                f"physics: missing (or empty) {', '.join(repr(k) for k in missing)}. "
                f"physics.device_noise is the device-noise standard deviation "
                f"as a fraction of the read current (0 = no device noise; "
                f"e.g. 0.002 = 0.2 %). It changes the simulated response, so it "
                f"must be written in the configuration file."
            )

    device_noise = physics.get("device_noise")
    if device_noise is not None and (
        not _is_finite_number(device_noise) or float(device_noise) < 0.0
    ):
        problems.append(
            f"physics.device_noise: {device_noise!r} must be a number >= 0 "
            f"(a fraction of the read current; 0 = no device noise; "
            f"e.g. 0.002 = 0.2 %)."
        )


def _check_pulse(data: dict, problems: list[str]) -> None:
    pulse = _device_section(data, "pulse", KNOWN_PULSE_KEYS, problems)
    if not pulse:
        return

    sampling = pulse.get("vin_sampling")
    if sampling is not None and sampling not in VIN_SAMPLING:
        problems.append(
            f"pulse.vin_sampling: {sampling!r} is not a known option. Use "
            f"'endpoint' (default) or 'half_sample_offset'. The second shifts "
            f"the whole waveform by half a sample."
        )

    shape = pulse.get("shape")
    if shape is not None and shape not in PULSE_SHAPES:
        problems.append(
            f"pulse.shape: {shape!r} is not a known waveform. "
            f"Use 'square' or 'triangular'."
        )

    # Report non-numeric values here; compute_derived converts them with float().
    for key in ("t_pw", "samples_per_step", "signal_on", "signal_off"):
        value = pulse.get(key)
        if value is not None and not _is_number(value):
            problems.append(f"pulse.{key}: {value!r} is not a number.")

    t_pw = pulse.get("t_pw")
    if _is_number(t_pw) and not (_is_finite_number(t_pw) and float(t_pw) > 0.0):
        problems.append(f"pulse.t_pw: {t_pw!r} must be a number > 0 (pulse width in seconds).")

    per_step = pulse.get("samples_per_step")
    if _is_number(per_step) and not (
        _is_finite_number(per_step) and float(per_step) == int(per_step) and int(per_step) >= 1
    ):
        problems.append(
            f"pulse.samples_per_step: {per_step!r} must be a whole number >= 1 "
            f"(response samples recorded per pulse)."
        )


def _check_dataset(data: dict, problems: list[str]) -> None:
    """The RC ``dataset:`` section. Structural checks only; the values are
    checked against the actual data by :mod:`fefet_reservoir.io.dataset`."""
    dataset = data.get("dataset")
    if dataset is None:
        return
    if not isinstance(dataset, dict):
        problems.append("dataset: must be a mapping of 'name: value' lines.")
        return

    unknown = sorted(set(dataset) - set(KNOWN_DATASET_KEYS))
    if unknown:
        problems.append(
            f"dataset: unknown key(s) {', '.join(repr(k) for k in unknown)}. "
            f"Known keys: {', '.join(repr(k) for k in KNOWN_DATASET_KEYS)}. "
            f"This is usually a typo -- if the key is meant for another "
            f"section, move it there; otherwise remove it."
        )

    if dataset.get("path") is None:
        problems.append(
            "dataset.path is missing. It is the directory that holds the "
            "input file and the response file (it may be outside this "
            "repository -- your data stays where it is)."
        )

    missing = [k for k in REQUIRED_DATASET_KEYS if dataset.get(k) is None]
    if missing:
        problems.append(
            f"dataset: missing (or empty) {', '.join(repr(k) for k in missing)}. These "
            f"values change the score, so they must be written in the "
            f"configuration file. Run "
            f"`python -m fefet_reservoir suggest <data dir>` to print a "
            f"suggestion block for your data, then write the values you "
            f"decide on into the file."
        )

    ridge_lambda = dataset.get("ridge_lambda")
    if ridge_lambda is not None and (
        not _is_number(ridge_lambda) or float(ridge_lambda) <= 0.0
    ):
        problems.append(
            f"dataset.ridge_lambda: {ridge_lambda!r} must be a positive number."
        )

    washout = dataset.get("washout")
    if washout is not None and (not _is_number(washout) or float(washout) < 0
                                or float(washout) != int(washout)):
        problems.append(
            f"dataset.washout: {washout!r} must be a non-negative whole number "
            f"of pulses (not samples): the leading pulses of every set left out "
            f"of scoring."
        )

    # Validate the SHAPE of train_sets / test_sets here too (not just
    # after loading the actual data in io.dataset), using the shared token
    # grammar, so a typo like "banana" is caught when the file is read.
    for key in ("train_sets", "test_sets"):
        value = dataset.get(key)
        if value is None:
            continue
        try:
            parse_set_field(value, key)
        except SetFieldError as exc:
            problems.append(str(exc))

    # A missing protocol is reported by the required-key check above.
    protocol = dataset.get("protocol")
    protocol_valid = protocol in PROTOCOLS
    if protocol is not None and not protocol_valid:
        problems.append(
            f"dataset.protocol: {protocol!r} is not a known protocol. Use "
            f"'kfold' (shuffled cross-validation, one fold per set) or 'fixed' "
            f"(a single train/test split). "
            f"See README.md, 'Scoring'."
        )

    shuffle_seed = dataset.get("shuffle_seed")
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
    if shuffle_seed is not None and (
        not _is_number(shuffle_seed) or float(shuffle_seed) != int(shuffle_seed)
    ):
        problems.append(
            f"dataset.shuffle_seed: {shuffle_seed!r} must be a whole number."
        )

    max_delay = dataset.get("max_delay", DEFAULT_MAX_DELAY)
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

    parse_readout_fields(dataset, data.get("pulse") or {}, problems)


def parse_readout_fields(
    dataset: dict, pulse: dict, problems: list[str]
) -> tuple[str | int | None, float | None, float | None, bool]:
    """Check the readout keys of a ``dataset:`` section, appending to *problems*.

    Shared by :func:`_check_dataset` (the YAML layer) and
    :meth:`fefet_reservoir.io.dataset.DatasetSpec.from_config` (the dict entry
    point), so both enforce exactly the same rules:

    * ``readout_bits``: ``'auto'``, ``'ideal'`` or a whole number from
      ``READOUT_BITS_MIN`` to ``READOUT_BITS_MAX`` (a bool is rejected). Its
      absence is reported by the caller's required-key check, not here.
    * ``readout_full_scale`` [A]: a number > 0; required unless
      ``readout_bits`` is ``'ideal'``.
    * ``sample_interval`` [s]: a number > 0. With a ``pulse:`` section it is
      derived as ``pulse.t_pw / pulse.samples_per_step``, and a written value
      that disagrees with the derived one is a problem. Without one, it is
      required when ``readout_bits`` is ``'auto'``.

    Returns ``(readout_bits, readout_full_scale, sample_interval,
    sample_interval_derived)``; a value that failed its check is ``None``.
    """
    bits = dataset.get("readout_bits")
    if bits is not None:
        if isinstance(bits, str):
            if bits not in READOUT_BITS_MODES:
                problems.append(
                    f"dataset.readout_bits: {bits!r} is not a known setting. "
                    f"Use 'auto' (bit count chosen from the sample interval), "
                    f"'ideal' (no quantization) or a whole number of bits "
                    f"from {READOUT_BITS_MIN} to {READOUT_BITS_MAX}."
                )
                bits = None
        elif (
            not _is_finite_number(bits)
            or float(bits) != int(bits)
            or not READOUT_BITS_MIN <= int(bits) <= READOUT_BITS_MAX
        ):
            problems.append(
                f"dataset.readout_bits: {bits!r} must be 'auto', 'ideal' or a "
                f"whole number of bits from {READOUT_BITS_MIN} to "
                f"{READOUT_BITS_MAX}."
            )
            bits = None
        else:
            bits = int(bits)

    # readout_full_scale with 'ideal', and sample_interval with an explicit
    # bit count, are accepted (unused) so a file can toggle between settings.
    full_scale = dataset.get("readout_full_scale")
    if full_scale is not None:
        if not _is_finite_number(full_scale) or float(full_scale) <= 0.0:
            problems.append(
                f"dataset.readout_full_scale: {full_scale!r} must be a number "
                f"> 0 (amperes; the converter spans 0..full_scale)."
            )
            full_scale = None
        else:
            full_scale = float(full_scale)
    elif dataset.get("readout_bits") is not None and bits != "ideal":
        # An invalid-but-present readout_bits (bits is None here) is not
        # 'ideal' either, so a missing full scale is reported in the same pass.
        problems.append(
            f"dataset.readout_full_scale is missing. With readout_bits: "
            f"{dataset.get('readout_bits')!r} the response is quantized by a converter spanning "
            f"0..full_scale [A], so its range must be written "
            f"in the configuration file. Use readout_bits: ideal to "
            f"score without quantization."
        )

    given = dataset.get("sample_interval")
    given_ok = given is not None
    if given_ok:
        if not _is_finite_number(given) or float(given) <= 0.0:
            problems.append(
                f"dataset.sample_interval: {given!r} must be a number > 0 "
                f"(seconds between the samples recorded within one pulse)."
            )
            given_ok = False
        else:
            given = float(given)

    derived = None
    t_pw = pulse.get("t_pw")
    per_step = pulse.get("samples_per_step")
    if (
        _is_finite_number(t_pw)
        and _is_finite_number(per_step)
        and float(t_pw) > 0.0
        and float(per_step) > 0.0
    ):
        derived = _dt_from_pulse(pulse)

    if given_ok and derived is not None and not math.isclose(given, derived, rel_tol=1e-9):
        problems.append(
            f"dataset.sample_interval: {given!r} disagrees with the value "
            f"derived from the pulse: section (pulse.t_pw / "
            f"pulse.samples_per_step = {derived!r}). Remove "
            f"dataset.sample_interval, or make the two agree."
        )
        given_ok = False
    if bits == "auto" and dataset.get("sample_interval") is None and derived is None:
        if pulse:
            source = (
                "the pulse: section does not give it (pulse.t_pw and "
                "pulse.samples_per_step must both be numbers > 0)"
            )
        else:
            source = (
                "this file has no pulse: section to derive it from "
                "(pulse.t_pw / pulse.samples_per_step)"
            )
        problems.append(
            f"dataset.sample_interval is missing. readout_bits: auto chooses "
            f"the bit count from the time between the samples of one pulse, "
            f"and {source}. Write it in seconds, or set readout_bits to a "
            f"whole number of bits."
        )

    if given_ok:
        return bits, full_scale, given, False
    if dataset.get("sample_interval") is not None:
        return bits, full_scale, None, False
    return bits, full_scale, derived, derived is not None


def _check_simulation(data: dict, problems: list[str]) -> None:
    """The ``simulation:`` section (run size for ``simulate-rc``): required
    keys, unknown keys, value ranges. Structural checks only."""
    simulation = data.get("simulation")
    if simulation is None:
        return
    if not isinstance(simulation, dict):
        problems.append("simulation: must be a mapping of 'name: value' lines.")
        return

    unknown = sorted(set(simulation) - set(KNOWN_SIMULATION_KEYS))
    if unknown:
        problems.append(
            f"simulation: unknown key(s) {', '.join(repr(k) for k in unknown)}. "
            f"Known keys: {', '.join(repr(k) for k in KNOWN_SIMULATION_KEYS)}. "
            f"This is usually a typo -- if the key is meant for another "
            f"section, move it there; otherwise remove it."
        )

    # A key with no value counts as missing.
    missing = [k for k in REQUIRED_SIMULATION_KEYS if simulation.get(k) is None]
    if missing:
        problems.append(
            f"simulation: missing (or empty) {', '.join(repr(k) for k in missing)}. "
            f"These define the run itself and must be written in the "
            f"configuration file: "
            f"simulation.n_inputs and simulation.n_sets set the size of the "
            f"run, simulation.seed makes it reproducible, and "
            f"simulation.output_dir says where the result is written."
        )

    n_inputs = simulation.get("n_inputs")
    if n_inputs is not None and (
        not _is_number(n_inputs) or float(n_inputs) != int(n_inputs) or int(n_inputs) < 1
    ):
        problems.append(
            f"simulation.n_inputs: {n_inputs!r} must be a whole number >= 1."
        )

    n_sets = simulation.get("n_sets")
    if n_sets is not None and (
        not _is_number(n_sets) or float(n_sets) != int(n_sets) or int(n_sets) < 2
    ):
        problems.append(
            f"simulation.n_sets: {n_sets!r} must be a whole number >= 2 -- at "
            f"least two sets are needed to split into train and test."
        )

    seed = simulation.get("seed")
    if seed is not None and (
        not _is_number(seed) or float(seed) != int(seed) or int(seed) < 0
    ):
        problems.append(
            f"simulation.seed: {seed!r} must be a non-negative whole number."
        )

    output_dir = simulation.get("output_dir")
    if output_dir is not None and not isinstance(output_dir, str):
        problems.append(
            f"simulation.output_dir: {output_dir!r} must be a string (a "
            f"directory path)."
        )


def validate(data: dict, *, source: str | Path | None = None) -> None:
    """Run every consistency check; raise :class:`ConfigError` listing all failures."""
    problems: list[str] = []
    _check_no_derived_fields(data, problems)
    _check_device(data, problems)
    _check_domains(data, problems)
    _check_material_mosfet(data, problems)
    _check_physics(data, problems)
    _check_pulse(data, problems)
    _check_dataset(data, problems)
    _check_simulation(data, problems)
    raise_problems(problems, source=source, guide=GUIDE)


# ---------------------------------------------------------------------------
# configuration object + reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """One loaded configuration.

    Use it like a dict for values from the file (``cfg["pulse"]["t_pw"]``) and
    ``cfg.derived`` for values computed from it.
    """

    path: Path
    data: dict = field(repr=False)
    derived: dict = field(repr=False)

    @property
    def device(self) -> str | None:
        """The ``device:`` value, or ``None`` in a scoring-only file without one."""
        return self.data.get("device")

    @property
    def experiment(self) -> str:
        return self.data.get("experiment", self.path.stem)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def section(self, name: str) -> dict:
        """``cfg.section("pulse")`` -- the section, or ``{}`` if absent."""
        return self.data.get(name) or {}


def get_by_path(data: dict, dotted: str, default: Any = None) -> Any:
    """``get_by_path(cfg.data, "pulse.t_pw")`` -- one value by dotted path."""
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"{path} is not valid YAML: {exc}\n\nYAML does not allow tab "
                f"characters for indentation -- use spaces."
            ) from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"The top level of a configuration file must be 'name: value' lines: {path}"
        )
    return coerce_numbers(raw)


def load_config(path: str | Path) -> Config:
    """Read a configuration file, check it, and fill in derived values."""
    path = Path(path)
    data = _read_yaml(path)
    validate(data, source=path)
    return Config(path=path, data=data, derived=compute_derived(data))


def require_section(cfg: Config, name: str, required_keys: Sequence[str]) -> dict:
    """Return ``cfg.section(name)``, checked to be present and complete.

    ``load_config`` checks a section only when it is present. A command that
    needs a section calls this to confirm it is present and complete before
    doing any work.

    Parameters
    ----------
    cfg :
        A loaded :class:`Config`.
    name :
        The section name, e.g. ``"simulation"`` or ``"dataset"``.
    required_keys :
        Keys that must be present (and non-empty -- ``key: `` with no value
        counts as missing, same rule as every other required-key check in
        this module) for the section to be usable.

    Raises
    ------
    ConfigError
        If the section is missing entirely, or missing (or empty) any key
        in *required_keys*.
    """
    section = cfg.section(name)
    if not section:
        raise ConfigError(
            f"section '{name}' is missing entirely; this command needs it. "
            f"Add a '{name}:' section to {cfg.path} with "
            f"{', '.join(repr(k) for k in required_keys)}."
        )
    missing = [k for k in required_keys if section.get(k) is None]
    if missing:
        raise ConfigError(
            f"section '{name}' is missing (or empty) "
            f"{', '.join(repr(k) for k in missing)} in {cfg.path}."
        )
    return section


def require_device_parameters(cfg: Config) -> None:
    """Check that every value the simulator reads is present, before it runs.

    ``load_config`` accepts a file without device parameters (scoring never
    needs them); ``simulate-rc`` calls this before it runs. Raises
    :class:`ConfigError` listing every missing key of
    ``schema.REQUIRED_DEVICE_PARAMETER_KEYS``.
    """
    missing = [
        key for key in REQUIRED_DEVICE_PARAMETER_KEYS if get_by_path(cfg.data, key) is None
    ]
    if not missing:
        return
    raise ConfigError(
        f"simulate-rc needs device parameters that were not found in "
        f"{cfg.path}:\n"
        + "\n".join(f"  - {key}" for key in missing)
        + "\n\nWrite them in the configuration file. "
        "configs/example_simulated.yaml lists every one with example values."
    )
