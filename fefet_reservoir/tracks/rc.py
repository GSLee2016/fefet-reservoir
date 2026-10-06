"""Public RC-track scoring API: :func:`score_rc`.

Ties the kernels in :mod:`fefet_reservoir.tracks._kernel` to an
:class:`~fefet_reservoir.io.dataset.RCDataset`: picks the loop (fixed or
kfold) from ``dataset.spec`` and packages the results.

Readout
-------
The response goes through :func:`fefet_reservoir.tracks.readout.quantize`
(per ``spec.readout_bits``) before the kernels; the bit count used is echoed
in ``settings``. :class:`~fefet_reservoir.io.dataset.ReadoutWarning` is
issued here.

Protocol
--------
* ``"fixed"`` -> :func:`_kernel._fixed_split` (one split, no shuffle).
* ``"kfold"`` -> :func:`_kernel._run_delay_benchmark` (shuffle the columns
  once, then one fold per cyclic shift; ``fold_num = n_sets``).

Seed discipline (kfold only)
-----------------------------
A single ``spec.shuffle_seed`` covers the whole dataset. :func:`score_rc`
creates a **fresh** ``np.random.default_rng(spec.shuffle_seed)`` for each
task it scores (STM, then PC). Because ``rng.shuffle(order)`` is the *only*
random draw either loop makes, and ``order`` has the same length
(``n_sets``) for both tasks, both tasks see the same fold order for a given
``shuffle_seed`` -- the same dataset column is "held out" in the same fold
across STM and PC.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..config.errors import raise_problems
from ..config.schema import DEFAULT_MAX_DELAY
from ..io.dataset import ReadoutWarning
from . import _kernel, readout

__all__ = ["score_rc", "RCScores"]

_TASKS = ("stm", "pc")
_PROTOCOLS = ("fixed", "kfold")

GUIDE = "README.md, 'Scoring'"


@dataclass(frozen=True)
class RCScores:
    """The result of :func:`score_rc`.

    ``stm_curve`` / ``pc_curve`` have length ``max_delay + 1`` (delay 0
    included); ``stm_capacity`` / ``pc_capacity`` equal
    ``sum(curve[1:])`` -- delay 0 (the device trivially reading out its own
    current input) is never counted toward capacity.

    ``per_fold`` holds one ``mc_per_fold`` array per scored task when
    ``protocol == "kfold"``, each of
    length ``n_sets``; it is ``None`` for ``protocol == "fixed"``.

    ``settings`` echoes every value that determined the score;
    :func:`~fefet_reservoir.tracks.report.format_rc_summary` prints it with
    the capacities.
    """

    stm_capacity: float
    stm_curve: np.ndarray
    pc_capacity: float
    pc_curve: np.ndarray
    protocol: str
    per_fold: dict[str, np.ndarray] | None
    settings: dict[str, object]


def score_rc(dataset, *, tasks: tuple[str, ...] = _TASKS) -> RCScores:
    """Score an :class:`~fefet_reservoir.io.dataset.RCDataset` on the RC track.

    Runs whichever of ``"stm"``, ``"pc"`` are named in
    ``tasks`` (default: both), using the protocol, split and
    hyperparameters recorded on ``dataset.spec``. See this module's
    docstring for the protocol and seed discipline.

    Raises
    ------
    ConfigError
        Fewer than 2 pulses remain per set after ``washout``; the training
        pulses (``usable_rows * len(train_sets)``) do not outnumber
        ``samples_per_pulse``; ``max_delay`` is not less than the usable
        pulses per set; or a quantizing readout has no
        ``readout_full_scale`` > 0.
    ValueError
        Re-checks for hand-built dataset objects: ``train_index`` /
        ``test_index`` do not match ``spec.train_sets`` / ``spec.test_sets``;
        unknown ``protocol``; ``washout`` outside ``0 <= washout < n_inputs``;
        or ``readout_bits`` cannot be resolved (invalid, or ``auto`` without a
        sample interval).
    """
    spec = dataset.spec

    expected_train_index = tuple(sorted(s - 1 for s in spec.train_sets))
    expected_test_index = tuple(sorted(s - 1 for s in spec.test_sets))
    if dataset.train_index != expected_train_index or dataset.test_index != expected_test_index:
        raise ValueError(
            f"dataset.train_index/test_index are inconsistent with "
            f"dataset.spec.train_sets/test_sets: train_sets={spec.train_sets} "
            f"/ test_sets={spec.test_sets} imply train_index="
            f"{expected_train_index} / test_index={expected_test_index}, but "
            f"the dataset carries train_index={dataset.train_index} / "
            f"test_index={dataset.test_index}. This dataset object was not "
            f"built consistently -- rebuild it from spec.train_sets/test_sets."
        )

    if spec.protocol not in _PROTOCOLS:
        raise ValueError(
            f"dataset.spec.protocol: {spec.protocol!r} is not one of "
            f"{_PROTOCOLS!r}."
        )

    if not 0 <= spec.washout < dataset.n_inputs:
        raise ValueError(
            f"dataset.washout ({spec.washout}) must satisfy "
            f"0 <= washout < n_inputs ({dataset.n_inputs}); a negative "
            f"washout would silently score pulses from the end of the "
            f"sequence instead of discarding a warm-up transient, and "
            f"washout >= n_inputs would discard every pulse."
        )

    # The three checks below are about the data and the dataset: settings,
    # so they are reported as ConfigError (exit code 2 on the command line,
    # same shape as every other configuration problem).
    usable_rows = dataset.n_inputs - spec.washout
    if not usable_rows >= 2:
        raise_problems(
            [
                f"Only {usable_rows} pulse(s) per set are left for scoring: "
                f"each set has {dataset.n_inputs} pulses and dataset.washout "
                f"({spec.washout}) leaves out the first {spec.washout}. At "
                f"least 2 are needed to compute a score. Lower "
                f"dataset.washout, or record more pulses per set."
            ],
            guide=GUIDE,
        )

    train_rows = usable_rows * len(spec.train_sets)
    if not train_rows > dataset.samples_per_pulse:
        raise_problems(
            [
                f"Too few training pulses to fit the readout: "
                f"dataset.train_sets has {len(spec.train_sets)} set(s), each "
                f"with {usable_rows} pulse(s) left after dataset.washout "
                f"({spec.washout}), so {train_rows} training pulse(s) in "
                f"total. That must be more than the {dataset.samples_per_pulse} "
                f"samples recorded per pulse. Lower dataset.washout, add sets "
                f"to dataset.train_sets, or record more pulses per set."
            ],
            guide=GUIDE,
        )

    if not spec.max_delay < usable_rows:
        raise_problems(
            [
                f"dataset.max_delay (the longest delay scored; "
                f"{DEFAULT_MAX_DELAY} if the line is not written) is "
                f"{spec.max_delay}, but only {usable_rows} "
                f"pulse(s) per set are left for scoring ({dataset.n_inputs} "
                f"pulses per set minus dataset.washout {spec.washout}). "
                f"max_delay must be less than that. Lower dataset.max_delay "
                f"(write 'max_delay: {usable_rows - 1}' or less under "
                f"dataset:) or dataset.washout, or record more pulses per set."
            ],
            guide=GUIDE,
        )

    tasks = tuple(tasks)
    for task in tasks:
        if task not in _TASKS:
            raise ValueError(f"score_rc: unknown task {task!r}; available: {_TASKS!r}")

    bits_used = readout.resolve_readout_bits(spec.readout_bits, spec.sample_interval)
    if bits_used is None:
        X = dataset.X
    else:
        full_scale = spec.readout_full_scale
        if full_scale is None or not full_scale > 0.0:
            raise_problems(
                [
                    f"dataset.readout_full_scale ({full_scale!r}) must be a "
                    f"number > 0 when dataset.readout_bits is "
                    f"{spec.readout_bits!r}."
                ],
                guide=GUIDE,
            )
        X = readout.quantize(dataset.X, bits_used, full_scale)

        if bits_used <= 4:
            warnings.warn(
                f"dataset.readout_bits is {spec.readout_bits!r} ({bits_used} "
                f"bit used): coarse quantization can distort memory scores. "
                f"Confirm this is intentional.",
                ReadoutWarning,
                stacklevel=2,
            )
        raw = np.asarray(dataset.X, dtype=np.float64)
        code = np.round(raw / (full_scale / 2.0**bits_used))
        n_saturated = int(np.count_nonzero(code > 2.0**bits_used - 1))
        if n_saturated:
            warnings.warn(
                f"{n_saturated} of {raw.size} response sample(s) "
                f"({100.0 * n_saturated / raw.size:.3g} %) lie above the "
                f"range of the {bits_used}-bit readout converter spanning "
                f"0..dataset.readout_full_scale ({full_scale:.3g} A) and are "
                f"clipped to its top code (largest value: "
                f"{float(np.max(raw)):.3g} A). Raise "
                f"readout_full_scale to cover the response, or confirm the "
                f"clipping is intentional.",
                ReadoutWarning,
                stacklevel=2,
            )

    samples_per_pulse = dataset.samples_per_pulse
    washout = spec.washout
    ridge_lambda = spec.ridge_lambda
    max_delay = spec.max_delay
    train_sets = spec.train_sets
    test_sets = spec.test_sets
    protocol = spec.protocol
    kfold = protocol == "kfold"

    stm_capacity, stm_curve = float("nan"), np.full(max_delay + 1, np.nan)
    pc_capacity, pc_curve = float("nan"), np.full(max_delay + 1, np.nan)
    per_fold: dict[str, np.ndarray] | None = {} if kfold else None

    if "stm" in tasks:
        if kfold:
            result = _kernel._run_delay_benchmark(
                dataset.u,
                X,
                target_fn=_kernel.stm_target,
                ridge_lambda=ridge_lambda,
                rng=np.random.default_rng(spec.shuffle_seed),
                samples_per_pulse=samples_per_pulse,
                washout=washout,
                max_delay=max_delay,
                train_sets=train_sets,
                test_sets=test_sets,
            )
            stm_capacity = result["mc_kfold"]
            stm_curve = result["cor2_kfold"]
            per_fold["stm"] = result["mc_per_fold"]
        else:
            result = _kernel._fixed_split(
                dataset.u,
                X,
                target_fn=_kernel.stm_target,
                train_sets=train_sets,
                test_sets=test_sets,
                samples_per_pulse=samples_per_pulse,
                washout=washout,
                ridge_lambda=ridge_lambda,
                max_delay=max_delay,
            )
            stm_capacity = result["mc"]
            stm_curve = result["cor2"]

    if "pc" in tasks:
        if kfold:
            result = _kernel._run_delay_benchmark(
                dataset.u,
                X,
                target_fn=_kernel.pc_target,
                ridge_lambda=ridge_lambda,
                rng=np.random.default_rng(spec.shuffle_seed),
                samples_per_pulse=samples_per_pulse,
                washout=washout,
                max_delay=max_delay,
                train_sets=train_sets,
                test_sets=test_sets,
            )
            pc_capacity = result["mc_kfold"]
            pc_curve = result["cor2_kfold"]
            per_fold["pc"] = result["mc_per_fold"]
        else:
            result = _kernel._fixed_split(
                dataset.u,
                X,
                target_fn=_kernel.pc_target,
                train_sets=train_sets,
                test_sets=test_sets,
                samples_per_pulse=samples_per_pulse,
                washout=washout,
                ridge_lambda=ridge_lambda,
                max_delay=max_delay,
            )
            pc_capacity = result["mc"]
            pc_curve = result["cor2"]

    settings = {
        "washout": spec.washout,
        "train_sets": train_sets,
        "test_sets": test_sets,
        "ridge_lambda": ridge_lambda,
        "max_delay": max_delay,
        "protocol": protocol,
        "shuffle_seed": spec.shuffle_seed,
        "samples_per_pulse": samples_per_pulse,
        "n_inputs": dataset.n_inputs,
        "n_sets": dataset.n_sets,
        "tasks": tasks,
        "readout_bits": spec.readout_bits,
        "readout_bits_used": "ideal" if bits_used is None else bits_used,
        "readout_full_scale": spec.readout_full_scale,
        "sample_interval": spec.sample_interval,
        "sample_interval_derived": spec.sample_interval_derived,
    }

    return RCScores(
        stm_capacity=stm_capacity,
        stm_curve=stm_curve,
        pc_capacity=pc_capacity,
        pc_curve=pc_curve,
        protocol=protocol,
        per_fold=per_fold,
        settings=settings,
    )
