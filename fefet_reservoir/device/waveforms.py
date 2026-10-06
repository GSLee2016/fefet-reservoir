"""Input waveform construction: breakpoint resampling and 0/1 pulse-train drivers.

Does two things.

1. :func:`breakpoints_to_waveform` -- resamples a list of ``(segment length,
   voltage)`` breakpoints onto the simulation time grid. Pure ``linspace``
   arithmetic with no randomness.
2. :func:`digital_to_voltage` / :func:`build_breakpoints` /
   :func:`operating_time_grid` / :func:`pulse_train` -- turns a 0/1 digital
   input into a pulse-train voltage waveform. The pulse parameters come from
   the configuration file's ``pulse:`` section.

Resampling modes
----------------
Two resampling modes are available; they differ in **sampling phase**.

``endpoint``
    ``linspace(v0, v1, period)`` -- includes both endpoints of the segment,
    so two adjacent segments share one overlapping sample at the breakpoint.

``half_sample_offset``
    ``samples = linspace(v0, v1, 2*period+1)``, keeping every second sample
    starting with the second (``samples[1::2]``) -- every output sample is
    shifted back by half a step, and the segment's starting voltage never
    appears at all.

The default is ``endpoint``; neither mode is more correct in general. If
the sampling phase matters for your results, state ``pulse.vin_sampling``
explicitly in the configuration file.
"""

from __future__ import annotations

import numpy as np

from ..config.schema import PULSE_SHAPES
from ..config.schema import VIN_SAMPLING as VARIANTS

__all__ = [
    "VARIANTS",
    "PULSE_SHAPES",
    "breakpoints_to_waveform",
    "digital_to_voltage",
    "build_breakpoints",
    "operating_time_grid",
    "pulse_train",
]


def _ramp_samples(start: float, stop: float, num: int) -> np.ndarray:
    """``numpy.linspace``, except that ``num == 1`` returns the endpoint
    (``stop``) rather than ``start``."""
    if num <= 0:
        return np.empty(0, dtype=np.float64)
    if num == 1:
        return np.array([stop], dtype=np.float64)
    return np.linspace(start, stop, num, dtype=np.float64)


def breakpoints_to_waveform(time, signal, *, variant: str = "endpoint") -> np.ndarray:
    """Resample a list of breakpoints onto the ``time`` grid.

    Parameters
    ----------
    time :
        1-D simulation time grid [s], ascending.
    signal :
        ``(M, 2)`` array. Column 0 is segment **length**, column 1 is the
        voltage at that breakpoint. Segment ``t`` ramps from
        ``signal[t, 1]`` to ``signal[t+1, 1]``, ending at the cumulative sum
        of lengths through row ``t+1``.
    variant :
        ``"endpoint"`` (default) or ``"half_sample_offset"``.
        Selected in a configuration file via ``pulse.vin_sampling`` -- see
        the module note above.

    Returns
    -------
    ``np.ndarray`` of length ``len(time)``. Any tail not reached by the
    breakpoint list is 0.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown breakpoints_to_waveform variant {variant!r}; available: {list(VARIANTS)}")

    time = np.asarray(time, dtype=np.float64).reshape(-1)
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] < 2:
        raise ValueError(f"signal must be (M, 2); got shape {signal.shape}")

    Vin = np.zeros(time.size, dtype=np.float64)
    durations = signal[:, 0]
    values = signal[:, 1]

    acctime = 1  # 1-based write cursor (converted to 0-based at the slice below)
    for t in range(signal.shape[0] - 1):
        # number of grid points up to this segment's end, minus those already written
        boundary = float(np.sum(durations[: t + 2]))
        period = int(np.count_nonzero(time <= boundary)) - acctime + 1
        if period <= 0:
            acctime += period
            continue

        if variant == "endpoint":
            block = _ramp_samples(values[t], values[t + 1], period)
        else:
            samples = _ramp_samples(values[t], values[t + 1], 2 * period + 1)
            block = samples[1::2]  # every second sample, starting with the second (half-sample offset)

        start = acctime - 1  # to 0-based
        stop = start + period
        if start >= Vin.size:
            break
        if stop > Vin.size:  # defensive: truncate a block that would overrun the time grid
            block = block[: Vin.size - start]
            stop = Vin.size
        Vin[start:stop] = block

        acctime += period

    return Vin


# ---------------------------------------------------------------------------
# 0/1 digital input -> pulse-train voltage waveform
# ---------------------------------------------------------------------------


def digital_to_voltage(
    u_in,
    *,
    signal_on: float,
    signal_off: float,
) -> np.ndarray:
    """Turn a 0/1 input into the target voltage of each pulse.

    ``signal_V = u_in * (signal_on - signal_off) + signal_off``, i.e. symbol 0
    is driven at ``signal_off`` and symbol 1 at ``signal_on``.
    """
    u_in = np.asarray(u_in, dtype=np.float64).reshape(-1)
    span = float(signal_on) - float(signal_off)
    return u_in * span + float(signal_off)


def build_breakpoints(
    signal_v,
    *,
    shape: str,
    t_pw: float,
    t_slope: float,
    signal_off: float,
    signal_ref: float,
) -> np.ndarray:
    """Breakpoint list to feed into :func:`breakpoints_to_waveform`.

    Square wave: starts at ``signal_off``, and for each sample appends
    ``[t_slope, V]`` then ``[(t_pw - t_slope), V]`` -- a fast ramp up to the
    target voltage, held until the pulse ends.

    Triangular wave: starts at ``signal_ref``, and for each sample appends
    ``[t_pw/2, V]`` then ``[t_pw/2, signal_ref]`` -- ramps up to the target
    voltage and back to the reference voltage every step.

    ``t_slope`` and ``signal_ref`` come from ``cfg.derived`` (computed by
    :func:`fefet_reservoir.config.loader.compute_derived`), not from the
    configuration file.
    """
    signal_v = np.asarray(signal_v, dtype=np.float64).reshape(-1)
    n = signal_v.size
    rows: list[tuple[float, float]] = [(0.0, 0.0)] * (2 * n + 1)
    if shape == "square":
        rows[0] = (0.0, float(signal_off))
        for i in range(n):
            rows[1 + 2 * i] = (float(t_slope), float(signal_v[i]))
            rows[2 + 2 * i] = (float(t_pw) - float(t_slope), float(signal_v[i]))
    elif shape == "triangular":
        rows[0] = (0.0, float(signal_ref))
        half = float(t_pw) / 2.0
        for i in range(n):
            rows[1 + 2 * i] = (half, float(signal_v[i]))
            rows[2 + 2 * i] = (half, float(signal_ref))
    else:
        raise ValueError(
            f"pulse.shape: '{shape}' is not a known waveform. "
            f"Available values: {' | '.join(PULSE_SHAPES)}"
        )
    return np.asarray(rows, dtype=np.float64)


def operating_time_grid(end_time: float, dt: float) -> np.ndarray:
    """Midpoint time grid ``dt/2, 3*dt/2, ...`` covering ``end_time``
    (``round(end_time / dt)`` points).
    """
    n = int(round(float(end_time) / float(dt)))
    return float(dt) / 2.0 + float(dt) * np.arange(n, dtype=np.float64)


def pulse_train(
    u_in,
    *,
    shape: str,
    t_pw: float,
    signal_on: float,
    signal_off: float,
    dt: float,
    t_slope: float,
    signal_ref: float,
    vin_sampling: str = "endpoint",
) -> tuple[np.ndarray, np.ndarray]:
    """One row of 0/1 input -> ``(op_time, Vin)``. Calls the four functions above in order.

    ``dt``, ``t_slope``, ``signal_ref`` come from ``cfg.derived``
    (``dt = t_pw / samples_per_step``, ``t_slope = t_pw / 1000``,
    ``signal_ref = (signal_on + signal_off) / 2``).
    """
    signal_v = digital_to_voltage(
        u_in,
        signal_on=signal_on,
        signal_off=signal_off,
    )
    signal = build_breakpoints(
        signal_v,
        shape=shape,
        t_pw=t_pw,
        t_slope=t_slope,
        signal_off=signal_off,
        signal_ref=signal_ref,
    )
    end_time = float(np.sum(signal[:, 0]))
    op_time = operating_time_grid(end_time, dt)
    Vin = breakpoints_to_waveform(op_time, signal, variant=vin_sampling)
    return op_time, Vin
