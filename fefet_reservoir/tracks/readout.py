"""Readout converter applied to the device response before scoring.

A real measurement never hands the readout layer an exact current: the
response is digitised by a converter with a finite number of bits over a
finite range. :func:`quantize` models that converter (an N-bit unipolar
converter spanning ``0..full_scale``, for a drain current that flows in one
direction), and :func:`score_rc <fefet_reservoir.tracks.rc.score_rc>`
applies it to the response matrix right before the scoring kernels.

The resolution is set in the ``dataset:`` section as ``readout_bits``:

* a whole number of bits (1 to 24);
* ``auto`` -- the bit count a converter fast enough for the sample interval
  typically delivers (:func:`recommended_bits`);
* ``ideal`` -- no quantization at all (the response is scored as-is).
"""

from __future__ import annotations

import numpy as np

__all__ = ["quantize", "recommended_bits", "resolve_readout_bits", "READOUT_BITS_TABLE"]

#: ``(minimum sample interval [s], bits)``, fastest-last: the first row whose
#: interval is at or below the sample interval applies. The time between
#: the samples of one pulse bounds the converter's conversion rate, and
#: resolution falls as the rate rises.
READOUT_BITS_TABLE: tuple[tuple[float, int], ...] = (
    (1e-3, 16),  # >= 1 ms: precision source-measure unit
    (1e-6, 12),  # 1 us .. 1 ms
    (1e-7, 10),  # 100 ns .. 1 us
    (0.0, 8),  # < 100 ns: fast pulsed measurement / on-chip readout
)


def quantize(X, bits: int, full_scale: float) -> np.ndarray:
    """N-bit unipolar converter spanning ``0..full_scale``.

    ``LSB = full_scale / 2**bits``; each sample is rounded to the nearest
    code, codes are clipped to ``[0, 2**bits - 1]``, and ``code * LSB`` is
    returned. Samples above the range therefore saturate at the top code,
    and negative samples (e.g. small negative noise in measured data) map
    to 0.
    """
    X = np.asarray(X, dtype=np.float64)
    bits = int(bits)
    lsb = float(full_scale) / 2.0**bits
    code = np.clip(np.round(X / lsb), 0.0, 2.0**bits - 1)
    return code * lsb


def recommended_bits(sample_interval: float) -> int:
    """Typical converter resolution for a given sample interval [s].

    The sample interval is the time between the samples of one pulse:
    ``>= 1e-3`` -> 16 bit, ``1e-6 .. <1e-3`` -> 12, ``1e-7 .. <1e-6`` -> 10,
    ``< 1e-7`` -> 8 (:data:`READOUT_BITS_TABLE`).
    """
    interval = float(sample_interval)
    if not (np.isfinite(interval) and interval > 0.0):
        raise ValueError(f"sample_interval: {sample_interval!r} must be a number > 0 (seconds).")
    for lower, bits in READOUT_BITS_TABLE:
        if interval >= lower:
            return bits
    raise AssertionError("unreachable: the last table row has a lower bound of 0")


def resolve_readout_bits(readout_bits, sample_interval: float | None) -> int | None:
    """The bit count actually used for a configured ``readout_bits``.

    ``'ideal'`` -> ``None`` (no quantization); ``'auto'`` ->
    :func:`recommended_bits` of *sample_interval*; a whole number -> itself.

    Raises
    ------
    ValueError
        For ``'auto'`` without a sample interval, or any other value.
    """
    if readout_bits == "ideal":
        return None
    if readout_bits == "auto":
        if sample_interval is None:
            raise ValueError(
                "readout_bits: 'auto' needs a sample interval (dataset."
                "sample_interval, or a pulse: section to derive it from)."
            )
        return recommended_bits(sample_interval)
    if (
        isinstance(readout_bits, (int, float))
        and not isinstance(readout_bits, bool)
        and np.isfinite(readout_bits)
        and float(readout_bits) == int(readout_bits)
        and 1 <= int(readout_bits) <= 24
    ):
        return int(readout_bits)
    raise ValueError(
        f"readout_bits: {readout_bits!r} must be 'auto', 'ideal' or a whole "
        f"number of bits from 1 to 24."
    )
