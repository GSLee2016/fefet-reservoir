"""Device noise added to the simulated read current:
``X_noisy = X + N(0, 1) * fraction * |X|``, drawn independently for every
sample. ``fraction`` is ``physics.device_noise`` (``0`` = no noise).
"""

from __future__ import annotations

import numpy as np

__all__ = ["apply_device_noise"]


def apply_device_noise(X, fraction: float, *, generator: np.random.Generator) -> np.ndarray:
    """Return ``X + N(0, 1) * fraction * |X|``.

    Parameters
    ----------
    X :
        Noise-free read current.
    fraction :
        Noise standard deviation as a fraction of ``|X|`` (e.g. ``0.002`` =
        0.2 %).
    generator :
        The ``numpy.random.Generator`` to draw from.
    """
    X = np.asarray(X, dtype=np.float64)
    fraction = float(fraction)
    if fraction == 0.0:
        return X.copy()
    return X + generator.standard_normal(X.shape) * fraction * np.abs(X)
