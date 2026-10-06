"""Random-field generators for domain activation/back-switching fields, one per domain kind.

Model
-----
The activation field E_a and back-switching field E_b are drawn from normal
distributions (mean, std).

On randomness
-------------
Each function optionally takes a ``generator``. Without one, a fresh
unseeded ``numpy.random.default_rng()`` is used, so results are reproducible
only at the distribution level. For a reproducible run or a test, pass a
``numpy.random.Generator`` with a fixed seed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sample_Ea",
    "sample_Eb",
]


def _generator(generator) -> np.random.Generator:
    if generator is None:
        return np.random.default_rng()
    return generator


def sample_Ea(N: int, Ea: float, sigma_a: float, *, generator=None) -> np.ndarray:
    """Normally distributed activation field.

    Returns a 1-D array of length ``N`` drawn from a normal distribution with
    mean ``Ea`` and standard deviation ``sigma_a``.
    """
    return _generator(generator).normal(loc=Ea, scale=sigma_a, size=int(N))


def sample_Eb(N: int, Eb: float, sigma_b: float, *, generator=None) -> np.ndarray:
    """Normally distributed antiferroelectric back-switching field.

    Returns a 1-D array of length ``N`` drawn from a normal distribution with
    mean ``Eb`` and standard deviation ``sigma_b``.
    """
    return _generator(generator).normal(loc=Eb, scale=sigma_b, size=int(N))
