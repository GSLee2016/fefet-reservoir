"""Domain-switching kernel (vectorised with numpy): switches the
ferroelectric and antiferroelectric domains at every time step.

This is the NLS (nucleation-limited switching) model. The switching
parameters ``beta``, ``alpha`` and ``tau_inf`` come from the configuration
(``material.beta`` / ``material.alpha`` / ``material.tau_inf``); the
ferroelectric and antiferroelectric domains of one layer share the same
values. A domain with activation field ``Ea`` switches with time constant
``tau = tau_inf * exp((Ea / |E|)^alpha)`` (antiferroelectric domains use the
field measured from ``Eb`` instead of ``|E|``; see :func:`switch_afe_domains`),
and ``beta`` is the exponent of the switching probability
``1 - exp(h^beta - h_new^beta)``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["switch_fe_domains", "switch_afe_domains"]


def _switch_probability(h, h_new, beta: float):
    """``1 - exp(h^beta - h_new^beta)``."""
    with np.errstate(over="ignore", invalid="ignore"):
        return 1.0 - np.exp(
            np.power(h, beta) - np.power(h_new, beta)
        )


def switch_fe_domains(
    s,
    h,
    vfe: float,
    dt: float,
    Ea_real,
    rng,
    Tfecm: float,
    *,
    beta: float,
    alpha: float,
    tau_inf: float,
):
    """One time step of ferroelectric domain switching (NLS model)."""
    s = np.array(s, dtype=np.float64).copy()
    h = np.array(h, dtype=np.float64).copy()
    Ea_real = np.asarray(Ea_real, dtype=np.float64)
    rng = np.asarray(rng, dtype=np.float64).reshape(-1)

    E = np.float64(vfe) / np.float64(Tfecm)

    active = (s * E) < 0

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        tau = tau_inf * np.exp(np.power(Ea_real / abs(E), alpha))
        h_new = np.where(active, h + np.float64(dt) / tau, h)

    P = np.where(active, _switch_probability(h, h_new, beta), 0.0)

    h_out = h_new
    flip = active & (rng <= P)
    s_out = np.where(flip, -s, s)
    h_out = np.where(flip, 0.0, h_out)

    return s_out, h_out


def switch_afe_domains(
    s,
    h,
    vfe: float,
    dt: float,
    Ea_real,
    Eb_real,
    rng,
    Tfecm: float,
    *,
    beta: float,
    alpha: float,
    tau_inf: float,
):
    """One time step of antiferroelectric domain switching (NLS model): a
    domain polarises when ``|E| > Eb``, relaxes when ``|E| < Eb``, and
    reverses when the field opposes it."""
    s = np.array(s, dtype=np.float64).copy()
    h = np.array(h, dtype=np.float64).copy()
    Ea_real = np.asarray(Ea_real, dtype=np.float64)
    Eb_real = np.asarray(Eb_real, dtype=np.float64)
    rng = np.asarray(rng, dtype=np.float64).reshape(-1)

    E = np.float64(vfe) / np.float64(Tfecm)
    absE = abs(E)

    polarised = s != 0
    aligned = (s * absE) == E

    b_polarise = (~polarised) & (absE > Eb_real)
    b_relax = polarised & (absE < Eb_real)
    b_reverse = polarised & (absE > Eb_real) & (~aligned)
    active = b_polarise | b_relax | b_reverse

    denom = np.where(
        b_relax & (~aligned),
        np.abs(absE + Eb_real),
        np.abs(absE - Eb_real),
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        tau = tau_inf * np.exp(np.power(Ea_real / denom, alpha))
        h_new = np.where(active, h + np.float64(dt) / tau, h)

    P = np.where(active, _switch_probability(h, h_new, beta), 0.0)
    switch = active & (rng <= P)

    s_polarised = 1.0 if E >= 0 else -1.0
    s_out = np.select(
        [switch & b_polarise, switch & b_relax, switch & b_reverse],
        [np.full_like(s, s_polarised), np.zeros_like(s), -s],
        default=s,
    )

    h_out = np.where(switch, 0.0, h_new)

    return s_out, h_out
