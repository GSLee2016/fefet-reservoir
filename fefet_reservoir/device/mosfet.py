"""Surface-potential MOSFET compact model (charge-sheet formulation).

This module is the channel half of an NLS ferroelectric-FET simulation: it
provides the surface potential of an n-channel device on a uniformly doped
p-type body, the total inversion (gate) charge of the channel, and the drain
current.
"""

import math

import numpy as np

__all__ = [
    "T_REF",
    "Q_E",
    "K_B",
    "EPS_0",
    "intrinsic_carrier_density",
    "thermal_voltage",
    "surface_potential",
    "mos_charge",
    "drain_current",
]


# --- Physical constants -----------------------------------------------------

T_REF = 300.0  # operating temperature assumed by the model [K]
Q_E = 1.602176462e-19  # elementary charge [C]
K_B = 1.3806503e-23  # Boltzmann constant [J/K]
EPS_0 = 8.85e-14  # permittivity of free space [F/cm]


# --- Numerical-method settings ----------------------------------------------

_STEP_RTOL = 1e-14  # convergence tolerance
_MAX_ITER = 100  # iteration cap
_EXP_ARG_MAX = 700.0  # clamp for math.exp overflow


def _quiet():
    """Suppress IEEE-754 underflow/overflow/divide reporting for one block."""
    return np.errstate(all="ignore")


def intrinsic_carrier_density(ctemp=T_REF):
    """Intrinsic carrier density of silicon at ``ctemp`` [K], in cm^-3."""
    return 5.29e19 * (ctemp / 300.0) ** 2.54 * math.exp(-6726.0 / ctemp)


def thermal_voltage(ctemp=T_REF):
    """Thermal voltage kT/q at ``ctemp`` [K], in V."""
    return K_B * ctemp / Q_E


def surface_potential(Na, gamma, VGB, VFB, VCB):
    """Surface potential at one end of the channel, in V.

    ``Na`` [cm^-3] body doping, ``gamma`` [V^1/2] body-effect coefficient,
    ``VGB`` [V] gate-to-body bias, ``VFB`` [V] flat-band voltage, ``VCB`` [V]
    channel-to-body bias at the end being evaluated.

    Raises ``RuntimeError`` if the iteration does not converge.
    """
    vt = thermal_voltage()
    ni = intrinsic_carrier_density()
    phi_f = vt * math.log(Na / ni)
    v_ov = float(VGB) - float(VFB)
    u = float(VCB) + 2.0 * phi_f
    g2 = gamma * gamma

    arg = -u / vt
    minority = math.exp(arg if arg < _EXP_ARG_MAX else _EXP_ARG_MAX)

    # --- initial guess ----------------------------------------------------
    psi = 0.0
    if u > 0.0:
        disc = 1.0 + 4.0 * (v_ov - vt) / g2
        if disc < 0.0:
            disc = 0.0
        psi_dep = v_ov + 0.5 * g2 * (1.0 - math.sqrt(disc))

        if psi_dep >= u and v_ov > 0.0:
            log_arg = (v_ov - u) ** 2 / (g2 * vt)
            if log_arg < 1.0:
                log_arg = 1.0
            psi = u + vt * math.log(log_arg)
        elif v_ov < -gamma * math.sqrt(vt):
            psi = -2.0 * vt * math.log(v_ov / (-gamma * math.sqrt(vt)))
        elif v_ov > vt:
            psi = psi_dep

    if v_ov > 0.0:
        psi = min(v_ov, psi)
    else:
        psi = max(v_ov, psi)

    # --- Halley iteration, safeguarded by bisection -----------------------
    pos = 0.0
    neg = v_ov

    for _ in range(_MAX_ITER):
        arg = -psi / vt
        e_acc = math.exp(arg if arg < _EXP_ARG_MAX else _EXP_ARG_MAX)
        arg = psi / vt
        e_inv = math.exp(arg if arg < _EXP_ARG_MAX else _EXP_ARG_MAX)
        s = psi + vt * (e_acc - 1.0) + vt * minority * (e_inv - psi / vt - 1.0)
        ds = 1.0 - e_acc + minority * (e_inv - 1.0)
        d2s = (e_acc + minority * e_inv) / vt

        r = (v_ov - psi) ** 2 / g2 - s
        dr = -2.0 * (v_ov - psi) / g2 - ds
        d2r = 2.0 / g2 - d2s

        if r > 0.0:
            pos = psi
        elif r < 0.0:
            neg = psi

        try:
            psi_new = psi - r / (dr - r * d2r / (2.0 * dr))
        except ZeroDivisionError:
            psi_new = math.inf

        lo, hi = (pos, neg) if pos <= neg else (neg, pos)
        if not lo <= psi_new <= hi:
            psi_new = 0.5 * (lo + hi)

        step = psi_new - psi
        psi = psi_new

        if abs(step) <= _STEP_RTOL * max(abs(psi), vt):
            return np.float64(psi)

    raise RuntimeError(
        f"surface potential did not converge in {_MAX_ITER} iterations at "
        f"VGB={VGB!r}, VFB={VFB!r}, VCB={VCB!r}"
    )


def mos_charge(Na, Cox, gamma, W, L, VGB, VFB, VDB, VSB):
    """The charge on the MOS side of the internal gate node, in C (charge-sheet gate charge).

    Sign opposite to the channel/inversion charge; this is the quantity the
    outer solver balances against the ferroelectric charge Q_FE.

    ``Na`` [cm^-3], ``Cox`` [F/cm^2], ``gamma`` [V^1/2], ``W``/``L`` [cm]
    channel width and length, ``VGB``/``VFB`` [V], ``VDB``/``VSB`` [V]
    drain-end and source-end channel-to-body bias. The result is the total
    charge, already multiplied by W*L -- not a density.
    """
    with _quiet():
        phid = surface_potential(Na, gamma, VGB, VFB, VDB)
        phis = surface_potential(Na, gamma, VGB, VFB, VSB)
        psi_m = 0.5 * (phid + phis)
        dpsi = phid - phis
        v_ov = VGB - VFB

        # no inversion layer at or below flat band
        if VGB <= VFB:
            return np.float64(Cox * W * L * (v_ov - psi_m))

        root_psi = np.sqrt(psi_m)
        alpha = 1.0 + gamma / (2.0 * root_psi)
        q_i = v_ov - psi_m - gamma * root_psi

        h = q_i / alpha + thermal_voltage()
        return np.float64(Cox * W * L * (v_ov - psi_m + dpsi**2 / (12.0 * h)))


def drain_current(phis, phid, Cox, mobility, gamma, W, L, VGB, VFB):
    """Drain current in A (charge-sheet model).

    ``phis``/``phid`` [V] source-end and drain-end surface potentials, already
    solved by the caller; ``Cox`` [F/cm^2], ``mobility`` [cm^2/(V*s)] channel
    mobility, ``gamma`` [V^1/2], ``W``/``L`` [cm] channel width and length,
    ``VGB``/``VFB`` [V].
    """
    with _quiet():
        psi_m = 0.5 * (phid + phis)
        dpsi = phid - phis
        v_ov = VGB - VFB

        # no inversion layer at or below flat band
        if VGB <= VFB:
            alpha = 0.0
            q_i = 0.0
        else:
            root_psi = np.sqrt(psi_m)
            alpha = 1.0 + gamma / (2.0 * root_psi)
            q_i = v_ov - psi_m - gamma * root_psi

        return np.float64(
            (W / L) * mobility * Cox * (q_i + alpha * thermal_voltage()) * dpsi
        )
