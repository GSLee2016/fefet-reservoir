"""FE/AFE FET device -- internal gate voltage (fgate) self-consistent solve + time stepping.

Couples the ferroelectric layer (:mod:`~fefet_reservoir.device.domains`)
in series with the MOSFET (:mod:`~fefet_reservoir.device.mosfet`) and solves
the two together by charge balance: :func:`simulate_coupled_drain_current`.

At every time step the gate node floats. The ferroelectric stack's charge
``Q_FE(fgate)`` and the MOS charge ``Q_MOS(fgate)`` must agree, so the code
searches for the ``fgate`` that minimises ``|Q_FE - Q_MOS| / Afe``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import domains
from .mosfet import EPS_0, Q_E, drain_current, mos_charge, surface_potential

__all__ = ["DeviceParameters", "SolverSettings", "evaluate_charge_balance", "simulate_coupled_drain_current"]


@dataclass(frozen=True)
class DeviceParameters:
    """Material/structural constants of the device.

    The defaults match the example values in
    ``configs/example_simulated.yaml``. Derived quantities (``Cox``, ``gamma``,
    lengths in cm) are computed by the properties below. In a configuration
    file, the ``material:`` / ``mosfet:`` sections hold these values.
    """

    # Ferroelectric stack
    Ps: float = 18.42e-6  # [C/cm2]
    tfe: float = 10e-9  # [m]
    Afe: float = 4e-8  # [cm2] (2 um x 2 um)
    epsil_fe: float = 30.0

    # NLS switching (see :mod:`~fefet_reservoir.device.domains`)
    beta: float = 1.02
    alpha: float = 4.11
    tau_inf: float = 223e-9  # [s]

    # MOS
    Na: float = 1e17  # channel doping [cm-3]
    mobility: float = 300.0  # mobility [cm2/Vs]
    W: float = 10e-6  # [m]
    L: float = 10e-6  # [m]
    tox: float = 5e-9  # [m]
    vfb: float = -0.4  # flat-band voltage

    # Permittivities
    eps_ox: float = 3.9  # SiO2
    eps_si: float = 11.8  # Si

    # Bias
    vds: float = 50e-3
    vbs: float = 0.0

    @property
    def Tfecm(self) -> float:
        return self.tfe * 100.0

    @property
    def Wcm(self) -> float:
        return self.W * 100.0

    @property
    def Lcm(self) -> float:
        return self.L * 100.0

    @property
    def Toxcm(self) -> float:
        return self.tox * 100.0

    @property
    def Cox(self) -> float:
        return self.eps_ox * EPS_0 / self.Toxcm

    @property
    def gamma(self) -> float:
        return float(np.sqrt(2 * Q_E * self.Na * self.eps_si * EPS_0) / self.Cox)

    @property
    def vbd(self) -> float:
        return self.vbs - self.vds


@dataclass(frozen=True)
class SolverSettings:
    """Settings of the fgate scan."""

    initial_rep: float = 10.0  # initial half-width of the scan window [V]
    rep_floor: float = 0.001  # stop once the half-width falls to this [V]
    shrink: float = 1.0 / 3.0  # window shrink factor per pass
    grid_points: int = 4  # grid points per pass
    err_tolerance: float = 0.01e-6  # early exit
    err_min_init: float = 100000.0  # initial value of the best error


def evaluate_charge_balance(
    so,
    ho,
    NO: int,
    st,
    ht,
    NT: int,
    Vin: float,
    dt: float,
    Ea_oph,
    rngo,
    Ea_tph,
    Eb_tph,
    rngt,
    Tfecm: float,
    Afe: float,
    Ps: float,
    epsil_fe: float,
    Na: float,
    Cox: float,
    gamma: float,
    W: float,
    L: float,
    VGB: float,
    VFB: float,
    VDB: float,
    VSB: float,
    *,
    beta: float,
    alpha: float,
    tau_inf: float,
):
    """Charge-balance objective function for one trial gate voltage.

    Returns ``(out, so_out, ho_out, st_out, ht_out)`` where ``out`` is
    ``|Q_FE - Q_MOS| / Afe``.
    """
    vfe = Vin - VGB

    nls = dict(beta=beta, alpha=alpha, tau_inf=tau_inf)
    so_out, ho_out = domains.switch_fe_domains(
        s=so, h=ho, vfe=vfe, dt=dt, Ea_real=Ea_oph, rng=rngo, Tfecm=Tfecm, **nls
    )
    st_out, ht_out = domains.switch_afe_domains(
        s=st, h=ht, vfe=vfe, dt=dt, Ea_real=Ea_tph, Eb_real=Eb_tph, rng=rngt, Tfecm=Tfecm, **nls
    )

    total = NO + NT
    with np.errstate(invalid="ignore", divide="ignore"):
        Psum = Ps * Afe * (np.sum(so_out) + np.sum(st_out)) / total if total else np.nan
    if np.isnan(Psum):
        Psum = 0.0

    Q_FE = Psum + (Afe * vfe * epsil_fe * EPS_0 / Tfecm)
    Q_mos = mos_charge(Na, Cox, gamma, W, L, VGB, VFB, VDB, VSB)

    out = abs(Q_FE - Q_mos) / Afe
    return out, so_out, ho_out, st_out, ht_out


def simulate_coupled_drain_current(
    Vin,
    time,
    dt: float,
    Ea_oph,
    Ea_tph,
    Eb_tph,
    *,
    params: DeviceParameters | None = None,
    solver: SolverSettings | None = None,
    generator=None,
):
    """Simulate drain current over ``time``.

    The ferroelectric layer and the MOSFET are solved together at every time
    step: the internal gate voltage ``fgate`` is chosen so that the
    ferroelectric charge and the MOS charge balance.

    Parameters
    ----------
    Vin, time :
        Input voltage and time grid. Must be the same length.
    dt :
        Time step [s]. Passed straight through to the domain kernel.
    Ea_oph :
        Activation fields of the ferroelectric domains. Its length sets ``NO``.
    Ea_tph, Eb_tph :
        Activation / back-switching fields of the antiferroelectric domains.
        ``len(Ea_tph)`` sets ``NT``.

    Returns
    -------
    ``(Id_out, QFEden_monitoring, fgate_monitoring)``. ``Id_out`` is the
    noise-free device current; device noise is added afterwards, by the
    caller (:func:`fefet_reservoir.device.noise.apply_device_noise`).
    """
    p = params or DeviceParameters()
    cfg = solver or SolverSettings()
    gen = np.random.default_rng() if generator is None else generator

    Vin = np.asarray(Vin, dtype=np.float64).reshape(-1)
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    Ea_oph = np.asarray(Ea_oph, dtype=np.float64).reshape(-1)
    Ea_tph = np.asarray(Ea_tph, dtype=np.float64).reshape(-1)
    Eb_tph = np.asarray(Eb_tph, dtype=np.float64).reshape(-1)

    NO = Ea_oph.size
    NT = Ea_tph.size
    nt = time.size

    # FE domains start at random +/-1, AFE domains at 0.
    x1 = gen.binomial(1, 0.5, size=NO)
    so = -np.ones(NO, dtype=np.float64) + x1 * 2.0
    ho = np.zeros(NO, dtype=np.float64)
    st = np.zeros(NT, dtype=np.float64)
    ht = np.zeros(NT, dtype=np.float64)

    Id_gen = np.zeros(nt, dtype=np.float64)
    fgate_monitoring = np.zeros(nt, dtype=np.float64)
    QFEden_monitoring = np.zeros(nt, dtype=np.float64)

    vbs, vbd = p.vbs, p.vbd
    vgb = 0.0

    for i in range(nt):
        rngo = gen.random(NO)
        rngt = gen.random(NT)

        # ---- fgate scan -----------------------------------------------------
        best_charge_error = cfg.err_min_init
        fgate = vgb  # re-centre around the previous step's solution
        rep = cfg.initial_rep

        so_out, ho_out, st_out, ht_out = so, ho, st, ht

        while rep > cfg.rep_floor:
            for vgbt in np.linspace(fgate - rep, fgate + rep, cfg.grid_points):
                charge_error, so1, ho1, st1, ht1 = evaluate_charge_balance(
                    so, ho, NO, st, ht, NT, Vin[i], dt,
                    Ea_oph, rngo, Ea_tph, Eb_tph, rngt,
                    p.Tfecm, p.Afe, p.Ps, p.epsil_fe,
                    p.Na, p.Cox, p.gamma, p.Wcm, p.Lcm,
                    vgbt, p.vfb, -vbd, -vbs,
                    beta=p.beta, alpha=p.alpha, tau_inf=p.tau_inf,
                )
                if charge_error <= best_charge_error:
                    best_charge_error = charge_error
                    fgate = vgbt
                    so_out, ho_out, st_out, ht_out = so1, ho1, st1, ht1

            if best_charge_error < cfg.err_tolerance:
                break
            rep = rep * cfg.shrink

        # ---- step finalisation ----------------------------------------------
        vfe = Vin[i] - fgate
        vgb = fgate

        so, ho, st, ht = so_out, ho_out, st_out, ht_out

        phis = surface_potential(p.Na, p.gamma, vgb, p.vfb, -vbs)
        phid = surface_potential(p.Na, p.gamma, vgb, p.vfb, -vbd)
        Id = drain_current(
            phis, phid, p.Cox, p.mobility, p.gamma, p.Wcm, p.Lcm, vgb, p.vfb
        )

        total = NO + NT
        Psum = p.Ps * p.Afe * (np.sum(so) + np.sum(st)) / total if total else 0.0
        Q_FE = Psum + (p.Afe * vfe * p.epsil_fe * EPS_0 / p.Tfecm)

        fgate_monitoring[i] = fgate
        QFEden_monitoring[i] = Q_FE / p.Afe
        Id_gen[i] = Id

    return Id_gen, QFEden_monitoring, fgate_monitoring
