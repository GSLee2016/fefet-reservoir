"""Path B -- generate a standard-format RC dataset from the simulated device.

Simulates the device and writes the same ``input.csv`` /
``response.csv`` format that :func:`fefet_reservoir.io.load_rc_dataset` reads
(see README.md, "Preparing measured data (path A)").

The same ``seed`` gives the same result. Device noise uses its own random
stream, so changing ``physics.device_noise`` leaves the input sequence and
the noise-free response unchanged.
"""

from __future__ import annotations

import numbers
from pathlib import Path

import numpy as np

from ..config.errors import ConfigError
from ..config.loader import Config
from .distributions import sample_Ea, sample_Eb
from .fe_afe_fet import simulate_coupled_drain_current, DeviceParameters
from .noise import apply_device_noise
from .waveforms import pulse_train

__all__ = ["simulate_rc_dataset", "write_rc_dataset"]


def _device_parameters(cfg: Config) -> DeviceParameters:
    """Configuration ``material:`` / ``mosfet:`` sections -> :class:`DeviceParameters`."""
    material = cfg.section("material")
    mos = cfg.section("mosfet")
    return DeviceParameters(
        Ps=float(material["Ps"]),
        tfe=float(material["tfe"]),
        Afe=float(material["Afe"]),
        epsil_fe=float(material["epsil_fe"]),
        beta=float(material["beta"]),
        alpha=float(material["alpha"]),
        tau_inf=float(material["tau_inf"]),
        Na=float(mos["Na"]),
        mobility=float(mos["mobility"]),
        W=float(mos["W"]),
        L=float(mos["L"]),
        tox=float(mos["tox"]),
        vfb=float(mos["vfb"]),
        eps_ox=float(mos["eps_ox"]),
        eps_si=float(mos["eps_si"]),
        vds=float(mos["vds"]),
        vbs=float(mos["vbs"]),
    )


def simulate_rc_dataset(
    cfg: Config,
    *,
    n_inputs: int,
    n_sets: int,
    seed: int,
    device_noise: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``n_sets`` independent device runs of ``n_inputs`` pulses each.

    The same seed gives the same ``(u, X)``.

    Parameters
    ----------
    cfg :
        A loaded :class:`~fefet_reservoir.config.loader.Config`. Reads
        ``cfg.section("pulse")``, ``cfg.section("material")``,
        ``cfg.section("mosfet")``, ``cfg.section("physics")`` and
        ``cfg.derived["n_tetra"]`` / ``["n_ortho"]`` / ``["dt"]`` /
        ``["t_slope"]`` / ``["signal_ref"]``.
    n_inputs :
        Number of pulses per set (``N``, the rows of ``input.csv``).
    n_sets :
        Number of independent sets (``S``, the columns of both files). Must
        be at least 2: one set to train the readout, one to test it.
    seed :
        Seed for the single ``numpy.random.Generator`` that drives the
        entire run. The same seed gives ``(u, X)`` reproducible on the same
        software environment.
    device_noise :
        Device-noise standard deviation as a fraction of the read current
        (``0`` = none), see :func:`~fefet_reservoir.device.noise.apply_device_noise`.
        ``None`` (the default) takes ``physics.device_noise`` from the
        configuration, where it must then be written.

    Returns
    -------
    ``(u, X)`` : a ``(n_inputs, n_sets)`` binary array and a
    ``(samples_per_step * n_inputs, n_sets)`` response array, in the shape
    :func:`fefet_reservoir.io.load_rc_dataset` expects.
    """
    if int(n_inputs) < 1:
        raise ValueError(
            f"n_inputs: {n_inputs!r} must be at least 1 (it is N, the number "
            f"of pulses per set)."
        )
    if int(n_sets) < 2:
        raise ValueError(
            f"n_sets: {n_sets!r} must be at least 2. At least two sets are "
            f"needed to split train/test."
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed: {seed!r} must be an int.")

    pulse = cfg.section("pulse")
    material = cfg.section("material")
    physics = cfg.section("physics")

    if device_noise is None:
        device_noise = physics.get("device_noise")
        if device_noise is None:
            raise ConfigError(
                f"physics.device_noise is missing in {cfg.path}. It is the "
                f"device-noise standard deviation as a fraction of the read "
                f"current (0 = no device noise; e.g. 0.002 = 0.2 %). It "
                f"changes the simulated response, so it must be written in "
                f"the configuration file."
            )
    if (
        isinstance(device_noise, bool)
        or not isinstance(device_noise, numbers.Real)
        or not np.isfinite(device_noise)
        or device_noise < 0
    ):
        raise ValueError(
            f"device_noise: {device_noise!r} must be a real number >= 0 (a "
            f"fraction of the read current; 0 = no device noise)."
        )

    n_tetra = cfg.derived["n_tetra"]
    n_ortho = cfg.derived["n_ortho"]
    dt = cfg.derived["dt"]
    t_slope = cfg.derived["t_slope"]
    signal_ref = cfg.derived["signal_ref"]
    samples_per_step = int(pulse["samples_per_step"])

    rng = np.random.default_rng(seed)
    params = _device_parameters(cfg)

    # 1) The whole binary input block, in one call, drawn before anything else.
    u = (rng.random((n_inputs, n_sets)) > 0.5).astype(np.float64)

    # 2) One ferroelectric-layer draw, shared by every set.
    ea_oph = sample_Ea(n_ortho, float(material["ea"]), float(material["sigma_a"]), generator=rng)
    ea_tph = sample_Ea(n_tetra, float(material["ea"]), float(material["sigma_a"]), generator=rng)
    eb_tph = sample_Eb(n_tetra, float(material["eb"]), float(material["sigma_b"]), generator=rng)

    target = samples_per_step * n_inputs
    X = np.zeros((target, n_sets), dtype=np.float64)

    # 3) Per-set device runs. Each column of u was already
    # drawn in step 1, so nothing here draws another input sequence.
    for j in range(n_sets):
        op_time, vin = pulse_train(
            u[:, j],
            shape=str(pulse["shape"]),
            t_pw=float(pulse["t_pw"]),
            signal_on=float(pulse["signal_on"]),
            signal_off=float(pulse["signal_off"]),
            dt=dt,
            t_slope=t_slope,
            signal_ref=signal_ref,
            vin_sampling=str(pulse.get("vin_sampling", "endpoint")),
        )
        id_out, _, _ = simulate_coupled_drain_current(
            vin,
            op_time,
            dt,
            ea_oph,
            ea_tph,
            eb_tph,
            params=params,
            generator=rng,
        )
        if id_out.size < target:
            raise ValueError(
                f"device simulation produced only {id_out.size} sample(s) but "
                f"{target} are needed (samples_per_step * n_inputs = "
                f"{samples_per_step} * {n_inputs}). Check pulse.samples_per_step "
                f"and pulse.t_pw in the configuration."
            )
        # Asymmetric on purpose: too few samples is a hard error (above) --
        # there is no meaningful value to pad with -- but extra trailing samples
        # (the time grid can run a sample long) are truncated to `target`.
        X[:, j] = id_out[:target]

    # 4) Device noise.
    noise_seq = np.random.SeedSequence(seed).spawn(1)[0]
    for j, set_seq in enumerate(noise_seq.spawn(n_sets)):
        X[:, j] = apply_device_noise(
            X[:, j], device_noise, generator=np.random.default_rng(set_seq)
        )

    return u, X


def write_rc_dataset(
    out_dir: str | Path,
    u: np.ndarray,
    X: np.ndarray,
    *,
    input_file: str = "input.csv",
    response_file: str = "response.csv",
) -> None:
    """Write ``(u, X)`` as the two-CSV RC dataset format (``input.csv`` / ``response.csv``).

    Creates ``out_dir`` if it does not already exist. Both files are written
    with ``np.savetxt(delimiter=",", fmt="%.17g", encoding="utf-8-sig")`` --
    full float64 precision, so the round trip through
    :func:`fefet_reservoir.io.load_rc_dataset` is exact.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / input_file, u, delimiter=",", fmt="%.17g", encoding="utf-8-sig")
    np.savetxt(out_dir / response_file, X, delimiter=",", fmt="%.17g", encoding="utf-8-sig")
