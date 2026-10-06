"""Allowed values and required keys for configuration fields."""

from __future__ import annotations

__all__ = [
    "DEVICES",
    "VIN_SAMPLING",
    "PULSE_SHAPES",
    "DERIVED_KEYS",
    "REQUIRED_DATASET_KEYS",
    "OPTIONAL_DATASET_KEYS",
    "KNOWN_DATASET_KEYS",
    "REQUIRED_SIMULATION_KEYS",
    "KNOWN_SIMULATION_KEYS",
    "REQUIRED_SIMULATION_PHYSICS_KEYS",
    "KNOWN_PHYSICS_KEYS",
    "REQUIRED_DEVICE_PARAMETER_KEYS",
    "KNOWN_MATERIAL_KEYS",
    "KNOWN_MOSFET_KEYS",
    "KNOWN_DOMAINS_KEYS",
    "KNOWN_PULSE_KEYS",
    "POSITIVE_DEVICE_PARAMETER_KEYS",
    "READOUT_BITS_MODES",
    "READOUT_BITS_MIN",
    "READOUT_BITS_MAX",
    "SPLIT_PRECEDENTS",
    "DEFAULT_INPUT_FILE",
    "DEFAULT_RESPONSE_FILE",
    "PROTOCOLS",
    "DEFAULT_MAX_DELAY",
]

#: Device kind. Inside the simulator this becomes the split of switching
#: domains between the tetragonal (antiferroelectric) and orthorhombic
#: (ferroelectric) phases.
DEVICES = ("fefet", "afefet")

#: How the drive waveform is resampled onto the simulation grid. The second
#: option shifts the whole waveform by half a sample.
VIN_SAMPLING = ("endpoint", "half_sample_offset")

PULSE_SHAPES = ("square", "triangular")

#: Names that must NOT appear in a configuration file: they are computed from
#: other values.
DERIVED_KEYS = ("dt", "t_slope", "signal_ref", "n_tetra", "n_ortho", "NT", "NO")

#: Score-determining values of the RC track. All of them are required in the
#: ``dataset:`` section; a missing one stops loading with a message.
#: ``readout_bits`` is the resolution of the converter that reads the device
#: response before scoring (``auto``, ``ideal`` or a whole number of bits).
#: ``protocol`` is the scoring procedure (see ``PROTOCOLS``).
REQUIRED_DATASET_KEYS = (
    "washout", "train_sets", "test_sets", "ridge_lambda", "readout_bits", "protocol",
)

#: The two named ``dataset.readout_bits`` settings: ``auto`` picks the bit
#: count from the sample interval (``fefet_reservoir.tracks.readout.
#: recommended_bits``); ``ideal`` skips quantization entirely.
READOUT_BITS_MODES = ("auto", "ideal")

#: Allowed range for an explicit ``dataset.readout_bits`` (whole number).
READOUT_BITS_MIN = 1
READOUT_BITS_MAX = 24

#: The scoring protocol: a fixed single train/test split, or a shuffled
#: cross-validation with one fold per set (cyclic rotation of the shuffled
#: sets). See README.md, "Scoring".
PROTOCOLS = ("fixed", "kfold")

#: Default for dataset.max_delay: the longest delay summed into the memory
#: capacity (delays 1..max_delay).
DEFAULT_MAX_DELAY = 20

#: Optional keys allowed in the ``dataset:`` section, beyond the required
#: six and ``path``.
#: ``readout_full_scale`` and ``sample_interval`` are listed here because
#: they are only CONDITIONALLY required: ``readout_full_scale`` [A] unless
#: ``readout_bits`` is ``ideal``; ``sample_interval`` [s] when
#: ``readout_bits`` is ``auto`` and the file has no ``pulse:`` section to
#: derive it from (``pulse.t_pw / pulse.samples_per_step``). The loader
#: enforces both conditions.
OPTIONAL_DATASET_KEYS = (
    "shuffle_seed",
    "max_delay",
    "input_file",
    "response_file",
    "readout_full_scale",
    "sample_interval",
)

#: Every key ``dataset:`` may contain; anything else is rejected.
KNOWN_DATASET_KEYS = REQUIRED_DATASET_KEYS + OPTIONAL_DATASET_KEYS + ("path",)

#: Required keys of the ``simulation:`` section (run size, seed and output
#: folder for ``simulate-rc``). If one is missing, simulate-rc stops and
#: names it.
REQUIRED_SIMULATION_KEYS = ("n_inputs", "n_sets", "seed", "output_dir")

#: Every key ``simulation:`` may contain (all required).
KNOWN_SIMULATION_KEYS = REQUIRED_SIMULATION_KEYS

#: ``physics:`` keys required whenever the file has a ``simulation:``
#: section. ``device_noise`` is the device-noise standard deviation as a
#: fraction of the read current (``0`` = none); it changes the simulated
#: response, so it must be written in the configuration file.
REQUIRED_SIMULATION_PHYSICS_KEYS = ("device_noise",)

#: Every key ``physics:`` may contain.
KNOWN_PHYSICS_KEYS = ("device_noise",)

#: Device and drive values the simulator reads (``simulate-rc``), as paths in
#: the file (``t_ratio`` is a top-level line). They are written in the
#: configuration file; ``simulate-rc`` checks that every one is present before
#: it starts. Scoring alone never needs them.
REQUIRED_DEVICE_PARAMETER_KEYS = (
    "t_ratio",
    "material.Ps",
    "material.tfe",
    "material.Afe",
    "material.epsil_fe",
    "material.ea",
    "material.sigma_a",
    "material.eb",
    "material.sigma_b",
    "material.beta",
    "material.alpha",
    "material.tau_inf",
    "mosfet.Na",
    "mosfet.mobility",
    "mosfet.W",
    "mosfet.L",
    "mosfet.tox",
    "mosfet.vfb",
    "mosfet.eps_ox",
    "mosfet.eps_si",
    "mosfet.vds",
    "mosfet.vbs",
    "domains.count",
    "pulse.shape",
    "pulse.t_pw",
    "pulse.signal_on",
    "pulse.signal_off",
    "pulse.samples_per_step",
)


def _keys_of(section: str) -> tuple[str, ...]:
    prefix = section + "."
    return tuple(k[len(prefix):] for k in REQUIRED_DEVICE_PARAMETER_KEYS if k.startswith(prefix))


#: Every key the ``material:`` / ``mosfet:`` / ``domains:`` / ``pulse:``
#: sections may contain (``pulse.vin_sampling`` is the only optional one).
KNOWN_MATERIAL_KEYS = _keys_of("material")
KNOWN_MOSFET_KEYS = _keys_of("mosfet")
KNOWN_DOMAINS_KEYS = _keys_of("domains")
KNOWN_PULSE_KEYS = _keys_of("pulse") + ("vin_sampling",)

#: Device parameters that are physically meaningful only as a number > 0
#: (thicknesses, areas, lengths, permittivities, doping, mobility,
#: polarization, the NLS switching parameters). Fields, voltages and spreads
#: are not listed: their sign or a zero value is not checked here.
POSITIVE_DEVICE_PARAMETER_KEYS = (
    "material.Ps",
    "material.tfe",
    "material.Afe",
    "material.epsil_fe",
    "material.beta",
    "material.alpha",
    "material.tau_inf",
    "mosfet.Na",
    "mosfet.mobility",
    "mosfet.W",
    "mosfet.L",
    "mosfet.tox",
    "mosfet.eps_ox",
    "mosfet.eps_si",
)

#: Example splits -- a starting point, not a standard. Indexed by number of
#: sets and shown in the suggestion block.
SPLIT_PRECEDENTS = (
    (10, "1-8", "9-10"),
    (30, "1-25", "26-30"),
    (100, "1-80", "81-100"),
)

#: Default file names inside a dataset directory. Both can be changed.
DEFAULT_INPUT_FILE = "input.csv"
DEFAULT_RESPONSE_FILE = "response.csv"
