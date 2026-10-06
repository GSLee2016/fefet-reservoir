"""NLS compact model for ferroelectric / antiferroelectric FETs.

Generates device responses for users without hardware, written in the same
input.csv / response.csv format as measured data.

Modules (not imported eagerly here -- import the one you need directly):

  distributions.py   activation/back-switching field random generators
  domains.py         domain-switching kernel (vectorised)
  mosfet.py          MOS charge / surface-potential / drain-current model
  fe_afe_fet.py      gate-voltage self-consistent solve + time stepping
  waveforms.py       input waveform construction (breakpoints_to_waveform, pulse_train)
  noise.py           device (read-current) noise
  simulate_rc.py     standard-format RC dataset generator (path B: no hardware needed)
"""
