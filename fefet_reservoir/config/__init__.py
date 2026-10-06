"""Configuration loading, derived-value computation and section validation.

All experiment conditions live in YAML files rather than in the code. Most
values that determine a score are required; dataset.max_delay defaults to 20.

Public entry points::

    from fefet_reservoir.config import load_config, ConfigError

    cfg = load_config("configs/example_simulated.yaml")
"""

from .errors import ConfigError, raise_problems
from .loader import (
    Config,
    coerce_numbers,
    compute_derived,
    get_by_path,
    load_config,
    require_section,
    validate,
)

__all__ = [
    "ConfigError",
    "raise_problems",
    "Config",
    "load_config",
    "validate",
    "compute_derived",
    "coerce_numbers",
    "get_by_path",
    "require_section",
]
