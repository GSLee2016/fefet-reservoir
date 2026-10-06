"""Dataset loading and standard-format validation.

Accepts device output from either a real measurement (path A) or the bundled
simulator (path B), and refuses input whose structure cannot support a
meaningful score.
"""

from .dataset import (
    DatasetSpec,
    RCDataset,
    ReadoutWarning,
    ScaleWarning,
    load_rc_dataset,
    suggest_dataset_block,
)

__all__ = [
    "DatasetSpec",
    "RCDataset",
    "load_rc_dataset",
    "suggest_dataset_block",
    "ScaleWarning",
    "ReadoutWarning",
]
