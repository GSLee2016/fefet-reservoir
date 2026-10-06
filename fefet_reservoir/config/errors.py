"""One exception type, ConfigError, for both configuration and dataset problems.

All problems found in one pass are reported together, numbered; each says what
was found and what to do, and most end with a pointer into the documentation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

__all__ = ["ConfigError", "raise_problems"]


class ConfigError(ValueError):
    """Raised when a configuration or an input dataset cannot be accepted."""


def raise_problems(
    problems: Sequence[str],
    *,
    source: str | Path | None = None,
    guide: str | None = None,
) -> None:
    """Raise :class:`ConfigError` listing every problem found. No-op if empty.

    Parameters
    ----------
    problems:
        One message per problem.
    source:
        The file the problems were found in, if there is one.
    guide:
        Where to read more, e.g. ``"README.md, 'Configuration files'"``.
    """
    if not problems:
        return

    where = f" in {source}" if source is not None else ""
    count = len(problems)
    noun = "problem" if count == 1 else "problems"
    body = "\n".join(f"  {i}. {text}" for i, text in enumerate(problems, 1))
    tail = f"\n\nMore detail: {guide}" if guide else ""
    raise ConfigError(f"Found {count} {noun}{where}:\n{body}{tail}")
