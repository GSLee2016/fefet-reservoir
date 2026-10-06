"""Shared token grammar for the RC-track set-number fields.

``train_sets`` and ``test_sets`` accept a whole number, an inclusive
``"start-end"`` range string, or a list mixing both (e.g. ``["1-8", 10]``).
Used by :mod:`fefet_reservoir.config.loader` (shape check when the file is
read) and :mod:`fefet_reservoir.io.dataset` (parsing into 1-based set
numbers).
"""

from __future__ import annotations

import math
import re
from typing import Any

__all__ = ["parse_set_field", "SetFieldError"]

#: A bare integer, or a "start-end" range (inclusive), as text.
_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_INT_RE = re.compile(r"^\s*\d+\s*$")


class SetFieldError(ValueError):
    """``train_sets`` / ``test_sets`` does not match the token grammar.

    Callers that collect problems into a list (rather than raising
    immediately) should catch this and append ``str(exc)``.
    """


def parse_set_field(value: Any, field_name: str) -> tuple[int, ...]:
    """Parse ``train_sets`` / ``test_sets``: a list of ints, a range string
    like ``"1-8"``, or a list mixing both. Inclusive, sorted, de-duplicated.

    Raises :class:`SetFieldError` (never a bare ``ValueError``/``TypeError``)
    on anything that does not match the grammar.
    """
    items = value if isinstance(value, list) else [value]
    numbers: list[int] = []
    for token in items:
        numbers.extend(_parse_set_token(token, field_name))
    return tuple(sorted(set(numbers)))


def _parse_set_token(token: Any, field_name: str) -> tuple[int, ...]:
    if isinstance(token, bool):
        pass  # bool is a subclass of int; reject it explicitly below
    elif isinstance(token, (int, float)):
        # 3.0 (from YAML number coercion) is fine; 3.7 or nan is not -- a
        # silently truncated set number is an off-by-one split failure that
        # changes the score without any error.
        if isinstance(token, float) and (not math.isfinite(token) or token != int(token)):
            raise SetFieldError(
                f"dataset.{field_name}: {token!r} is not a whole set number. "
                f"Set numbers are whole numbers 1..S."
            )
        return (int(token),)
    elif isinstance(token, str):
        text = token.strip()
        m = _RANGE_RE.match(text)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                raise SetFieldError(
                    f"dataset.{field_name}: '{text}' has a start greater than "
                    f"its end. Use 'start-end' with start <= end (both inclusive)."
                )
            return tuple(range(lo, hi + 1))
        if _INT_RE.match(text):
            return (int(text),)
    raise SetFieldError(
        f"dataset.{field_name}: {token!r} is not a valid set number or range. "
        f"Use a whole number, or a range string like '1-8' (inclusive)."
    )
