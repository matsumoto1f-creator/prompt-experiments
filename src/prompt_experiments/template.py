"""Prompt templating.

Two rules, both about failing at the right moment:

1. The TEMPLATE is versioned, never the filled prompt. Versioning filled prompts
   would create a new version per request and make the registry meaningless.
2. A missing variable is an error at serve time, not an empty string. A prompt
   silently rendered with a blank where the customer's name should be does not fail —
   it just gets quietly worse, and shows up weeks later as an unexplained drop in the
   metric with no deploy to blame.
"""

from __future__ import annotations

import re

PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class MissingVariables(ValueError):
    def __init__(self, missing: set[str]) -> None:
        self.missing = missing
        super().__init__(f"template needs variables that were not supplied: {sorted(missing)}")


def variables(template: str) -> set[str]:
    return set(PLACEHOLDER.findall(template))


def render(template: str, values: dict[str, str], *, strict: bool = True) -> str:
    """Fill a template. Raises `MissingVariables` unless `strict=False`."""
    needed = variables(template)
    missing = needed - set(values)
    if missing and strict:
        raise MissingVariables(missing)

    def _sub(match: re.Match[str]) -> str:
        return str(values.get(match.group(1), match.group(0)))

    return PLACEHOLDER.sub(_sub, template)
