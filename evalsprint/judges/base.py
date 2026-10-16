from __future__ import annotations

from ..types import Case, JudgeResult, TargetOutput


class JudgeConfigError(ValueError, TypeError):
    """A judge or check is configured wrongly. A ValueError for the CLI (a config error, exit 2 with a one-line
    message) and a TypeError for callers that construct judges directly and treat a missing option as one."""


def _string_set(value, what: str) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}  # `when_tags: safety` means the one tag, not the letters s, a, f, e, t, y
    if not isinstance(value, (list, tuple, set)) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"'{what}' must be a list of strings, got {value!r}")
    return set(value)


class Judge:
    """Base judge. Subclasses implement `evaluate`.

    Common config keys:
      name           unique name (shown in reports)
      when_tags      only apply to cases having at least one of these tags
      when_behavior  only apply to cases with one of these expected_behavior values
    """

    kind = "base"

    def __init__(self, name: str, when_tags: list[str] | None = None, when_behavior: list[str] | None = None):
        # No **kwargs catch-all: an unknown setting (`when_tag`, `sample`, `mod`) must fail loudly,
        # not silently run the judge on every case with the default behaviour.
        self.name = name
        self.when_tags = _string_set(when_tags, "when_tags")
        self.when_behavior = _string_set(when_behavior, "when_behavior")

    def applies(self, case: Case) -> bool:
        if self.when_tags and not (self.when_tags & set(case.tags)):
            return False
        if self.when_behavior and case.expected_behavior not in self.when_behavior:
            return False
        return True

    def evaluate(self, case: Case, output: TargetOutput) -> JudgeResult:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg: dict, **kw):
        return cls(**cfg, **kw)
