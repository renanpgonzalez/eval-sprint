"""Suite configuration (eval.yaml).

Supports ${VAR} and ${VAR:-default} environment expansion, and CLI overrides
like `--set target.params.prompt=v2` (values parsed as YAML).

Everything is validated when the file is loaded: an unknown key or a wrong type
is a configuration error with a "did you mean" hint, never a silently ignored
typo (a misspelled `gate:` or `repeat:` would otherwise switch the feature off).
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .gate_defaults import DEFAULT_GATE, GATE_KEYS
from .types import suggest

__all__ = ["DEFAULT_GATE", "GATE_KEYS", "MAX_REPEAT", "SuiteConfig", "apply_overrides", "load_gate", "merge_gate", "parse_repeat", "redact_secrets"]

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# Whole-value ${VAR} typing is deliberately narrow. YAML would turn `0123` into 83, `no` into False,
# `12:30` into 750 and `abc #def` into `abc`; only unambiguous numbers and booleans are typed.
_INT_RE = re.compile(r"(0|[1-9][0-9]{0,14})")  # no leading zeros; longer digit strings are IDs, not numbers
_FLOAT_RE = re.compile(r"[0-9]+\.[0-9]+")
_TOP_LEVEL_KEYS = ("suite", "dataset", "target", "judges", "pass_rule", "concurrency", "gate", "pricing", "runs_dir", "repeat", "behaviors", "tag_groups", "target_retries")
_TARGET_KEYS = ("callable", "params")
_REPEAT_KEYS = ("n", "per_tag", "tags")  # `tags` is the cycle-3 spelling: {n: 5, tags: [safety]} == {n: 1, per_tag: {safety: 5}}
MAX_REPEAT = 50
DEFAULT_TARGET_RETRIES = 1
MAX_TARGET_RETRIES = 5  # every retry is a paid call: more than a few is a broken target, not a transient crash


def _typed(text: str) -> Any:
    if _INT_RE.fullmatch(text):
        return int(text)
    if _FLOAT_RE.fullmatch(text):
        return float(text)
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else "")
        expanded = _ENV_RE.sub(repl, value)
        if expanded != value and _ENV_RE.fullmatch(value):
            return _typed(expanded)  # whole value was a variable: strict ints, floats and booleans only
        return expanded
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def apply_overrides(data: dict, overrides: list[str] | None) -> dict:
    data = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw) if raw != "" else ""
        node = data
        parts = key.split(".")

        def step(container, part):
            if isinstance(container, list):
                try:
                    return container[int(part)]
                except (ValueError, IndexError):
                    raise ValueError(f"override {key!r}: {part!r} is not a valid position (the list has {len(container)} item(s), counting from 0)") from None
            if not isinstance(container.get(part), (dict, list)):
                container[part] = {}  # missing, or an empty YAML key (`gate:` -> None)
            return container[part]

        for part in parts[:-1]:
            node = step(node, part)
        last = parts[-1]
        if isinstance(node, list):
            try:
                node[int(last)] = value
            except (ValueError, IndexError):
                raise ValueError(f"override {key!r}: {last!r} is not a valid position (the list has {len(node)} item(s), counting from 0)") from None
        else:
            node[last] = value
    return data


# ------------------------------------------------------------------------- secrets
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SECRET_WORDS = {
    "secret", "secrets", "password", "passwd", "pwd", "passphrase", "token", "credential", "credentials", "authorization",
    "bearer", "jwt", "cookie", "cookies", "session", "dsn", "apikey", "privatekey", "signingkey", "accesskey", "secretkey",
}
_SECRET_PAIRS = {("api", "key"), ("access", "key"), ("secret", "key"), ("private", "key"), ("signing", "key"), ("auth", "key"), ("connection", "string")}
_NOT_A_SECRET_LAST = {"env", "var", "url", "uri", "name", "file", "path"}  # `api_key_env: MY_KEY` names a variable; `token_url` is an endpoint
_BENIGN_KEY_PREFIX = {"cache", "sort", "primary", "foreign", "partition", "group", "order", "lookup", "hash", "shard", "idempotency"}
_URL_USERINFO = re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*://)[^/\s@]*@")
_TOKEN_SHAPES = re.compile(
    r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=\-]{8,}"
    r"|\bsk-[A-Za-z0-9_\-]{16,}|\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{20,}|\bxox[abprs]-[A-Za-z0-9\-]{10,}"
    r"|\beyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]*"
)


def _looks_secret(key: str) -> bool:
    words = [w.lower() for w in re.split(r"[^A-Za-z0-9]+", _CAMEL.sub("_", key)) if w]
    if not words or words[-1] in _NOT_A_SECRET_LAST:
        return False
    if any(w in _SECRET_WORDS for w in words):
        return True
    if any(pair in _SECRET_PAIRS for pair in zip(words, words[1:])):
        return True
    if words[-1] == "key":  # `key`, `openai_key`, `stripe_key`; but not `cache_key` or `sort_key`
        return len(words) == 1 or words[-2] not in _BENIGN_KEY_PREFIX
    return False


def _mask_text(text: str) -> str:
    text = _URL_USERINFO.sub(r"\1***@", text)
    return _TOKEN_SHAPES.sub(lambda m: f"{m.group(1)} ***" if m.group(1) else "***", text)


def redact_secrets(value: Any) -> Any:
    """Copy of `value` with secret-looking entries masked as `***`, safe to store in summary.json (which is
    committed as the baseline) or to print. A value is masked when its key names a secret (api_key, secret_key,
    private_key, token, password, bearer, jwt, cookie, session_id, dsn, connection_string, credentials, ...;
    but not `max_tokens` or `api_key_env`), and secrets are removed from any string: `user:pass@` in URLs,
    `Bearer ...` headers and well-known token shapes."""
    if isinstance(value, dict):
        return {
            k: ("***" if isinstance(k, str) and _looks_secret(k) and v not in (None, "") else redact_secrets(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(v) for v in value]
    if isinstance(value, str):
        return _mask_text(value)
    return value


# ---------------------------------------------------------------------------- gate
def merge_gate(user_gate: Any, where: str = "gate") -> dict:
    """DEFAULT_GATE overlaid with `user_gate`; unknown keys are an error (with a suggestion)."""
    if user_gate is None:
        user_gate = {}
    if not isinstance(user_gate, dict):
        raise ValueError(f"'{where}' must be a mapping of settings, got {type(user_gate).__name__}")
    for key in user_gate:
        if key not in GATE_KEYS:
            raise ValueError(f"unknown {where} setting {key!r}{suggest(key, GATE_KEYS)} (valid: {', '.join(GATE_KEYS)})")
    gate = copy.deepcopy(DEFAULT_GATE)
    gate.update(copy.deepcopy(user_gate))
    gate["critical_tags"] = _critical_tags(gate.get("critical_tags"), where)
    return gate


def _critical_tags(value: Any, where: str) -> list[str]:
    """`critical_tags` as a list of tags: `critical_tags: safety` means the one tag (not the letters s, a, f, e, t, y),
    and `null` means none. Anything that is not a string or a list of non-empty strings is a config error."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(t, str) and t.strip() for t in value):
        raise ValueError(f"{where}.critical_tags must be a tag or a list of tags (e.g. [safety, billing]), got {value!r}")
    return list(value)


def load_gate(path: str | Path) -> dict:
    """The complete gate defined by the `gate:` block of one YAML file (an eval.yaml, or a file with only `gate:`).

    Nothing else is consulted: settings the file does not mention fall back to DEFAULT_GATE, never to a run's
    own summary. CI passes the base branch's copy so a pull request cannot loosen its own gate.
    """
    with open(path, encoding="utf-8-sig") as fh:  # -sig: an editor may have added a BOM
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping with a 'gate:' block")
    return merge_gate(_expand_env(data).get("gate"))


# -------------------------------------------------------------------------- config
def _as_str_list(value: Any, what: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"'{what}' must be a non-empty list of strings, got {value!r}")
    return list(value)


def _int_setting(value: Any, what: str, minimum: int = 1, maximum: int | None = None) -> int:
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"'{what}' must be a whole number, got {value!r}")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"'{what}' must be between {minimum} and {maximum}, got {value}" if maximum else f"'{what}' must be at least {minimum}, got {value}")
    return value


def _check_keys(data: dict, valid: tuple[str, ...], where: str) -> None:
    for key in data:
        if key not in valid:
            raise ValueError(f"unknown {where} setting {key!r}{suggest(key, valid)} (valid: {', '.join(valid)})")


def parse_repeat(raw: Any, where: str = "repeat") -> dict | None:
    """The repeat protocol of a config value (C2, C17): `{"n": int, "per_tag": {tag: attempts}}`, or None when every case runs once.

    `repeat: {n: 3, per_tag: {safety: 5}}` runs every case 3 times and the cases tagged `safety` 5 times: a case runs the LARGEST
    `per_tag` count among its tags, else `n`. `repeat: 3` is `{n: 3}`. `per_tag` alone (`{per_tag: {safety: 5}}`) means `n: 1`.
    The old spelling `{n: 5, tags: [safety]}` (n attempts on the listed tags only) is read as `{n: 1, per_tag: {safety: 5}}`; naming both
    `tags` and `per_tag` is an error. `per_tag` keys come back sorted.
    """
    if raw is None or raw is False:
        return None
    if isinstance(raw, int) and not isinstance(raw, bool):
        raw = {"n": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"'{where}' must look like {{n: 3, per_tag: {{safety: 5}}}}, got {raw!r}")
    _check_keys(raw, _REPEAT_KEYS, where)
    if raw.get("tags") is not None and raw.get("per_tag") is not None:
        raise ValueError(f"'{where}' has both 'tags' (the old spelling) and 'per_tag': keep per_tag, e.g. {where}: {{n: 3, per_tag: {{safety: 5}}}}")
    if "n" not in raw and "per_tag" not in raw:
        raise ValueError(f"'{where}' needs 'n' (how many times to run each case), e.g. {where}: {{n: 3}}")
    n = _int_setting(raw["n"], f"{where}.n", 1, MAX_REPEAT) if "n" in raw else 1
    per_tag: dict[str, int] = {}
    if raw.get("per_tag") is not None:
        given = raw["per_tag"]
        if not isinstance(given, dict):
            raise ValueError(f"'{where}.per_tag' must map a tag to its number of attempts, e.g. per_tag: {{safety: 5}}, got {given!r}")
        for tag, attempts in given.items():
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError(f"'{where}.per_tag' keys must be tags (non-empty strings), got {tag!r}")
            per_tag[tag] = _int_setting(attempts, f"{where}.per_tag.{tag}", 1, MAX_REPEAT)
    elif raw.get("tags") is not None:
        tags = _as_str_list(raw["tags"], f"{where}.tags")
        if n > 1:  # n attempts on the listed tags, every other case once
            per_tag = {tag: n for tag in tags}
            n = 1
    per_tag = dict(sorted(per_tag.items()))
    if n <= 1 and not any(attempts > 1 for attempts in per_tag.values()):
        return None  # one attempt everywhere: no repetition
    return {"n": n, "per_tag": per_tag}


_parse_repeat = parse_repeat  # kept under its old name


def _parse_tag_groups(raw: Any) -> dict[str, list[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("'tag_groups' must map a group name to a list of tags, e.g. tag_groups: {type: [answerable, safety]}")
    groups: dict[str, list[str]] = {}
    for name, tags in raw.items():
        groups[str(name)] = _as_str_list(tags, f"tag_groups.{name}")
    return groups


@dataclass
class SuiteConfig:
    suite: str
    dataset: Path
    target: dict
    judges: list[dict]
    base_dir: Path
    pass_rule: str | list[str] = "all"
    concurrency: int = 4
    gate: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_GATE))
    pricing: dict = field(default_factory=dict)
    runs_dir: Path | None = None
    repeat: dict | None = None  # {"n": 3, "per_tag": {"safety": 5}}; None = every case runs once
    target_retries: int = DEFAULT_TARGET_RETRIES  # extra calls of a target attempt that errored, inside the run (0 = off)
    behaviors: list[str] | None = None  # allowed expected_behavior values; None = not checked
    tag_groups: dict[str, list[str]] = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "SuiteConfig":
        path = Path(path).resolve()
        with open(path, encoding="utf-8-sig") as fh:  # -sig: an editor may have added a BOM
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path.name}: expected a YAML mapping (suite:, dataset:, target:, ...)")
        data = apply_overrides(data, overrides)
        data = _expand_env(data)
        return cls.from_dict(data, base_dir=path.parent)

    @classmethod
    def from_dict(cls, data: dict, base_dir: str | Path = ".") -> "SuiteConfig":
        base = Path(base_dir).resolve()
        for req in ("suite", "dataset", "target"):
            if req not in data:
                raise ValueError(f"eval config is missing '{req}'")
        for key in data:
            if isinstance(key, str) and key.startswith(("_", "x-", "x_")):
                continue  # free-form: YAML anchors (`_common: &c`) and vendor extensions
            if key not in _TOP_LEVEL_KEYS:
                raise ValueError(f"unknown setting {key!r} in the eval config{suggest(key, _TOP_LEVEL_KEYS)} (valid: {', '.join(_TOP_LEVEL_KEYS)})")

        target = data["target"]
        if not isinstance(target, dict):
            raise ValueError("'target' must be a mapping with 'callable: module:function'")
        target = dict(target)
        _check_keys(target, _TARGET_KEYS, "target")
        if not target.get("callable") or not isinstance(target["callable"], str):
            raise ValueError("target needs 'callable: module:function'")
        params = target.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"'target.params' must be a mapping, got {type(params).__name__}")
        target["params"] = params

        judges = data.get("judges")
        if judges is None:
            judges = []
        if not isinstance(judges, list):
            raise ValueError(f"'judges' must be a list, got {type(judges).__name__}")

        pass_rule = data.get("pass_rule", "all")
        if not (isinstance(pass_rule, str) or (isinstance(pass_rule, list) and pass_rule and all(isinstance(n, str) for n in pass_rule))):
            raise ValueError(f"'pass_rule' must be 'all', 'any', a judge name or a list of judge names, got {pass_rule!r}")

        pricing = data.get("pricing") or {}
        if not isinstance(pricing, dict):
            raise ValueError("'pricing' must map a model name to its prices")
        runs_dir = data.get("runs_dir")
        behaviors = data.get("behaviors")
        return cls(
            suite=str(data["suite"]),
            dataset=(base / str(data["dataset"])).resolve(),
            target=target,
            judges=list(judges),
            base_dir=base,
            pass_rule=pass_rule,
            concurrency=_int_setting(data.get("concurrency", 4), "concurrency", 1),
            gate=merge_gate(data.get("gate")),
            pricing=dict(pricing),
            runs_dir=(base / runs_dir).resolve() if runs_dir else (base / "runs").resolve(),
            repeat=parse_repeat(data.get("repeat")),
            target_retries=_int_setting(data.get("target_retries", DEFAULT_TARGET_RETRIES), "target_retries", 0, MAX_TARGET_RETRIES),
            behaviors=_as_str_list(behaviors, "behaviors") if behaviors is not None else None,
            tag_groups=_parse_tag_groups(data.get("tag_groups")),
            raw=data,
        )
