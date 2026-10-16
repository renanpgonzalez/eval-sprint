import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from evalsprint.config import DEFAULT_GATE, GATE_KEYS, SuiteConfig, apply_overrides, load_gate, merge_gate


def write_yaml(directory: Path, data, name: str = "eval.yaml") -> Path:
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


BASE = {
    "suite": "demo",
    "dataset": "cases.jsonl",
    "target": {"callable": "t.py:fn", "params": {"model": "m1", "prompt": "v1"}},
    "judges": [{"type": "contains", "name": "c", "values": ["x"]}, {"type": "max_chars", "name": "len", "value": 100}],
}


class EnvExpansionTests(unittest.TestCase):
    def load(self, target_params, env=None):
        data = copy.deepcopy(BASE)
        data["target"]["params"] = target_params
        with tempfile.TemporaryDirectory() as tmp:
            path = write_yaml(Path(tmp), data)
            with mock.patch.dict(os.environ, env or {}):
                return SuiteConfig.load(path).target["params"]

    def test_plain_variable(self):
        self.assertEqual(self.load({"k": "${ES_TEST_A}"}, {"ES_TEST_A": "hello"})["k"], "hello")

    def test_variable_embedded_in_a_string(self):
        params = self.load({"k": "pre-${ES_TEST_A}-${ES_TEST_B:-bee}-post"}, {"ES_TEST_A": "a"})
        self.assertEqual(params["k"], "pre-a-bee-post")

    def test_default_used_when_unset_and_ignored_when_set(self):
        self.assertEqual(self.load({"k": "${ES_TEST_UNSET:-fallback}"})["k"], "fallback")
        self.assertEqual(self.load({"k": "${ES_TEST_A:-fallback}"}, {"ES_TEST_A": "real"})["k"], "real")

    def test_unset_without_default_is_empty_string(self):
        self.assertEqual(self.load({"k": "${ES_TEST_UNSET}"})["k"], "")
        self.assertEqual(self.load({"k": "x-${ES_TEST_UNSET}-y"})["k"], "x--y")

    def test_empty_default(self):
        self.assertEqual(self.load({"k": "${ES_TEST_UNSET:-}"})["k"], "")

    def test_whole_value_variables_are_typed(self):
        params = self.load(
            {"n": "${ES_N}", "f": "${ES_F}", "b": "${ES_B}", "s": "${ES_S}", "from_default": "${ES_UNSET:-8}"},
            {"ES_N": "4", "ES_F": "0.25", "ES_B": "true", "ES_S": "claude-haiku-4-5"},
        )
        self.assertEqual(params["n"], 4)
        self.assertIsInstance(params["n"], int)
        self.assertEqual(params["f"], 0.25)
        self.assertIs(params["b"], True)
        self.assertEqual(params["s"], "claude-haiku-4-5")
        self.assertEqual(params["from_default"], 8)

    def test_embedded_variables_stay_strings(self):
        self.assertEqual(self.load({"k": "n=${ES_N}"}, {"ES_N": "4"})["k"], "n=4")

    def test_unparseable_whole_value_falls_back_to_string(self):
        self.assertEqual(self.load({"k": "${ES_X}"}, {"ES_X": "a: [unclosed"})["k"], "a: [unclosed")

    def test_expansion_reaches_nested_dicts_and_lists(self):
        params = self.load({"deep": {"list": ["${ES_A}", {"k": "${ES_B:-z}"}]}}, {"ES_A": "1"})
        self.assertEqual(params["deep"]["list"][0], 1)
        self.assertEqual(params["deep"]["list"][1]["k"], "z")

    def test_non_string_values_untouched(self):
        params = self.load({"n": 3, "flag": False, "none": None, "lst": [1, 2]})
        self.assertEqual(params, {"n": 3, "flag": False, "none": None, "lst": [1, 2]})

    def test_switching_provider_via_env(self):
        data = copy.deepcopy(BASE)
        data["judges"] = [{"type": "llm", "name": "q", "provider": "${LLM_PROVIDER:-mock}", "model": "${JUDGE_MODEL:-claude-haiku-4-5}"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_yaml(Path(tmp), data)
            self.assertEqual(SuiteConfig.load(path).judges[0]["provider"], "mock")
            with mock.patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}):
                self.assertEqual(SuiteConfig.load(path).judges[0]["provider"], "anthropic")


class OverrideTests(unittest.TestCase):
    def test_simple_and_nested_existing_keys(self):
        out = apply_overrides(BASE, ["suite=other", "target.params.prompt=v2"])
        self.assertEqual(out["suite"], "other")
        self.assertEqual(out["target"]["params"]["prompt"], "v2")
        self.assertEqual(out["target"]["params"]["model"], "m1")  # sibling untouched

    def test_new_nested_keys_are_created(self):
        out = apply_overrides(BASE, ["gate.max_pass_rate_drop=0.1", "target.params.extra.deeper.key=1"])
        self.assertEqual(out["gate"], {"max_pass_rate_drop": 0.1})
        self.assertEqual(out["target"]["params"]["extra"], {"deeper": {"key": 1}})

    def test_list_index(self):
        out = apply_overrides(BASE, ["judges.1.value=250", "judges.0.name=renamed"])
        self.assertEqual(out["judges"][1]["value"], 250)
        self.assertEqual(out["judges"][0]["name"], "renamed")
        self.assertEqual(len(out["judges"]), 2)

    def test_list_element_replaced_wholesale(self):
        out = apply_overrides({"pass_rule": ["a", "b"]}, ["pass_rule.1=c"])
        self.assertEqual(out["pass_rule"], ["a", "c"])

    def test_values_are_parsed_as_yaml(self):
        out = apply_overrides({}, ["a=3", "b=0.5", "c=true", "d=null", "e=[x, y]", "f={k: v}", "g=plain text", "h=v2"])
        self.assertEqual(out, {"a": 3, "b": 0.5, "c": True, "d": None, "e": ["x", "y"], "f": {"k": "v"}, "g": "plain text", "h": "v2"})

    def test_empty_value_is_empty_string(self):
        self.assertEqual(apply_overrides({}, ["a="]), {"a": ""})

    def test_only_first_equals_splits(self):
        out = apply_overrides({}, ["target.params.url=http://x.test/?a=b&c=d"])
        self.assertEqual(out["target"]["params"]["url"], "http://x.test/?a=b&c=d")

    def test_later_override_wins(self):
        self.assertEqual(apply_overrides({}, ["a=1", "a=2"]), {"a": 2})

    def test_missing_equals_raises(self):
        with self.assertRaises(ValueError) as ctx:
            apply_overrides(BASE, ["suite"])
        self.assertIn("key.path=value", str(ctx.exception))

    def test_input_is_not_mutated(self):
        original = copy.deepcopy(BASE)
        apply_overrides(BASE, ["suite=x", "target.params.prompt=y", "judges.0.name=z"])
        self.assertEqual(BASE, original)

    def test_none_or_empty_overrides(self):
        self.assertEqual(apply_overrides(BASE, None), BASE)
        self.assertEqual(apply_overrides(BASE, []), BASE)

    def test_empty_yaml_section_can_be_overridden(self):
        # regression: `gate:` with nothing under it loads as None; --set gate.x=y used to crash
        out = apply_overrides({"gate": None, "pricing": None}, ["gate.max_pass_rate_drop=0.1", "pricing.my-model.input=2"])
        self.assertEqual(out["gate"], {"max_pass_rate_drop": 0.1})
        self.assertEqual(out["pricing"], {"my-model": {"input": 2}})

    def test_overrides_are_applied_before_env_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_yaml(Path(tmp), BASE)
            with mock.patch.dict(os.environ, {"ES_PROMPT": "from-env"}):
                cfg = SuiteConfig.load(path, ["target.params.prompt=${ES_PROMPT}"])
        self.assertEqual(cfg.target["params"]["prompt"], "from-env")

    def test_load_with_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_yaml(Path(tmp), BASE)
            cfg = SuiteConfig.load(path, ["concurrency=2", "gate.max_pass_rate_drop=0.1", "target.params.prompt=v2", "pass_rule=[c, len]"])
        self.assertEqual(cfg.concurrency, 2)
        self.assertEqual(cfg.gate["max_pass_rate_drop"], 0.1)
        self.assertEqual(cfg.target["params"]["prompt"], "v2")
        self.assertEqual(cfg.pass_rule, ["c", "len"])


class RequiredKeyTests(unittest.TestCase):
    def test_missing_required_top_level_keys(self):
        for key in ("suite", "dataset", "target"):
            with self.subTest(missing=key):
                data = {k: v for k, v in BASE.items() if k != key}
                with self.assertRaises(ValueError) as ctx:
                    SuiteConfig.from_dict(data)
                self.assertIn(key, str(ctx.exception))

    def test_target_needs_callable(self):
        data = dict(BASE, target={"params": {}})
        with self.assertRaises(ValueError) as ctx:
            SuiteConfig.from_dict(data)
        self.assertIn("callable", str(ctx.exception))

    def test_empty_file_reports_missing_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_yaml(Path(tmp), "")
            with self.assertRaises(ValueError) as ctx:
                SuiteConfig.load(path)
        self.assertIn("suite", str(ctx.exception))

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            SuiteConfig.load("/nonexistent/eval.yaml")


class PathAndDefaultTests(unittest.TestCase):
    def test_paths_resolve_relative_to_the_config_file_not_cwd(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as elsewhere:
            tmp = Path(tmp).resolve()
            data = dict(BASE, dataset="../data/cases.jsonl", runs_dir="out/runs")
            path = write_yaml(tmp / "evals" / "sub", data)
            old = os.getcwd()
            os.chdir(elsewhere)
            try:
                cfg = SuiteConfig.load(path)
            finally:
                os.chdir(old)
            self.assertEqual(cfg.base_dir, tmp / "evals" / "sub")
            self.assertEqual(cfg.dataset, tmp / "evals" / "data" / "cases.jsonl")
            self.assertEqual(cfg.runs_dir, tmp / "evals" / "sub" / "out" / "runs")

    def test_runs_dir_defaults_next_to_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SuiteConfig.load(write_yaml(Path(tmp), BASE))
            self.assertEqual(cfg.runs_dir, Path(tmp).resolve() / "runs")

    def test_absolute_dataset_path_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = dict(BASE, dataset=str(Path(tmp) / "abs.jsonl"))
            cfg = SuiteConfig.load(write_yaml(Path(tmp), data))
            self.assertEqual(cfg.dataset, (Path(tmp) / "abs.jsonl").resolve())

    def test_defaults(self):
        cfg = SuiteConfig.from_dict({"suite": "s", "dataset": "d.jsonl", "target": {"callable": "a:b"}}, base_dir="/tmp")
        self.assertEqual(cfg.pass_rule, "all")
        self.assertEqual(cfg.concurrency, 4)
        self.assertEqual(cfg.judges, [])
        self.assertEqual(cfg.pricing, {})
        self.assertEqual(cfg.target["params"], {})
        self.assertEqual(cfg.gate, DEFAULT_GATE)

    def test_gate_is_merged_and_default_gate_is_not_mutated(self):
        snapshot = copy.deepcopy(DEFAULT_GATE)
        data = dict(BASE, gate={"max_pass_rate_drop": 0.1, "critical_tags": ["refund"], "min_pass_rate": 0.8})
        cfg = SuiteConfig.from_dict(data, base_dir="/tmp")
        self.assertEqual(cfg.gate["max_pass_rate_drop"], 0.1)
        self.assertEqual(cfg.gate["critical_tags"], ["refund"])
        self.assertEqual(cfg.gate["min_pass_rate"], 0.8)
        self.assertEqual(cfg.gate["alpha"], DEFAULT_GATE["alpha"])  # untouched keys keep defaults
        self.assertEqual(DEFAULT_GATE, snapshot)
        self.assertEqual(SuiteConfig.from_dict(BASE, base_dir="/tmp").gate, snapshot)

    def test_gate_null_section_uses_defaults(self):
        cfg = SuiteConfig.from_dict(dict(BASE, gate=None), base_dir="/tmp")
        self.assertEqual(cfg.gate, DEFAULT_GATE)

    def test_default_gate_keys_match_contract(self):
        from evalsprint import gate_defaults

        # the source of truth is gate_defaults.GATE_KEYS, so a key added there cannot make this test drift again;
        # the keys below are the ones the contract promises and must never disappear
        promised = {
            "max_pass_rate_drop", "alpha", "require_significance", "min_pass_rate", "critical_tags", "max_cost_increase_pct",
            "max_p95_latency_increase_pct", "max_error_rate", "max_removed_cases", "max_changed_cases", "min_judge_kappa", "require_calibration",
            "allow_judge_changes",
        }
        self.assertEqual(set(DEFAULT_GATE), set(gate_defaults.GATE_KEYS))
        self.assertLessEqual(promised, set(DEFAULT_GATE))
        self.assertEqual(DEFAULT_GATE["max_pass_rate_drop"], "auto")
        self.assertIs(DEFAULT_GATE["allow_judge_changes"], False)  # a changed judge blocks the merge unless the PR says otherwise
        self.assertEqual(tuple(DEFAULT_GATE), GATE_KEYS)

    def test_default_gate_is_re_exported_from_gate_defaults(self):
        from evalsprint import gate_defaults

        self.assertIs(DEFAULT_GATE, gate_defaults.DEFAULT_GATE)

    def test_each_config_gets_its_own_copy_of_mutable_gate_defaults(self):
        a = SuiteConfig.from_dict(BASE, base_dir="/tmp")
        a.gate["critical_tags"].append("leak")
        self.assertEqual(SuiteConfig.from_dict(BASE, base_dir="/tmp").gate["critical_tags"], [])
        self.assertEqual(DEFAULT_GATE["critical_tags"], [])

    def test_pricing_and_pass_rule_and_raw(self):
        data = dict(BASE, pricing={"m": {"input": 1}}, pass_rule=["c"], concurrency="3")
        cfg = SuiteConfig.from_dict(data, base_dir="/tmp")
        self.assertEqual(cfg.pricing, {"m": {"input": 1}})
        self.assertEqual(cfg.pass_rule, ["c"])
        self.assertEqual(cfg.concurrency, 3)
        self.assertEqual(cfg.raw["suite"], "demo")

    def test_suite_name_coerced_to_string(self):
        cfg = SuiteConfig.from_dict(dict(BASE, suite=2026), base_dir="/tmp")
        self.assertEqual(cfg.suite, "2026")


class StrictEnvTypingTests(unittest.TestCase):
    """Whole-value ${VAR} typing: strict ints (no leading zeros), floats (\\d+\\.\\d+) and true/false; everything else stays a string."""

    def expand(self, value):
        from evalsprint.config import _expand_env

        with mock.patch.dict(os.environ, {"ES_V": value}):
            return _expand_env("${ES_V}")

    def test_typed_values(self):
        for text, expected in (("4", 4), ("0", 0), ("8", 8), ("0.25", 0.25), ("10.5", 10.5), ("true", True), ("false", False), ("True", True), ("FALSE", False)):
            with self.subTest(text=text):
                out = self.expand(text)
                self.assertEqual(out, expected)
                self.assertIs(type(out), type(expected))

    def test_everything_else_stays_a_string(self):
        for text in (
            "0123", "007", "no", "yes", "on", "off", "null", "~", "2026-09-29", "1_000", "0x1F", "0o17", "1e3", "12:30", "+1", "-5", ".5", "5.",
            "abc #def", "key: value", "- a", "[1, 2]", "{a: 1}", "*alias", "1.2.3", "a: [unclosed", "1234567890123456789012",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.expand(text), text)
                self.assertIsInstance(self.expand(text), str)

    def test_empty_stays_empty(self):
        self.assertEqual(self.expand(""), "")

    def test_defaults_are_typed_the_same_way(self):
        from evalsprint.config import _expand_env

        self.assertEqual(_expand_env("${ES_UNSET_X:-8}"), 8)
        self.assertEqual(_expand_env("${ES_UNSET_X:-0123}"), "0123")
        self.assertIs(_expand_env("${ES_UNSET_X:-false}"), False)
        self.assertEqual(_expand_env("${ES_UNSET_X:-claude-sonnet-5-5}"), "claude-sonnet-5-5")

    def test_embedded_variables_are_never_typed(self):
        from evalsprint.config import _expand_env

        with mock.patch.dict(os.environ, {"ES_V": "4"}):
            self.assertEqual(_expand_env("n=${ES_V}"), "n=4")
            self.assertEqual(_expand_env("${ES_V}${ES_V}"), "44")


class OverrideErrorTests(unittest.TestCase):
    def test_list_position_out_of_range_is_a_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            apply_overrides(BASE, ["judges.9.value=1"])
        self.assertIn("judges.9.value", str(ctx.exception))
        self.assertIn("2 item", str(ctx.exception))

    def test_non_numeric_position(self):
        with self.assertRaises(ValueError):
            apply_overrides(BASE, ["judges.first.value=1"])
        with self.assertRaises(ValueError):
            apply_overrides(BASE, ["judges.5=1"])


class ConfigValidationTests(unittest.TestCase):
    def load(self, **extra):
        return SuiteConfig.from_dict({**BASE, **extra}, base_dir="/tmp")

    def assertConfigError(self, contains, **extra):
        with self.assertRaises(ValueError) as ctx:
            self.load(**extra)
        self.assertIn(contains, str(ctx.exception))
        return str(ctx.exception)

    def test_unknown_top_level_key_suggests_the_right_one(self):
        message = self.assertConfigError("unknown setting 'gates'", gates={"max_pass_rate_drop": 0.1})
        self.assertIn("did you mean 'gate'", message)
        self.assertIn("did you mean 'repeat'", self.assertConfigError("unknown setting 'repeats'", repeats=3))
        self.assertIn("did you mean 'concurrency'", self.assertConfigError("'concurency'", concurency=2))

    def test_free_form_keys_are_allowed_for_yaml_anchors(self):
        self.load(_common={"a": 1}, **{"x-notes": "hi", "x_notes": "hi"})

    def test_unknown_gate_key_is_an_error_at_load_time(self):
        message = self.assertConfigError("unknown gate setting 'max_pass_rate_dop'", gate={"max_pass_rate_dop": 0.1})
        self.assertIn("did you mean 'max_pass_rate_drop'", message)

    def test_unknown_target_key(self):
        self.assertConfigError("unknown target setting 'param'", target={"callable": "a:b", "param": {}})

    def test_shape_errors(self):
        self.assertConfigError("'judges' must be a list", judges={"type": "contains"})
        self.assertConfigError("'target.params' must be a mapping", target={"callable": "a:b", "params": [1]})
        self.assertConfigError("'target' must be a mapping", target="a:b")
        self.assertConfigError("'pass_rule'", pass_rule=3)
        self.assertConfigError("'concurrency' must be a whole number", concurrency="many")
        self.assertConfigError("'pricing'", pricing=["x"])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                SuiteConfig.load(write_yaml(Path(tmp), "- just\n- a list\n"))
            self.assertIn("YAML mapping", str(ctx.exception))

    def test_concurrency_below_one_is_a_config_error(self):
        for value in (0, -1, "0", "-3"):
            with self.subTest(concurrency=value):
                self.assertConfigError("'concurrency' must be at least 1", concurrency=value)
        self.assertEqual(self.load(concurrency=1).concurrency, 1)
        self.assertEqual(self.load(concurrency="8").concurrency, 8)
        self.assertEqual(self.load().concurrency, 4)  # the default

    def test_concurrency_from_the_command_line_is_checked_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_yaml(Path(tmp), BASE)
            for override in ("concurrency=0", "concurrency=-2"):
                with self.subTest(override=override), self.assertRaises(ValueError) as ctx:
                    SuiteConfig.load(path, [override])
                self.assertIn("at least 1", str(ctx.exception))

    def test_a_single_critical_tag_written_as_a_string_is_a_one_item_list(self):
        for gate_value, expected in (("safety", ["safety"]), (["safety", "billing"], ["safety", "billing"]), (None, []), ([], [])):
            with self.subTest(critical_tags=gate_value):
                self.assertEqual(self.load(gate={"critical_tags": gate_value}).gate["critical_tags"], expected)
        self.assertEqual(self.load().gate["critical_tags"], [])  # unset: the default

    def test_critical_tags_of_the_wrong_type_are_errors(self):
        for bad in (5, {"a": 1}, ["ok", 3], [""], ["  "], True):
            with self.subTest(critical_tags=bad):
                message = self.assertConfigError("gate.critical_tags must be a tag or a list of tags", gate={"critical_tags": bad})
                self.assertIn(repr(bad), message)

    def test_critical_tags_string_via_set_and_gate_file(self):
        from evalsprint.config import load_gate, merge_gate

        with tempfile.TemporaryDirectory() as tmp:
            path = write_yaml(Path(tmp), BASE)
            self.assertEqual(SuiteConfig.load(path, ["gate.critical_tags=safety"]).gate["critical_tags"], ["safety"])
            self.assertEqual(load_gate(write_yaml(Path(tmp), {"gate": {"critical_tags": "safety"}}, "gate.yaml"))["critical_tags"], ["safety"])
        self.assertEqual(merge_gate({"critical_tags": "billing"})["critical_tags"], ["billing"])

    def test_a_bom_at_the_start_of_the_yaml_files_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.yaml"
            path.write_bytes(b"\xef\xbb\xbf" + yaml.safe_dump(dict(BASE, gate={"critical_tags": ["x"]})).replace("\n", "\r\n").encode("utf-8"))
            self.assertEqual(SuiteConfig.load(path).gate["critical_tags"], ["x"])
            from evalsprint.config import load_gate

            gate_file = Path(tmp) / "gate.yaml"
            gate_file.write_bytes(b"\xef\xbb\xbfgate:\r\n  max_error_rate: 0.2\r\n")
            self.assertEqual(load_gate(gate_file)["max_error_rate"], 0.2)

    def test_repeat_forms(self):
        self.assertIsNone(self.load().repeat)
        self.assertEqual(self.load(repeat={"n": 3}).repeat, {"n": 3, "per_tag": {}})
        self.assertEqual(self.load(repeat=4).repeat, {"n": 4, "per_tag": {}})  # shorthand
        self.assertIsNone(self.load(repeat={"n": 1}).repeat)  # one attempt is no repetition
        self.assertIsNone(self.load(repeat=None).repeat)

    def test_per_tag_repeat_forms(self):
        """C17: `repeat: {n: 3, per_tag: {safety: 5}}`; the keys are sorted; `n` may be left out (it is 1 then)."""
        self.assertEqual(self.load(repeat={"n": 3, "per_tag": {"safety": 5}}).repeat, {"n": 3, "per_tag": {"safety": 5}})
        self.assertEqual(list(self.load(repeat={"n": 3, "per_tag": {"zeta": 4, "alpha": 5}}).repeat["per_tag"]), ["alpha", "zeta"])
        self.assertEqual(self.load(repeat={"per_tag": {"safety": 5}}).repeat, {"n": 1, "per_tag": {"safety": 5}})
        self.assertEqual(self.load(repeat={"n": 3, "per_tag": {}}).repeat, {"n": 3, "per_tag": {}})
        self.assertEqual(self.load(repeat={"n": 3, "per_tag": {"docs": 1}}).repeat, {"n": 3, "per_tag": {"docs": 1}})  # a tag can run fewer times
        self.assertIsNone(self.load(repeat={"n": 1, "per_tag": {"docs": 1}}).repeat)  # one attempt everywhere: no repetition

    def test_the_old_tags_spelling_is_a_per_tag_protocol(self):
        """C17: `{n: 5, tags: [safety]}` (n attempts on the listed tags only) is `{n: 1, per_tag: {safety: 5}}`."""
        self.assertEqual(self.load(repeat={"n": 5, "tags": ["safety"]}).repeat, {"n": 1, "per_tag": {"safety": 5}})
        self.assertEqual(self.load(repeat={"n": 5, "tags": ["b", "a"]}).repeat, {"n": 1, "per_tag": {"a": 5, "b": 5}})
        self.assertIsNone(self.load(repeat={"n": 1, "tags": ["safety"]}).repeat)  # n == 1 never repeated anything
        self.assertConfigError("both 'tags'", repeat={"n": 3, "tags": ["a"], "per_tag": {"a": 5}})

    def test_per_tag_errors(self):
        self.assertConfigError("'repeat.per_tag' must map a tag", repeat={"n": 3, "per_tag": ["safety"]})
        self.assertConfigError("'repeat.per_tag.safety'", repeat={"n": 3, "per_tag": {"safety": 0}})
        self.assertConfigError("'repeat.per_tag.safety'", repeat={"n": 3, "per_tag": {"safety": 1000}})
        self.assertConfigError("'repeat.per_tag.safety' must be a whole number", repeat={"n": 3, "per_tag": {"safety": "many"}})
        self.assertConfigError("'repeat.per_tag' keys must be tags", repeat={"n": 3, "per_tag": {"": 5}})

    def test_target_retries(self):
        """C18: `target_retries: 1` by default, 0 switches it off, a few at most (every retry is a paid call)."""
        self.assertEqual(self.load().target_retries, 1)
        self.assertEqual(self.load(target_retries=0).target_retries, 0)
        self.assertEqual(self.load(target_retries=3).target_retries, 3)
        self.assertConfigError("'target_retries'", target_retries=-1)
        self.assertConfigError("'target_retries'", target_retries=50)
        self.assertConfigError("'target_retries' must be a whole number", target_retries="often")
        self.assertIn("did you mean 'target_retries'", self.assertConfigError("unknown setting 'target_retry'", target_retry=1))

    def test_repeat_errors(self):
        self.assertIn("did you mean 'n'", self.assertConfigError("unknown repeat setting 'nn'", repeat={"nn": 3}))
        self.assertConfigError("needs 'n'", repeat={"tags": ["a"]})
        self.assertConfigError("'repeat.n'", repeat={"n": 0})
        self.assertConfigError("'repeat.n'", repeat={"n": 1000})
        self.assertConfigError("'repeat.n' must be a whole number", repeat={"n": 2.5})
        self.assertConfigError("'repeat.tags'", repeat={"n": 3, "tags": "safety"})
        self.assertConfigError("'repeat'", repeat="three")

    def test_tag_groups_are_passed_through(self):
        groups = {"type": ["answerable", "safety"], "topic": ["billing"]}
        self.assertEqual(self.load(tag_groups=groups).tag_groups, groups)
        self.assertEqual(self.load().tag_groups, {})
        self.assertConfigError("'tag_groups.type'", tag_groups={"type": "answerable"})
        self.assertConfigError("'tag_groups'", tag_groups=["a"])

    def test_behaviors(self):
        self.assertEqual(self.load(behaviors=["answer", "abstain", "refuse"]).behaviors, ["answer", "abstain", "refuse"])
        self.assertIsNone(self.load().behaviors)
        self.assertConfigError("'behaviors'", behaviors="answer")
        self.assertConfigError("'behaviors'", behaviors=[])

    def test_env_values_reach_the_new_sections_typed(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"ES_N": "3"}):
            path = write_yaml(Path(tmp), {**BASE, "repeat": {"n": "${ES_N}", "per_tag": {"safety": "${ES_N}"}}})
            self.assertEqual(SuiteConfig.load(path).repeat, {"n": 3, "per_tag": {"safety": 3}})
            path = write_yaml(Path(tmp), {**BASE, "repeat": {"n": "${ES_N}", "tags": ["safety"]}})
            self.assertEqual(SuiteConfig.load(path).repeat, {"n": 1, "per_tag": {"safety": 3}})  # the old spelling, normalised


class GateLoadingTests(unittest.TestCase):
    def test_merge_gate_overlays_defaults_and_copies(self):
        gate = merge_gate({"min_pass_rate": 0.8})
        self.assertEqual(gate["min_pass_rate"], 0.8)
        self.assertEqual(gate["max_error_rate"], DEFAULT_GATE["max_error_rate"])
        gate["critical_tags"].append("x")
        self.assertEqual(DEFAULT_GATE["critical_tags"], [])
        self.assertEqual(merge_gate(None), DEFAULT_GATE)

    def test_merge_gate_rejects_unknown_keys_and_non_mappings(self):
        with self.assertRaises(ValueError) as ctx:
            merge_gate({"min_pass_rat": 0.8})
        self.assertIn("did you mean 'min_pass_rate'", str(ctx.exception))
        with self.assertRaises(ValueError):
            merge_gate([1])

    def test_load_gate_takes_only_the_gate_block_of_that_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = write_yaml(Path(tmp), {**BASE, "gate": {"max_pass_rate_drop": 0.02, "critical_tags": ["safety"]}}, "base_eval.yaml")
            gate = load_gate(full)
            self.assertEqual(gate["max_pass_rate_drop"], 0.02)
            self.assertEqual(gate["critical_tags"], ["safety"])
            self.assertEqual(gate["max_error_rate"], DEFAULT_GATE["max_error_rate"])  # not mentioned -> default, never something else
            only_gate = write_yaml(Path(tmp), "gate:\n  min_pass_rate: 0.9\n", "gate_only.yaml")  # a file without suite/dataset/target is fine
            self.assertEqual(load_gate(only_gate)["min_pass_rate"], 0.9)
            self.assertEqual(load_gate(write_yaml(Path(tmp), "suite: x\n", "nogate.yaml")), DEFAULT_GATE)
            self.assertEqual(load_gate(write_yaml(Path(tmp), "", "empty.yaml")), DEFAULT_GATE)

    def test_load_gate_expands_env_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_yaml(Path(tmp), "gate:\n  max_pass_rate_drop: ${ES_DROP:-0.07}\n")
            self.assertEqual(load_gate(path)["max_pass_rate_drop"], 0.07)
            with self.assertRaises(ValueError):
                load_gate(write_yaml(Path(tmp), "gate:\n  bogus: 1\n", "bad.yaml"))
            with self.assertRaises(ValueError):
                load_gate(write_yaml(Path(tmp), "- a\n", "list.yaml"))
            with self.assertRaises(FileNotFoundError):
                load_gate(Path(tmp) / "missing.yaml")


class RedactSecretsTests(unittest.TestCase):
    SECRET_KEYS = [
        "api_key", "apiKey", "x-api-key", "API_KEY", "openai_api_key", "access_token", "auth_token", "token", "client_secret", "secret", "secret_key",
        "private_key", "secret_access_key", "AWS_SECRET_ACCESS_KEY", "password", "db_password", "passwd", "Authorization", "Proxy-Authorization",
        "bearer", "jwt", "cookie", "session_id", "session", "dsn", "connection_string", "key", "credentials", "credential", "signing_key",
        "webhook_secret", "stripe_key", "PRIVATE-KEY", "refreshToken", "accessKey",
    ]
    HARMLESS_KEYS = ["max_tokens", "api_key_env", "token_url", "cache_key", "sort_key", "top_k", "prompt", "model", "keywords", "tokens_per_minute", "provider", "temperature"]

    def test_every_secret_looking_key_is_masked(self):
        from evalsprint.config import redact_secrets

        out = redact_secrets({k: "VALUE-12345" for k in self.SECRET_KEYS})
        self.assertEqual([k for k in self.SECRET_KEYS if out[k] != "***"], [])

    def test_harmless_keys_are_left_alone(self):
        from evalsprint.config import redact_secrets

        data = {k: "plain-value" for k in self.HARMLESS_KEYS}
        self.assertEqual(redact_secrets(data), data)

    def test_empty_and_none_values_are_kept_and_input_is_not_mutated(self):
        from evalsprint.config import redact_secrets

        data = {"api_key": "", "token": None, "password": "p", "nested": {"secret": "s"}}
        out = redact_secrets(data)
        self.assertEqual((out["api_key"], out["token"], out["password"], out["nested"]), ("", None, "***", {"secret": "***"}))
        self.assertEqual(data["password"], "p")
        self.assertEqual(data["nested"], {"secret": "s"})

    def test_structured_secret_values_are_masked_whole(self):
        from evalsprint.config import redact_secrets

        self.assertEqual(redact_secrets({"credentials": {"user": "u", "pass": "p"}, "cookies": ["a=b"]}), {"credentials": "***", "cookies": "***"})

    def test_urls_with_userinfo_and_token_shapes_are_masked_in_any_value(self):
        from evalsprint.config import redact_secrets

        out = redact_secrets(
            {
                "url": "https://user:pass@host.example.com/x?y=1",
                "db": "postgres://svc:hunter2@db.internal:5432/app",
                "note": ["Bearer abcdef1234567890", "key sk-ant-api03-abcdefghijklmnop1234 leaked", {"deep": "ghp_" + "a" * 30}],
                "clean_url": "https://api.example.com/v1/messages",
            }
        )
        blob = str(out)
        for leaked in ("user:pass", "hunter2", "abcdef1234567890", "sk-ant-api03", "ghp_aaaa"):
            self.assertNotIn(leaked, blob)
        self.assertEqual(out["url"], "https://***@host.example.com/x?y=1")
        self.assertEqual(out["clean_url"], "https://api.example.com/v1/messages")


if __name__ == "__main__":
    unittest.main()
