"""Configuration and test-set validation (C8): typos fail loudly, at `describe`/load time, with a suggestion.

Every one of these used to be silently ignored, and a silent typo in a judge or a case turns into a
green gate that measures nothing. Errors are one line on stderr and exit code 2; nothing is run.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from evalsprint.cli import main
from evalsprint.config import SuiteConfig
from evalsprint.judges import build_judges, validate_pass_rule
from evalsprint.pricing import PriceTable

TARGET_PY = '''
def answer(input, params):
    return {"text": "answer to " + str(input), "model": "claude-haiku-4-5", "usage": {"input_tokens": 10, "output_tokens": 5}}
'''

JUDGE_PY = '''
import json


def responder(system, messages, model):
    return {"text": json.dumps({"criteria": {"ok": {"reasoning": "fine", "pass": True}}}), "usage": {"input_tokens": 50, "output_tokens": 10}}
'''

CUSTOM_PY = '''
def check(case, output):
    return True
'''

CASES = [
    {"id": "a-1", "input": "one", "tags": ["billing"], "expected_behavior": "answer"},
    {"id": "a-2", "input": "two", "tags": ["billing", "core"], "expected_behavior": "answer"},
    {"id": "b-1", "input": "three", "tags": ["shipping"], "expected_behavior": "abstain"},
]

LLM_JUDGE = {
    "type": "llm",
    "name": "quality",
    "provider": "mock",
    "provider_options": {"responder": "judge_sim.py:responder"},
    "model": "claude-haiku-4-5",
    "rubric": {"name": "r", "task": "Support answers.", "criteria": [{"id": "ok", "question": "Is it fine?"}]},
}

BASE = {
    "suite": "validation",
    "dataset": "cases.jsonl",
    "target": {"callable": "target.py:answer", "params": {}},
    "judges": [LLM_JUDGE],
}


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


class SuiteCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "target.py").write_text(TARGET_PY, encoding="utf-8")
        (self.dir / "judge_sim.py").write_text(JUDGE_PY, encoding="utf-8")
        (self.dir / "my_judges.py").write_text(CUSTOM_PY, encoding="utf-8")
        self.write_cases(CASES)

    def write_cases(self, cases):
        (self.dir / "cases.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")

    def config(self, **changes):
        data = {**BASE, **changes}
        data = {k: v for k, v in data.items() if v is not None}
        path = self.dir / "eval.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return path

    def assertRejected(self, command, path, *fragments):
        code, out, err = cli(command, "-c", path, *(["--quiet"] if command == "run" else []))
        self.assertEqual(code, 2, (out, err))
        lines = err.strip().splitlines()
        self.assertEqual(len(lines), 1, err)  # one line, no traceback
        self.assertTrue(lines[0].startswith("error: "), err)
        self.assertNotIn("Traceback", err)
        for fragment in fragments:
            self.assertIn(fragment, err)
        self.assertFalse((self.dir / "runs").exists())  # rejected before anything was run or paid for
        return err


class HeuristicJudgeTypoTests(SuiteCase):
    def test_a_misspelled_setting_suggests_the_right_one(self):
        cfg = self.config(judges=[{"type": "contains", "name": "has_link", "valeus": ["http"]}])
        self.assertRejected("describe", cfg, "judge 'has_link' (contains)", "unknown setting 'valeus'", "did you mean 'values'")

    def test_a_missing_required_setting_is_named(self):
        cfg = self.config(judges=[{"type": "max_chars", "name": "short"}])
        self.assertRejected("describe", cfg, "missing required setting", "'value'")

    def test_regex_typos_and_bad_patterns(self):
        cfg = self.config(judges=[{"type": "regex", "name": "r", "patern": "x"}])
        self.assertRejected("describe", cfg, "did you mean 'pattern'")
        cfg = self.config(judges=[{"type": "regex", "name": "r", "pattern": "("}])
        self.assertRejected("describe", cfg, "judge 'r' (regex)")  # an invalid pattern is a config error, not a crash

    def test_wrong_value_types_are_config_errors(self):
        for judge, fragment in (
            ({"type": "contains", "name": "c", "values": 7}, "values"),
            ({"type": "contains", "name": "c", "values": ["ok", 3]}, "values"),
            ({"type": "contains", "name": "c", "values": []}, "values"),
            ({"type": "contains", "name": "c", "values": ["x"], "mode": "some"}, "mode"),
            ({"type": "max_chars", "name": "m", "value": -3}, "value"),
            ({"type": "contains", "name": "c", "values": ["x"], "case_insensitive": "maybe"}, "case_insensitive"),
        ):
            with self.subTest(judge=judge):
                self.assertRejected("describe", self.config(judges=[judge]), fragment)

    def test_scoping_settings_are_validated_on_every_judge(self):
        for scope in ({"when_tags": 5}, {"when_tags": ["billing", 2]}, {"when_behavior": {"a": 1}}):
            with self.subTest(scope=scope):
                cfg = self.config(judges=[{"type": "contains", "name": "c", "values": ["x"], **scope}])
                self.assertRejected("describe", cfg, next(iter(scope)))

    def test_a_bare_string_means_one_value_not_its_letters(self):
        cfg = self.config(judges=[{"type": "contains", "name": "c", "values": "http", "when_tags": "billing"}])
        code, out, err = cli("describe", "-c", cfg)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("(2/3 cases)", out)  # `when_tags: billing` scopes to the billing cases, not to the tags b, i, l, ...

    def test_a_valid_heuristic_judge_passes(self):
        cfg = self.config(judges=[{"type": "contains", "name": "c", "values": ["answer"], "mode": "all", "when_tags": ["billing"]}])
        code, out, err = cli("describe", "-c", cfg)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("contains", out)
        self.assertIn("(2/3 cases)", out)  # scoped to the two billing cases

    def test_numeric_strings_from_the_environment_are_accepted(self):
        cfg = self.config(judges=[{"type": "max_chars", "name": "m", "value": "${MAX_CHARS}"}])
        with mock.patch.dict(os.environ, {"MAX_CHARS": "200"}):
            self.assertEqual(cli("describe", "-c", cfg)[0], 0)


class CustomJudgeTypoTests(SuiteCase):
    def test_a_misspelled_callable(self):
        cfg = self.config(judges=[{"type": "custom", "name": "mine", "callabel": "my_judges.py:check"}])
        self.assertRejected("describe", cfg, "judge 'mine' (custom)", "unknown setting 'callabel'", "did you mean 'callable'")

    def test_a_missing_callable(self):
        cfg = self.config(judges=[{"type": "custom", "name": "mine"}])
        self.assertRejected("describe", cfg, "missing required setting 'callable'")

    def test_a_misspelled_scope(self):
        cfg = self.config(judges=[{"type": "custom", "name": "mine", "callable": "my_judges.py:check", "when_tag": ["billing"]}])
        self.assertRejected("describe", cfg, "did you mean 'when_tags'")

    def test_a_callable_that_does_not_exist(self):
        cfg = self.config(judges=[{"type": "custom", "name": "mine", "callable": "my_judges.py:chek"}])
        self.assertRejected("describe", cfg, "chek")

    def test_a_valid_custom_judge(self):
        cfg = self.config(judges=[{"type": "custom", "name": "mine", "callable": "my_judges.py:check"}])
        self.assertEqual(cli("describe", "-c", cfg)[0], 0)


class LlmJudgeValidationTests(SuiteCase):
    def test_its_own_errors_reach_the_user_unchanged(self):
        # the LLM judge validates its own settings (its signature is the source of truth): build_judges must not swallow that
        boom = ValueError("llm judge 'quality': the rubric is missing the criterion marker-7f3a")
        with mock.patch("evalsprint.judges.LLMJudge.from_config", side_effect=boom):
            self.assertRejected("describe", self.config(), "marker-7f3a")
        with self.assertRaises(ValueError) as caught, mock.patch("evalsprint.judges.LLMJudge.from_config", side_effect=boom):
            build_judges([dict(LLM_JUDGE)], self.dir, PriceTable({}))
        self.assertIs(caught.exception, boom)

    def test_a_misspelled_setting_is_rejected_with_a_suggestion(self):
        cfg = self.config(judges=[{**LLM_JUDGE, "temprature": 0}])
        self.assertRejected("describe", cfg, "'temprature'", "did you mean")

    def test_missing_model_and_rubric(self):
        no_model = {k: v for k, v in LLM_JUDGE.items() if k != "model"}
        self.assertRejected("describe", self.config(judges=[no_model]), "quality", "model")
        no_rubric = {k: v for k, v in LLM_JUDGE.items() if k != "rubric"}
        self.assertRejected("describe", self.config(judges=[no_rubric]), "quality", "rubric")

    def test_describe_builds_the_llm_judge_offline_without_an_api_key(self):
        real = {**LLM_JUDGE, "provider": "anthropic", "provider_options": {}}
        cfg = self.config(judges=[real])
        with mock.patch.dict(os.environ, {}, clear=True):
            code, out, err = cli("describe", "-c", cfg)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("anthropic/claude-haiku-4-5", out)  # the configured provider is shown, not the offline stand-in

    def test_describe_still_validates_the_llm_judge_settings_offline(self):
        real = {**LLM_JUDGE, "provider": "anthropic", "provider_options": {}, "rubrik": "x"}
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertRejected("describe", self.config(judges=[real]), "'rubrik'")

    def test_an_unknown_provider_is_named_with_a_suggestion(self):
        cfg = self.config(judges=[{**LLM_JUDGE, "provider": "antropic", "provider_options": {}}])
        self.assertRejected("describe", cfg, "unknown provider 'antropic'", "did you mean 'anthropic'")


class JudgeListTests(SuiteCase):
    def test_an_unknown_type_suggests_the_closest_one(self):
        cfg = self.config(judges=[{"type": "contain", "name": "c", "values": ["x"]}])
        self.assertRejected("describe", cfg, "unknown judge type 'contain'", "did you mean 'contains'")

    def test_a_judge_needs_a_type(self):
        self.assertRejected("describe", self.config(judges=[{"name": "c", "values": ["x"]}]), "needs a 'type'")

    def test_a_judge_must_be_a_mapping(self):
        self.assertRejected("describe", self.config(judges=["contains"]), "mapping")

    def test_judges_must_be_a_list(self):
        self.assertRejected("describe", self.config(judges={"type": "contains"}), "list")

    def test_duplicate_names(self):
        cfg = self.config(judges=[{"type": "contains", "name": "c", "values": ["x"]}, {"type": "not_contains", "name": "c", "values": ["y"]}])
        self.assertRejected("describe", cfg, "duplicate judge name 'c'")

    def test_no_judges_stops_a_run_before_it_starts(self):
        cfg = self.config(judges=[])
        self.assertRejected("run", cfg, "no judges configured")

    def test_no_judges_is_a_warning_in_describe(self):
        code, out, err = cli("describe", "-c", self.config(judges=[]))
        self.assertEqual(code, 0)
        self.assertIn("no judges configured", out)

    def test_pass_rule_must_name_configured_judges(self):
        cfg = self.config(pass_rule=["qualty"])
        self.assertRejected("describe", cfg, "pass_rule names judge 'qualty'", "did you mean 'quality'", "judges: quality")
        self.assertRejected("run", cfg, "pass_rule names judge 'qualty'")

    def test_valid_pass_rules(self):
        for rule in ("all", "any", "quality", ["quality"]):
            with self.subTest(rule=rule):
                validate_pass_rule(rule, ["quality"])
        for rule in (3, [], [1], None, {"a": 1}):
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                validate_pass_rule(rule, ["quality"])

    def test_case_checks_are_added_when_cases_carry_them(self):
        judges = build_judges([{"type": "min_chars", "name": "m", "value": 1}], self.dir, PriceTable({}), cases=[])
        self.assertEqual([j.name for j in judges], ["m"])


class ConfigKeyTests(SuiteCase):
    def test_unknown_top_level_key(self):
        data = {**BASE, "judgs": [LLM_JUDGE]}
        path = self.dir / "typo.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.assertRejected("describe", path, "unknown", "'judgs'", "did you mean 'judges'")

    def test_unknown_target_key(self):
        cfg = self.config(target={"callable": "target.py:answer", "parms": {"x": 1}})
        self.assertRejected("describe", cfg, "'parms'", "did you mean 'params'")

    def test_unknown_gate_key(self):
        cfg = self.config(gate={"max_pass_rate_dop": 0.1})
        self.assertRejected("describe", cfg, "'max_pass_rate_dop'", "did you mean 'max_pass_rate_drop'")
        self.assertRejected("run", cfg, "'max_pass_rate_dop'")

    def test_concurrency_below_one_is_rejected(self):
        for value in (0, -1):
            self.assertRejected("describe", self.config(concurrency=value), "'concurrency' must be at least 1")
        self.assertRejected("run", self.config(concurrency=0), "'concurrency' must be at least 1")

    def test_critical_tags_of_the_wrong_type(self):
        self.assertRejected("describe", self.config(gate={"critical_tags": 7}), "gate.critical_tags must be a tag or a list of tags")

    def test_malformed_gate_values(self):
        self.assertRejected("describe", self.config(gate={"max_pass_rate_drop": 7}), "max_pass_rate_drop")
        self.assertRejected("describe", self.config(gate={"max_error_rate": "lots"}), "max_error_rate")

    def test_the_repeat_block_is_validated(self):
        self.assertRejected("describe", self.config(repeat={"n": 3, "tag": ["core"]}), "'tag'", "did you mean 'tags'")
        self.assertRejected("describe", self.config(repeat={"n": 0}), "repeat")
        self.assertRejected("describe", self.config(repeat={"n": 500}), "50")

    def test_behaviors_must_be_a_list_of_names(self):
        self.assertRejected("describe", self.config(behaviors="answer"), "behaviors")

    def test_valid_config_with_every_new_block_is_accepted(self):
        cfg = self.config(
            repeat={"n": 3, "tags": ["core"]},
            behaviors=["answer", "abstain"],
            tag_groups={"money": ["billing"]},
            gate={"max_pass_rate_drop": 0.05, "critical_tags": ["core"], "max_error_rate": 0.1},
        )
        code, out, err = cli("describe", "-c", cfg)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("Repeat", out)
        self.assertIn("Tag groups", out)
        self.assertIn("Behaviors", out)


class DescribeSecretsTests(SuiteCase):
    PARAMS = {
        "model": "claude-haiku-4-5",
        "api_key": "sk-ant-api03-abcdefghijklmnop",
        "auth_token": "hunter2-hunter2",
        "client_secret": "s3cr3t-value",
        "endpoint": "postgres://svc:p4ssw0rd@db.internal:5432/app",
        "note": "call with Bearer abcdefghijklmnopqrstuvwxyz123456",
        "nested": {"password": "pw-123456", "region": "eu-west-1"},
    }
    SECRETS = ("sk-ant-api03-abcdefghijklmnop", "hunter2-hunter2", "s3cr3t-value", "p4ssw0rd", "abcdefghijklmnopqrstuvwxyz123456", "pw-123456")

    def test_describe_does_not_print_secrets(self):
        cfg = self.config(target={"callable": "target.py:answer", "params": self.PARAMS})
        code, out, err = cli("describe", "-c", cfg)
        self.assertEqual((code, err), (0, ""))
        for secret in self.SECRETS:
            self.assertNotIn(secret, out)
        self.assertIn("claude-haiku-4-5", out)  # ordinary params stay readable
        self.assertIn("eu-west-1", out)

    def test_the_run_summary_does_not_keep_secrets_either(self):
        cfg = self.config(target={"callable": "target.py:answer", "params": self.PARAMS})
        out_dir = self.dir / "run"
        code, _, err = cli("run", "-c", cfg, "--out", out_dir, "--quiet")
        self.assertEqual(code, 0, err)
        stored = (out_dir / "summary.json").read_text(encoding="utf-8")
        for secret in self.SECRETS:
            self.assertNotIn(secret, stored)


class CaseSchemaCliTests(SuiteCase):
    """The unit-level checks live in test_dataset.py; here: they reach the user as a one-line usage error."""

    def test_tags_must_be_a_list_of_strings(self):
        self.write_cases([{"id": "x-1", "input": "q", "tags": "billing"}])
        self.assertRejected("describe", self.config(), "x-1", "tags")

    def test_an_empty_id(self):
        self.write_cases([{"id": "", "input": "q"}])
        self.assertRejected("describe", self.config(), "id")

    def test_a_bad_check_type_is_caught_at_load(self):
        self.write_cases([{"id": "x-1", "input": "q", "checks": [{"type": "contain", "values": ["a"]}]}])
        self.assertRejected("describe", self.config(), "x-1", "did you mean 'contains'")

    def test_a_check_setting_typo_is_caught_at_load(self):
        self.write_cases([{"id": "x-1", "input": "q", "checks": [{"type": "contains", "valeus": ["a"]}]}])
        self.assertRejected("describe", self.config(), "x-1", "did you mean 'values'")

    def test_expected_behavior_outside_the_declared_behaviors(self):
        self.write_cases([{"id": "x-1", "input": "q", "expected_behavior": "anser"}])
        self.assertRejected("describe", self.config(behaviors=["answer", "abstain"]), "x-1", "anser", "did you mean 'answer'")

    def test_duplicate_case_ids(self):
        self.write_cases([{"id": "x-1", "input": "q"}, {"id": "x-1", "input": "r"}])
        self.assertRejected("describe", self.config(), "duplicate")

    def test_invalid_json_line_is_located(self):
        (self.dir / "cases.jsonl").write_text('{"id": "a", "input": "q"}\n{oops\n', encoding="utf-8")
        self.assertRejected("describe", self.config(), "cases.jsonl:2")

    def test_a_valid_test_set_still_passes(self):
        code, out, err = cli("describe", "-c", self.config())
        self.assertEqual((code, err), (0, ""))
        self.assertIn("Config and test set are valid", out)


class DescribeWarningTests(SuiteCase):
    def test_critical_tag_that_no_case_has(self):
        code, out, _ = cli("describe", "-c", self.config(gate={"critical_tags": ["biling"]}))
        self.assertEqual(code, 0)
        self.assertIn("no case has the tag 'biling'", out)
        self.assertIn("did you mean 'billing'", out)

    def test_critical_tags_written_as_a_string_do_not_warn_per_character(self):
        code, out, err = cli("describe", "-c", self.config(gate={"critical_tags": "billing"}))
        self.assertEqual((code, err), (0, ""))
        self.assertNotIn("no case has the tag 'b'", out)  # a string used to be iterated: b, i, l, l, ...
        self.assertNotIn("no case has the tag 'i'", out)
        self.assertIn("critical tags ['billing']", out)  # read as the one tag
        code, out, _ = cli("describe", "-c", self.config(gate={"critical_tags": "biling"}))
        self.assertEqual(out.count("no case has the tag"), 1)
        self.assertIn("no case has the tag 'biling'", out)

    def test_critical_tags_that_are_not_repeated(self):
        from evalsprint.compare import RECOMMENDED_CRITICAL_REPEATS

        code, out, _ = cli("describe", "-c", self.config(gate={"critical_tags": ["core"]}))
        self.assertIn("are not repeated", out)
        # C17: the recommendation is a per-tag protocol with the number compare recommends, never a block that replaces the whole protocol
        self.assertIn(f"Consider adding repeat: {{n: 1, per_tag: {{core: {RECOMMENDED_CRITICAL_REPEATS}}}}}", out)
        code, out, _ = cli("describe", "-c", self.config(gate={"critical_tags": ["core"]}, repeat={"n": 3, "per_tag": {"core": 5}}))
        self.assertNotIn("are not repeated", out)
        self.assertNotIn("fewer than", out)
        self.assertIn("8 extra target call(s)", out)  # the core case runs 5 times (4 extra); the other two keep n: 3 (2 extra each)
        # a protocol that repeats everything but too little for the critical tag is told to raise that tag, keeping n
        code, out, _ = cli("describe", "-c", self.config(gate={"critical_tags": ["core"]}, repeat={"n": 3}))
        self.assertIn("run fewer than", out)
        self.assertIn(f"adding per_tag: {{core: {RECOMMENDED_CRITICAL_REPEATS}}} under repeat (n: 3 stays as it is)", out)

    def test_a_judge_that_applies_to_no_case(self):
        cfg = self.config(judges=[LLM_JUDGE, {"type": "contains", "name": "orphan", "values": ["x"], "when_tags": ["refund"]}])
        code, out, _ = cli("describe", "-c", cfg)
        self.assertEqual(code, 0)
        self.assertIn("judge 'orphan' applies to no case", out)

    def test_tag_group_that_names_a_missing_tag(self):
        code, out, _ = cli("describe", "-c", self.config(tag_groups={"money": ["billing", "invoces"]}))
        self.assertIn("tag_groups.money: no case has the tag 'invoces'", out)


class LoaderTests(SuiteCase):
    def test_suite_config_load_reports_the_same_errors_programmatically(self):
        path = self.config(judges=[{"type": "contains", "name": "c", "valeus": ["x"]}])
        config = SuiteConfig.load(path)  # loading itself is lenient about judge settings; building them is not
        with self.assertRaises(ValueError) as caught:
            build_judges(config.judges, config.base_dir, PriceTable({}))
        self.assertIn("did you mean 'values'", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
