import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from evalsprint.judges import build_judges, run_judges
from evalsprint.judges.heuristics import (
    CaseChecksJudge,
    Contains,
    CustomJudge,
    ExactMatch,
    JsonValid,
    MaxChars,
    MinChars,
    NotContains,
    Regex,
)
from evalsprint.judges.llm_judge import LLMJudge, Rubric, extract_json
from evalsprint.pricing import PriceTable
from evalsprint.providers import MockProvider
from evalsprint.types import Case, JudgeResult, TargetOutput, Usage
from tests import make_case, make_output


def ev(judge, text="", case=None):
    return judge.evaluate(case or make_case(), make_output(text))


class ContainsTests(unittest.TestCase):
    def test_any_mode_pass_and_fail(self):
        j = Contains("c", ["refund", "money back"])
        self.assertTrue(ev(j, "You can get a REFUND within 30 days").passed)
        self.assertTrue(ev(j, "we return your money back").passed)
        r = ev(j, "sorry, no")
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.0)

    def test_all_mode(self):
        j = Contains("c", ["invoice", "settings"], mode="all")
        self.assertTrue(ev(j, "Open Settings and download the invoice").passed)
        r = ev(j, "Open Settings")
        self.assertFalse(r.passed)
        self.assertAlmostEqual(r.score, 0.5)
        self.assertIn("invoice", r.reasoning)

    def test_case_sensitivity(self):
        self.assertFalse(ev(Contains("c", "API", case_insensitive=False), "the api").passed)
        self.assertTrue(ev(Contains("c", "API", case_insensitive=False), "the API").passed)
        self.assertTrue(ev(Contains("c", "API"), "the api").passed)

    def test_single_string_value(self):
        self.assertTrue(ev(Contains("c", "hello"), "Hello world").passed)

    def test_whitespace_and_unicode_are_normalised(self):
        self.assertTrue(ev(Contains("c", "Settings > Billing"), "Go to  Settings\n>\tBilling now").passed)
        self.assertTrue(ev(Contains("c", "ﬁle"), "your file is ready").passed)  # NFKC folds the ligature

    def test_score_is_fraction_of_hits(self):
        r = ev(Contains("c", ["a", "b", "c", "d"]), "a b")
        self.assertAlmostEqual(r.score, 0.5)


class NotContainsTests(unittest.TestCase):
    def test_pass_and_fail(self):
        j = NotContains("n", ["as an ai", "i cannot"])
        self.assertTrue(ev(j, "Here is your answer.").passed)
        r = ev(j, "As an AI language model, I...")
        self.assertFalse(r.passed)
        self.assertIn("as an ai", r.reasoning.lower())

    def test_case_sensitive_option(self):
        j = NotContains("n", "SECRET", case_insensitive=False)
        self.assertTrue(ev(j, "a secret").passed)
        self.assertFalse(ev(j, "a SECRET").passed)


class RegexTests(unittest.TestCase):
    def test_must_match(self):
        j = Regex("r", r"\border #\d{4}\b")
        self.assertTrue(ev(j, "Your ORDER #1234 shipped").passed)  # default flag i
        self.assertFalse(ev(j, "Your order shipped").passed)

    def test_must_not_match(self):
        j = Regex("r", r"\b\d{16}\b", must_match=False)
        self.assertTrue(ev(j, "no card numbers here").passed)
        self.assertFalse(ev(j, "card 4111111111111111").passed)

    def test_flags(self):
        self.assertFalse(ev(Regex("r", "abc", flags=""), "ABC").passed)
        self.assertFalse(ev(Regex("r", "a.c", flags="s"), "A\nC").passed)  # s without i: case matters
        self.assertTrue(ev(Regex("r", "a.c", flags="si"), "A\nC").passed)
        self.assertTrue(ev(Regex("r", "^two$", flags="mi"), "one\nTWO\nthree").passed)
        self.assertFalse(ev(Regex("r", "^two$", flags="i"), "one\nTWO\nthree").passed)

    def test_invalid_pattern_fails_at_build_time(self):
        import re

        with self.assertRaises(re.error):
            Regex("r", "(unclosed")


class LengthTests(unittest.TestCase):
    def test_max_chars(self):
        j = MaxChars("m", 10)
        self.assertTrue(ev(j, "x" * 10).passed)
        r = ev(j, "x" * 11)
        self.assertFalse(r.passed)
        self.assertIn("11 chars", r.reasoning)

    def test_max_chars_accepts_numeric_string(self):
        self.assertTrue(ev(MaxChars("m", "5"), "12345").passed)

    def test_min_chars_strips_whitespace(self):
        j = MinChars("m", 5)
        self.assertTrue(ev(j, "hello").passed)
        self.assertFalse(ev(j, "   hi    ").passed)
        self.assertFalse(ev(j, "").passed)


class JsonValidTests(unittest.TestCase):
    def test_valid_and_invalid(self):
        j = JsonValid("j")
        self.assertTrue(ev(j, '{"a": 1}').passed)
        self.assertTrue(ev(j, "[1, 2, 3]").passed)
        r = ev(j, "not json")
        self.assertFalse(r.passed)
        self.assertIn("invalid JSON", r.reasoning)

    def test_code_fenced_json(self):
        j = JsonValid("j")
        self.assertTrue(ev(j, '```json\n{"a": 1}\n```').passed)
        self.assertTrue(ev(j, '```\n{"a": 1}\n```').passed)

    def test_required_keys(self):
        j = JsonValid("j", required_keys=["answer", "sources"])
        self.assertTrue(ev(j, '{"answer": "x", "sources": []}').passed)
        r = ev(j, '{"answer": "x"}')
        self.assertFalse(r.passed)
        self.assertIn("sources", r.reasoning)
        self.assertFalse(ev(j, "[1, 2]").passed)  # keys required but output is not an object

    def test_empty_output(self):
        self.assertFalse(ev(JsonValid("j"), "").passed)


class ExactMatchTests(unittest.TestCase):
    def test_match_ignores_case_and_whitespace_by_default(self):
        j = ExactMatch("e")
        self.assertTrue(ev(j, "  The  Answer ", make_case(reference="the answer")).passed)
        self.assertFalse(ev(j, "the answer!", make_case(reference="the answer")).passed)

    def test_case_sensitive(self):
        j = ExactMatch("e", case_insensitive=False)
        self.assertFalse(ev(j, "Yes", make_case(reference="yes")).passed)
        self.assertTrue(ev(j, "yes", make_case(reference="yes")).passed)

    def test_missing_reference_is_an_error_not_a_pass(self):
        r = ev(ExactMatch("e"), "anything", make_case(reference=None))
        self.assertFalse(r.passed)
        self.assertIn("no reference", r.error)


class CaseChecksTests(unittest.TestCase):
    def case(self, checks):
        return make_case(checks=checks)

    def test_applies_only_to_cases_with_checks(self):
        j = CaseChecksJudge(name="case_checks")
        self.assertTrue(j.applies(self.case([{"type": "contains", "values": ["x"]}])))
        self.assertFalse(j.applies(self.case([])))

    def test_all_checks_pass(self):
        case = self.case([{"type": "contains", "values": ["Settings > Billing"]}, {"type": "max_chars", "value": 100}])
        r = ev(CaseChecksJudge(name="case_checks"), "Go to Settings > Billing.", case)
        self.assertTrue(r.passed)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.details, {"contains#0": True, "max_chars#1": True})

    def test_partial_failure_reports_failing_check(self):
        case = self.case([{"type": "contains", "values": ["reset link"]}, {"type": "not_contains", "values": ["password is"]}])
        r = ev(CaseChecksJudge(name="case_checks"), "Your password is hunter2", case)
        self.assertFalse(r.passed)
        self.assertAlmostEqual(r.score, 0.0)
        self.assertIn("contains#0", r.reasoning)
        self.assertIn("not_contains#1", r.reasoning)
        r = ev(CaseChecksJudge(name="case_checks"), "reset link sent; password is hidden", case)
        self.assertAlmostEqual(r.score, 0.5)
        self.assertNotIn("contains#0", r.reasoning.replace("not_contains#1", ""))

    def test_check_options_are_forwarded(self):
        case = self.case([{"type": "contains", "values": ["A", "B"], "mode": "all"}, {"type": "regex", "pattern": "^x$", "flags": ""}])
        self.assertFalse(ev(CaseChecksJudge(name="c"), "A only", case).passed)
        self.assertTrue(ev(CaseChecksJudge(name="c"), "x", self.case([{"type": "regex", "pattern": "^x$", "flags": ""}])).passed)

    def test_original_checks_are_not_mutated(self):
        checks = [{"type": "contains", "values": ["x"]}]
        ev(CaseChecksJudge(name="c"), "x", self.case(checks))
        self.assertEqual(checks, [{"type": "contains", "values": ["x"]}])

    def test_exact_match_check_uses_case_reference(self):
        case = make_case(reference="42", checks=[{"type": "exact_match"}])
        self.assertTrue(ev(CaseChecksJudge(name="c"), "42", case).passed)

    def test_unknown_check_type_is_recorded_as_judge_error(self):
        case = self.case([{"type": "telepathy"}])
        (res,) = run_judges([CaseChecksJudge(name="case_checks")], case, make_output("x"))
        self.assertFalse(res.passed)
        self.assertIn("KeyError", res.error)

    def test_check_without_type_is_recorded_as_judge_error(self):
        (res,) = run_judges([CaseChecksJudge(name="case_checks")], self.case([{"values": ["x"]}]), make_output("x"))
        self.assertFalse(res.passed)
        self.assertIsNotNone(res.error)


class CustomJudgeTests(unittest.TestCase):
    def test_bool_return(self):
        j = CustomJudge("c", fn=lambda case, out: "ok" in out.text)
        self.assertTrue(ev(j, "ok").passed)
        self.assertFalse(ev(j, "no").passed)

    def test_dict_return(self):
        j = CustomJudge("c", fn=lambda case, out: {"passed": True, "score": 0.75, "reasoning": "fine"})
        r = ev(j, "x")
        self.assertEqual((r.passed, r.score, r.reasoning), (True, 0.75, "fine"))
        r = ev(CustomJudge("c", fn=lambda case, out: {"passed": False}), "x")
        self.assertEqual((r.passed, r.score, r.reasoning), (False, None, ""))

    def test_fn_receives_case_and_output(self):
        seen = []
        j = CustomJudge("c", fn=lambda case, out: seen.append((case.id, out.text)) or True)
        ev(j, "the text", make_case("case-9"))
        self.assertEqual(seen, [("case-9", "the text")])

    def test_from_config_loads_callable_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "my_judge.py").write_text("def check(case, output):\n    return {'passed': len(output.text) < 5, 'reasoning': 'short?'}\n")
            (judge,) = build_judges([{"type": "custom", "name": "short", "callable": "my_judge.py:check"}], tmp, PriceTable())
        self.assertEqual(judge.name, "short")
        self.assertTrue(ev(judge, "abc").passed)
        self.assertFalse(ev(judge, "abcdefgh").passed)

    def test_exception_is_recorded_not_raised(self):
        def broken(case, output):
            raise ZeroDivisionError("nope")

        (res,) = run_judges([CustomJudge("c", fn=broken)], make_case(), make_output("x"))
        self.assertFalse(res.passed)
        self.assertEqual(res.error, "ZeroDivisionError: nope")


class ApplicabilityTests(unittest.TestCase):
    def test_when_tags(self):
        j = Contains("c", "x", when_tags=["billing", "refund"])
        self.assertTrue(j.applies(make_case(tags=["billing"])))
        self.assertTrue(j.applies(make_case(tags=["core", "refund"])))
        self.assertFalse(j.applies(make_case(tags=["shipping"])))
        self.assertFalse(j.applies(make_case(tags=[])))

    def test_when_behavior(self):
        j = Contains("c", "x", when_behavior=["answer"])
        self.assertTrue(j.applies(make_case(expected_behavior="answer")))
        self.assertFalse(j.applies(make_case(expected_behavior="abstain")))
        self.assertFalse(j.applies(make_case(expected_behavior=None)))

    def test_both_conditions_must_hold(self):
        j = Contains("c", "x", when_tags=["billing"], when_behavior=["answer"])
        self.assertTrue(j.applies(make_case(tags=["billing"], expected_behavior="answer")))
        self.assertFalse(j.applies(make_case(tags=["billing"], expected_behavior="abstain")))
        self.assertFalse(j.applies(make_case(tags=["shipping"], expected_behavior="answer")))

    def test_no_conditions_applies_everywhere(self):
        self.assertTrue(Contains("c", "x").applies(make_case()))

    def test_run_judges_skips_inapplicable_judges(self):
        judges = [Contains("always", "x"), Contains("billing_only", "x", when_tags=["billing"])]
        res = run_judges(judges, make_case(tags=["shipping"]), make_output("x"))
        self.assertEqual([r.judge for r in res], ["always"])
        res = run_judges(judges, make_case(tags=["billing"]), make_output("x"))
        self.assertEqual([r.judge for r in res], ["always", "billing_only"])


class RunJudgesTests(unittest.TestCase):
    def test_target_error_short_circuits_every_judge(self):
        calls = []
        judges = [
            CustomJudge("spy", fn=lambda c, o: calls.append(1) or True),
            Contains("c", "x"),
        ]
        res = run_judges(judges, make_case(), TargetOutput(text="", error="RuntimeError: upstream down"))
        self.assertEqual(calls, [])  # judge code never ran
        self.assertEqual([r.judge for r in res], ["spy", "c"])
        for r in res:
            self.assertFalse(r.passed)
            self.assertEqual(r.reasoning, "target error")
            self.assertEqual(r.error, "RuntimeError: upstream down")

    def test_target_error_still_respects_applicability(self):
        res = run_judges([Contains("c", "x", when_tags=["a"])], make_case(tags=["b"]), TargetOutput(text="", error="boom"))
        self.assertEqual(res, [])

    def test_order_of_results_follows_judges(self):
        res = run_judges([Contains("z", "x"), Contains("a", "x"), Contains("m", "x")], make_case(), make_output("x"))
        self.assertEqual([r.judge for r in res], ["z", "a", "m"])


class BuildJudgesTests(unittest.TestCase):
    def build(self, cfgs, cases=None):
        return build_judges(cfgs, ".", PriceTable(), cases)

    def test_names_default_to_type(self):
        (a, b) = self.build([{"type": "contains", "values": ["x"]}, {"type": "max_chars", "value": 5}])
        self.assertEqual((a.name, b.name), ("contains", "max_chars"))

    def test_every_heuristic_type_builds(self):
        cfgs = [
            {"type": "contains", "values": ["x"]},
            {"type": "not_contains", "values": ["x"]},
            {"type": "regex", "pattern": "x"},
            {"type": "max_chars", "value": 5},
            {"type": "min_chars", "value": 1},
            {"type": "json_valid"},
            {"type": "exact_match"},
        ]
        judges = self.build(cfgs)
        self.assertEqual([j.name for j in judges], [c["type"] for c in cfgs])

    def test_common_options_reach_the_judge(self):
        (j,) = self.build([{"type": "contains", "name": "n", "values": ["x"], "when_tags": ["a"], "when_behavior": ["answer"]}])
        self.assertEqual(j.when_tags, {"a"})
        self.assertEqual(j.when_behavior, {"answer"})

    def test_unknown_type(self):
        with self.assertRaises(ValueError) as ctx:
            self.build([{"type": "vibes"}])
        self.assertIn("vibes", str(ctx.exception))

    def test_missing_type(self):
        with self.assertRaises(ValueError):
            self.build([{"name": "no-type", "values": ["x"]}])

    def test_duplicate_names(self):
        with self.assertRaises(ValueError) as ctx:
            self.build([{"type": "contains", "values": ["a"]}, {"type": "contains", "values": ["b"]}])
        self.assertIn("duplicate", str(ctx.exception))

    def test_required_option_missing_is_a_type_error(self):
        with self.assertRaises(TypeError):
            self.build([{"type": "contains"}])  # no `values`

    def test_config_dicts_are_not_mutated(self):
        cfg = {"type": "contains", "name": "c", "values": ["x"]}
        self.build([cfg])
        self.assertEqual(cfg, {"type": "contains", "name": "c", "values": ["x"]})

    def test_case_checks_judge_added_only_when_cases_have_checks(self):
        with_checks = [make_case("a"), make_case("b", checks=[{"type": "contains", "values": ["x"]}])]
        without = [make_case("a")]
        self.assertEqual([j.name for j in self.build([], with_checks)], ["case_checks"])
        self.assertEqual(self.build([], without), [])
        self.assertEqual(self.build([], None), [])

    def test_case_checks_judge_appended_after_configured_judges(self):
        cases = [make_case("b", checks=[{"type": "contains", "values": ["x"]}])]
        names = [j.name for j in self.build([{"type": "max_chars", "value": 5}], cases)]
        self.assertEqual(names, ["max_chars", "case_checks"])

    def test_llm_judge_builds_from_inline_rubric(self):
        cfg = {
            "type": "llm",
            "name": "q",
            "rubric": {"criteria": [{"id": "a", "question": "Is it good?"}]},
            "provider": "mock",
            "model": "claude-haiku-4-5",
            "samples": 3,
        }
        (j,) = self.build([cfg])
        self.assertIsInstance(j, LLMJudge)
        self.assertEqual(j.samples, 3)


# --------------------------------------------------------------------------- LLM judge


class Scripted:
    """A mock-provider responder that records calls and replays scripted replies."""

    def __init__(self, replies=None, default=None):
        self.replies = list(replies or [])
        self.default = default
        self.calls: list[dict] = []

    def __call__(self, system, messages, model):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "model": model})
        reply = self.replies.pop(0) if self.replies else self.default
        if reply is None:
            raise AssertionError("scripted responder ran out of replies")
        return reply


def verdicts(**kw):
    """{"a": True, "b": False} -> judge JSON reply text."""
    return json.dumps({"criteria": {cid: {"reasoning": f"why {cid}", "pass": ok} for cid, ok in kw.items()}})


def make_llm_judge(criteria, responder, **kw):
    rubric = Rubric("r", criteria, task=kw.pop("task", "Grade a support bot."), examples=kw.pop("examples", None))
    kw.setdefault("model", "claude-haiku-4-5")
    return LLMJudge("quality", rubric, MockProvider(responder=responder), **kw)


CRITERIA = [
    {"id": "grounded", "question": "Is every fact supported by the reference?"},
    {"id": "complete", "question": "Does it answer the whole question?"},
    {"id": "tone", "question": "Is the tone professional?", "critical": False},
]


class LLMPromptTests(unittest.TestCase):
    def test_system_prompt_contains_every_criterion_and_task(self):
        resp = Scripted(default=verdicts(grounded=True, complete=True, tone=True))
        judge = make_llm_judge(CRITERIA, resp, task="Grade the ACME support bot.")
        ev(judge, "answer")
        system = resp.calls[0]["system"]
        self.assertIn("Grade the ACME support bot.", system)
        for c in CRITERIA:
            self.assertIn(c["id"], system)
            self.assertIn(c["question"], system)
        self.assertIn("grounded [critical]", system)
        self.assertNotIn("tone [critical]", system)
        for cid in ("grounded", "complete", "tone"):  # the JSON shape lists every id
            self.assertIn(f'"{cid}": {{"reasoning"', system)
        self.assertIn("ONLY a JSON object", system)
        self.assertIn("true|false", system)  # nothing is pre-filled with a pass

    def test_user_message_has_tagged_sections_in_order(self):
        resp = Scripted(default=verdicts(grounded=True, complete=True, tone=True))
        judge = make_llm_judge(CRITERIA, resp)
        case = make_case(input="How do I get an invoice?", reference="Settings > Billing", expected_behavior="answer")
        out = make_output("Open Billing.", meta={"context": ["Doc A", "Doc B"]})
        judge.evaluate(case, out)
        msg = resp.calls[0]["messages"][0]["content"]
        nonce = re.search(r"<case_([0-9a-f]+)>", msg).group(1)
        tags = [f"<{name}_{nonce}>" for name in ("input", "expected_behavior", "reference_answer", "context", "ai_output")]
        for tag in tags:
            self.assertIn(tag, msg)
        self.assertNotIn("<input>", msg)  # bare tags would be forgeable
        self.assertIn("How do I get an invoice?", msg)
        self.assertIn(f"<expected_behavior_{nonce}>answer</expected_behavior_{nonce}>", msg)
        self.assertIn("Settings > Billing", msg)
        self.assertIn("Doc A\n\nDoc B", msg)
        self.assertIn(f"<ai_output_{nonce}>\nOpen Billing.\n</ai_output_{nonce}>", msg)
        positions = [msg.index(t) for t in tags]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(resp.calls[0]["messages"][0]["role"], "user")
        # the same nonce is announced in the system prompt
        self.assertIn(f'"_{nonce}"', resp.calls[0]["system"])

    def test_absent_sections_are_omitted(self):
        resp = Scripted(default=verdicts(grounded=True, complete=True, tone=True))
        make_llm_judge(CRITERIA, resp).evaluate(make_case(input="q"), make_output("a"))
        msg = resp.calls[0]["messages"][0]["content"]
        for name in ("reference_answer", "expected_behavior", "context"):
            self.assertNotIn(f"<{name}_", msg)

    def test_dict_input_is_rendered_as_json(self):
        resp = Scripted(default=verdicts(grounded=True, complete=True, tone=True))
        make_llm_judge(CRITERIA, resp).evaluate(make_case(input={"question": "hi", "plan": "pro"}), make_output("a"))
        msg = resp.calls[0]["messages"][0]["content"]
        self.assertIn('"question": "hi"', msg)
        self.assertIn('"plan": "pro"', msg)

    def test_include_restricts_sections_and_can_pull_metadata(self):
        resp = Scripted(default=verdicts(grounded=True, complete=True, tone=True))
        judge = make_llm_judge(CRITERIA, resp, include=["input", "must_include"])
        case = make_case(input="q", reference="SECRET-REF", metadata={"must_include": ["tracking", "email"]})
        judge.evaluate(case, make_output("a", meta={"context": ["CTX"]}))
        msg = resp.calls[0]["messages"][0]["content"]
        self.assertNotIn("SECRET-REF", msg)
        self.assertNotIn("CTX", msg)
        self.assertRegex(msg, r"<must_include_([0-9a-f]+)>\ntracking\n\nemail\n</must_include_\1>")

    def test_custom_context_key(self):
        resp = Scripted(default=verdicts(grounded=True, complete=True, tone=True))
        judge = make_llm_judge(CRITERIA, resp, context_key="docs")
        judge.evaluate(make_case(), make_output("a", meta={"docs": "from docs", "context": "ignored"}))
        msg = resp.calls[0]["messages"][0]["content"]
        self.assertIn("from docs", msg)
        self.assertNotIn("ignored", msg)

    def test_calibration_examples_are_injected_in_system_prompt(self):
        resp = Scripted(default=verdicts(grounded=True, complete=True, tone=True))
        examples = [{"input": "Q1", "output": "A1", "verdict": "fail", "reasoning": "invented a policy"}]
        make_llm_judge(CRITERIA, resp, examples=examples).evaluate(make_case(), make_output("a"))
        system = resp.calls[0]["system"]
        nonce = re.search(r'"_([0-9a-f]+)"', system).group(1)
        self.assertIn(f"<example_input_{nonce}>Q1</example_input_{nonce}>", system)
        self.assertIn(f"<example_output_{nonce}>A1</example_output_{nonce}>", system)
        self.assertIn(f"Human verdict: fail. Why: <example_reasoning_{nonce}>invented a policy</example_reasoning_{nonce}>", system)

    def test_model_and_limits_are_passed_to_the_provider(self):
        resp = Scripted(default=verdicts(grounded=True, complete=True, tone=True))
        make_llm_judge(CRITERIA, resp, model="claude-sonnet-5-5").evaluate(make_case(), make_output("a"))
        self.assertEqual(resp.calls[0]["model"], "claude-sonnet-5-5")


class LLMAggregationTests(unittest.TestCase):
    def judge(self, reply, criteria=CRITERIA, **kw):
        return make_llm_judge(criteria, Scripted(default=reply), **kw)

    def test_all_pass(self):
        r = ev(self.judge(verdicts(grounded=True, complete=True, tone=True)), "a")
        self.assertTrue(r.passed)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.reasoning, "all criteria passed")
        self.assertEqual(r.details["model"], "claude-haiku-4-5")
        self.assertEqual(r.details["criteria"]["grounded"], {"pass": True, "reasoning": "why grounded"})
        self.assertNotIn("self_agreement", r.details)

    def test_critical_failure_fails_the_case(self):
        r = ev(self.judge(verdicts(grounded=False, complete=True, tone=True)), "a")
        self.assertFalse(r.passed)
        self.assertAlmostEqual(r.score, 2 / 3)
        self.assertIn("grounded: why grounded", r.reasoning)
        self.assertNotIn("complete:", r.reasoning)

    def test_non_critical_failure_does_not_fail_the_case_but_lowers_score(self):
        r = ev(self.judge(verdicts(grounded=True, complete=True, tone=False)), "a")
        self.assertTrue(r.passed)
        self.assertAlmostEqual(r.score, 2 / 3)
        self.assertIn("tone", r.reasoning)

    def test_criteria_are_critical_by_default(self):
        criteria = [{"id": "a", "question": "?"}, {"id": "b", "question": "?"}]
        self.assertFalse(ev(self.judge(verdicts(a=True, b=False), criteria), "x").passed)

    def test_weighted_score(self):
        criteria = [{"id": "a", "question": "?", "weight": 3, "critical": False}, {"id": "b", "question": "?", "weight": 1}]
        r = ev(self.judge(verdicts(a=False, b=True), criteria), "x")
        self.assertAlmostEqual(r.score, 0.25)
        r = ev(self.judge(verdicts(a=True, b=False), criteria), "x")
        self.assertAlmostEqual(r.score, 0.75)

    def test_all_non_critical_rubric_requires_every_criterion(self):
        criteria = [{"id": "a", "question": "?", "critical": False}, {"id": "b", "question": "?", "critical": False}]
        self.assertTrue(ev(self.judge(verdicts(a=True, b=True), criteria), "x").passed)
        self.assertFalse(ev(self.judge(verdicts(a=True, b=False), criteria), "x").passed)

    def test_model_supplied_overall_verdict_is_ignored(self):
        reply = json.dumps({"passed": True, "criteria": {c["id"]: {"reasoning": "r", "pass": c["id"] != "grounded"} for c in CRITERIA}})
        self.assertFalse(ev(self.judge(reply), "a").passed)

    def test_string_verdicts_are_understood(self):
        reply = json.dumps({"criteria": {"grounded": {"reasoning": "r", "pass": "yes"}, "complete": {"reasoning": "r", "pass": "false"}, "tone": {"reasoning": "r", "pass": "PASS"}}})
        r = ev(self.judge(reply), "a")
        self.assertEqual([r.details["criteria"][c]["pass"] for c in ("grounded", "complete", "tone")], [True, False, True])

    def test_reply_may_omit_the_criteria_wrapper(self):
        flat = json.dumps({c["id"]: {"reasoning": "r", "pass": True} for c in CRITERIA})
        self.assertTrue(ev(self.judge(flat), "a").passed)

    def test_fenced_and_chatty_replies_are_parsed(self):
        body = verdicts(grounded=True, complete=True, tone=True)
        self.assertTrue(ev(self.judge(f"```json\n{body}\n```"), "a").passed)
        self.assertTrue(ev(self.judge(f"Sure! Here is my evaluation:\n{body}\nHope that helps."), "a").passed)

    def test_cost_and_usage_come_from_the_judge_call(self):
        reply = {"text": verdicts(grounded=True, complete=True, tone=True), "usage": {"input_tokens": 1000, "output_tokens": 100}}
        r = ev(self.judge(reply), "a")
        self.assertEqual(r.usage, Usage(1000, 100))
        self.assertAlmostEqual(r.cost_usd, (1000 * 1.0 + 100 * 5.0) / 1e6)  # claude-haiku-4-5 pricing

    def test_unknown_judge_model_costs_zero_not_crash(self):
        judge = make_llm_judge(CRITERIA, Scripted(default=verdicts(grounded=True, complete=True, tone=True)), model="mystery-judge-1")
        with contextlib.redirect_stderr(io.StringIO()):
            r = ev(judge, "a")
        self.assertEqual(r.cost_usd, 0.0)


class LLMParseRetryTests(unittest.TestCase):
    GOOD = verdicts(grounded=True, complete=True, tone=True)

    def test_malformed_json_then_success(self):
        resp = Scripted(
            replies=[
                {"text": "I think it passes!", "usage": {"input_tokens": 100, "output_tokens": 10}},
                {"text": self.GOOD, "usage": {"input_tokens": 150, "output_tokens": 30}},
            ]
        )
        r = ev(make_llm_judge(CRITERIA, resp), "a")
        self.assertTrue(r.passed)
        self.assertEqual(len(resp.calls), 2)
        retry = resp.calls[1]["messages"]
        self.assertEqual([m["role"] for m in retry], ["user", "assistant", "user"])
        self.assertEqual(retry[1]["content"], "I think it passes!")
        self.assertIn("ONLY the JSON", retry[2]["content"])
        self.assertEqual(r.usage, Usage(250, 40))  # both attempts are billed
        self.assertAlmostEqual(r.cost_usd, (250 * 1.0 + 40 * 5.0) / 1e6)

    def test_missing_criterion_triggers_retry(self):
        partial = verdicts(grounded=True, complete=True)  # tone missing
        resp = Scripted(replies=[partial, self.GOOD])
        self.assertTrue(ev(make_llm_judge(CRITERIA, resp), "a").passed)
        self.assertEqual(len(resp.calls), 2)

    def test_wrong_verdict_key_triggers_retry_instead_of_a_silent_fail(self):
        # regression: {"passed": true} used to be read as pass=None -> a confident but wrong FAIL, no retry
        wrong_key = json.dumps({"criteria": {c["id"]: {"reasoning": "r", "passed": True} for c in CRITERIA}})
        resp = Scripted(replies=[wrong_key, self.GOOD])
        r = ev(make_llm_judge(CRITERIA, resp), "a")
        self.assertTrue(r.passed)
        self.assertEqual(len(resp.calls), 2)
        final = ev(make_llm_judge(CRITERIA, Scripted(default=wrong_key)), "a")  # never right: an error, not a silent fail
        self.assertIsNotNone(final.error)
        self.assertIsNone(final.score)

    def test_null_verdict_is_a_parse_retry_not_a_silent_fail(self):
        null = json.dumps({"criteria": {c["id"]: {"reasoning": "r", "pass": None} for c in CRITERIA}})
        resp = Scripted(replies=[null, self.GOOD])
        self.assertTrue(ev(make_llm_judge(CRITERIA, resp), "a").passed)
        self.assertEqual(len(resp.calls), 2)
        self.assertIsNotNone(ev(make_llm_judge(CRITERIA, Scripted(default=null)), "a").error)

    def test_criterion_of_wrong_shape_triggers_retry(self):
        bad = json.dumps({"criteria": {"grounded": True, "complete": True, "tone": True}})  # not objects
        resp = Scripted(replies=[bad, self.GOOD])
        self.assertTrue(ev(make_llm_judge(CRITERIA, resp), "a").passed)

    def test_still_malformed_is_an_error_result_and_run_judges_keeps_it(self):
        resp = Scripted(default="never valid json")
        judge = make_llm_judge(CRITERIA, resp)
        r = ev(judge, "a")
        self.assertFalse(r.passed)
        self.assertIn("could not be parsed", r.error)
        self.assertEqual(len(resp.calls), 2)  # first try + max_parse_retries (1)

        (res,) = run_judges([judge], make_case(), make_output("a"))
        self.assertFalse(res.passed)
        self.assertIn("could not be parsed", res.error)
        self.assertEqual(res.judge, "quality")

    def test_max_parse_retries_is_configurable(self):
        resp = Scripted(default="nope")
        self.assertIsNotNone(ev(make_llm_judge(CRITERIA, resp, max_parse_retries=0), "a").error)
        self.assertEqual(len(resp.calls), 1)
        resp = Scripted(default="nope")
        self.assertIsNotNone(ev(make_llm_judge(CRITERIA, resp, max_parse_retries=3), "a").error)
        self.assertEqual(len(resp.calls), 4)

    def test_provider_exception_is_recorded_by_run_judges(self):
        def down(system, messages, model):
            raise ConnectionError("API unreachable")

        (res,) = run_judges([make_llm_judge(CRITERIA, down)], make_case(), make_output("a"))
        self.assertFalse(res.passed)
        self.assertIn("ConnectionError", res.error)


class LLMSelfConsistencyTests(unittest.TestCase):
    CRIT = [{"id": "a", "question": "?"}, {"id": "b", "question": "?"}]

    def test_majority_vote_and_self_agreement(self):
        resp = Scripted(
            replies=[
                verdicts(a=True, b=True),
                verdicts(a=True, b=False),
                verdicts(a=False, b=False),
            ]
        )
        r = ev(make_llm_judge(self.CRIT, resp, samples=3), "x")
        self.assertEqual(len(resp.calls), 3)
        self.assertTrue(r.details["criteria"]["a"]["pass"])  # 2 of 3
        self.assertFalse(r.details["criteria"]["b"]["pass"])  # 2 of 3 fail
        self.assertAlmostEqual(r.details["self_agreement"], 2 / 3)
        self.assertFalse(r.passed)  # b is critical and lost the vote
        self.assertAlmostEqual(r.score, 0.5)

    def test_unanimous_samples_have_full_agreement(self):
        resp = Scripted(default=verdicts(a=True, b=True))
        r = ev(make_llm_judge(self.CRIT, resp, samples=3), "x")
        self.assertEqual(r.details["self_agreement"], 1.0)
        self.assertTrue(r.passed)

    def test_agreement_is_averaged_over_criteria(self):
        resp = Scripted(replies=[verdicts(a=True, b=True), verdicts(a=True, b=True), verdicts(a=True, b=False)])
        r = ev(make_llm_judge(self.CRIT, resp, samples=3), "x")
        self.assertAlmostEqual(r.details["self_agreement"], (1.0 + 2 / 3) / 2)

    def test_reasoning_comes_from_a_sample_that_agrees_with_the_majority(self):
        def reply(reason, ok):
            return json.dumps({"criteria": {"a": {"reasoning": reason, "pass": ok}, "b": {"reasoning": "b", "pass": True}}})

        resp = Scripted(replies=[reply("outlier says fail", False), reply("majority says pass 1", True), reply("majority says pass 2", True)])
        r = ev(make_llm_judge(self.CRIT, resp, samples=3), "x")
        self.assertEqual(r.details["criteria"]["a"]["reasoning"], "majority says pass 1")

    def test_cost_and_usage_sum_over_samples(self):
        reply = {"text": verdicts(a=True, b=True), "usage": {"input_tokens": 200, "output_tokens": 20}}
        r = ev(make_llm_judge(self.CRIT, Scripted(default=reply), samples=3), "x")
        self.assertEqual(r.usage, Usage(600, 60))
        self.assertAlmostEqual(r.cost_usd, 3 * (200 * 1.0 + 20 * 5.0) / 1e6)

    def test_every_sample_sees_the_same_prompt(self):
        resp = Scripted(default=verdicts(a=True, b=True))
        ev(make_llm_judge(self.CRIT, resp, samples=3), "x")
        self.assertEqual(len({json.dumps(c["messages"]) for c in resp.calls}), 1)

    def test_a_sample_needing_a_reparse_still_counts_once(self):
        resp = Scripted(replies=[verdicts(a=True, b=True), "garbage", verdicts(a=True, b=True), verdicts(a=True, b=True)])
        r = ev(make_llm_judge(self.CRIT, resp, samples=3), "x")
        self.assertEqual(len(resp.calls), 4)
        self.assertEqual(r.details["self_agreement"], 1.0)

    def test_invalid_sample_counts_are_clamped(self):
        for n in (0, -2):
            self.assertEqual(make_llm_judge(self.CRIT, Scripted(), samples=n).samples, 1)


class RubricTests(unittest.TestCase):
    def test_defaults_are_filled_in(self):
        rubric = Rubric("r", [{"id": "a", "question": "?"}])
        self.assertEqual(rubric.criteria[0]["critical"], True)
        self.assertEqual(rubric.criteria[0]["weight"], 1.0)
        self.assertEqual(rubric.ids, ["a"])

    def test_validation(self):
        with self.assertRaises(ValueError):
            Rubric("r", [])
        with self.assertRaises(ValueError):
            Rubric("r", [{"id": "a"}])
        with self.assertRaises(ValueError):
            Rubric("r", [{"question": "?"}])

    def test_load_from_yaml_file_relative_to_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rubric.yaml").write_text(
                "name: support\ntask: Grade it.\ncriteria:\n  - {id: a, question: 'Is a true?'}\n  - {id: b, question: 'Is b true?', critical: false, weight: 2}\n"
            )
            rubric = Rubric.load("rubric.yaml", tmp)
        self.assertEqual(rubric.name, "support")
        self.assertEqual(rubric.ids, ["a", "b"])
        self.assertEqual(rubric.criteria[1]["weight"], 2)
        self.assertIn("Grade it.", rubric.system_prompt())

    def test_load_from_dict(self):
        rubric = Rubric.load({"name": "inline", "criteria": [{"id": "a", "question": "?"}]})
        self.assertEqual(rubric.name, "inline")

    def test_from_config_wires_mock_provider_responder_relative_to_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "rubric.yaml").write_text("criteria:\n  - {id: a, question: '?'}\n")
            (tmp / "sim.py").write_text(
                "import json\n"
                "def judge(system, messages, model):\n"
                "    return json.dumps({'criteria': {'a': {'reasoning': 'sim', 'pass': True}}})\n"
            )
            cfg = {"type": "llm", "name": "q", "rubric": "rubric.yaml", "provider": "mock", "provider_options": {"responder": "sim.py:judge"}, "model": "claude-haiku-4-5"}
            (judge,) = build_judges([cfg], tmp, PriceTable())
            r = ev(judge, "x")
        self.assertTrue(r.passed)
        self.assertEqual(r.details["criteria"]["a"]["reasoning"], "sim")

    def test_from_config_requires_model(self):
        cfg = {"type": "llm", "rubric": {"criteria": [{"id": "a", "question": "?"}]}, "provider": "mock"}
        with self.assertRaises(ValueError) as ctx:
            build_judges([cfg], ".", PriceTable())
        self.assertIn("model", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # the judge class enforces it too, not only build_judges
            LLMJudge.from_config({k: v for k, v in cfg.items() if k != "type"})
        self.assertIn("needs a `model`", str(ctx.exception))


class ExtractJsonTests(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('prefix {"a": {"b": 2}} suffix'), {"a": {"b": 2}})
        self.assertEqual(extract_json('  \n{"a": 1}\n  '), {"a": 1})

    def test_no_json_raises(self):
        for bad in ("", "no braces here", "} backwards {", None):
            with self.subTest(bad=bad):
                with self.assertRaises(json.JSONDecodeError):
                    extract_json(bad)


class BaseJudgeTests(unittest.TestCase):
    def test_evaluate_not_implemented(self):
        from evalsprint.judges import Judge

        with self.assertRaises(NotImplementedError):
            Judge("x").evaluate(make_case(), make_output())

    def test_judge_result_defaults(self):
        r = JudgeResult("j", True)
        self.assertEqual((r.score, r.reasoning, r.details, r.cost_usd, r.error), (None, "", {}, 0.0, None))


if __name__ == "__main__":
    unittest.main()
