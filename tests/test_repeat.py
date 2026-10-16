"""Repeats and the noise floor (contracts C2, C17): `repeat: {n, per_tag}` in eval.yaml (and the old `{n, tags}` spelling), `run --repeat N`,
attempts / stability / majority on the record, `params["_attempt"]` for the target, `summary.stability`."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from evalsprint.cli import main
from evalsprint.config import SuiteConfig
from evalsprint.runner import RunResult, _combine_attempts, load_run, majority_outcome, run_suite, summarize
from evalsprint.types import Case, CaseResult, JudgeResult, TargetOutput
from tests import FIXTURES, make_record

REPEAT_DIR = FIXTURES / "repeat"


class MajorityRuleTests(unittest.TestCase):
    def test_table(self):
        P, F, E = "pass", "fail", "error"
        table = [
            ([P], P), ([F], F), ([E], E),
            ([P, P, F], P), ([F, F, P], F), ([P, P, P], P), ([F, F, F], F),
            ([P, F], F),  # tie -> fail (conservative)
            ([P, P, F, F], F),
            ([P, P, P, F], P),
            ([P, E, P], P),  # 1 error < half of 3
            ([P, E], E),  # errors == half -> error
            ([E, E, P], E),
            ([E, F, F], F),  # 1 error of 3 is below half
            ([P, F, E, E], E),  # 2 of 4 = half
            ([P, P, E, F], P),  # 1 of 4 is below half; 2 pass > 1 fail
            ([P, F, E], F),  # scored attempts tie -> fail
            ([E, E, E], E),
        ]
        for outcomes, expected in table:
            with self.subTest(outcomes=outcomes):
                self.assertEqual(majority_outcome(outcomes), expected)


def result(outcome: str, text: str = "t", judge_cost: float = 0.0, cost: float | None = None) -> CaseResult:
    """A CaseResult whose resolved outcome is `outcome` (pass/fail/error-from-target)."""
    output = TargetOutput(text=text, cost_usd=cost, error="boom" if outcome == "error" else None)
    return CaseResult(Case(id="c", input="q"), output, [JudgeResult("j", outcome == "pass", cost_usd=judge_cost)], outcome == "pass")


class CombineAttemptsTests(unittest.TestCase):
    def test_single_attempt_is_returned_untouched(self):
        only = result("pass")
        self.assertIs(_combine_attempts([only]), only)
        self.assertNotIn("attempts", only.to_dict())

    def test_first_attempt_matching_the_majority_represents_the_case(self):
        a, b, c = result("fail", "a"), result("pass", "b"), result("pass", "c")
        rec = _combine_attempts([a, b, c]).to_dict()
        self.assertEqual((rec["outcome"], rec["passed"], rec["output"]["text"]), ("pass", True, "b"))
        self.assertAlmostEqual(rec["stability"], 2 / 3)
        self.assertEqual([x["outcome"] for x in rec["attempts"]], ["fail", "pass", "pass"])
        self.assertEqual([x["passed"] for x in rec["attempts"]], [False, True, True])
        self.assertEqual([x["output_sha"] for x in rec["attempts"]], [TargetOutput(text=t).sha for t in "abc"])

    def test_tie_is_a_fail_and_errors_at_half_are_an_error(self):
        tie = _combine_attempts([result("pass", "a"), result("fail", "b")]).to_dict()
        self.assertEqual((tie["outcome"], tie["passed"], tie["output"]["text"], tie["stability"]), ("fail", False, "b", 0.5))
        err = _combine_attempts([result("pass", "a"), result("error", "b")]).to_dict()
        self.assertEqual((err["outcome"], err["error_source"], err["passed"], err["stability"]), ("error", "target", False, 0.5))
        self.assertEqual(err["output"]["error"], "boom")  # the errored attempt represents the case

    def test_attempt_entries_carry_the_spend_of_every_attempt(self):
        rec = _combine_attempts([result("pass", judge_cost=0.25, cost=1.0), result("fail", judge_cost=0.5, cost=2.0)]).to_dict()
        self.assertEqual([a["judge_cost_usd"] for a in rec["attempts"]], [0.25, 0.5])
        self.assertEqual([a["cost_usd"] for a in rec["attempts"]], [1.0, 2.0])


class RepeatFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "suite"
        shutil.copytree(REPEAT_DIR, self.dir, ignore=shutil.ignore_patterns("__pycache__"))

    def config(self, overrides=None) -> SuiteConfig:
        return SuiteConfig.load(self.dir / "eval.yaml", overrides)

    def do_run(self, overrides=None, **kw) -> RunResult:
        kw.setdefault("progress", False)
        kw.setdefault("out_dir", self.dir / "out")
        return run_suite(self.config(overrides), **kw)


class ConfigRepeatTests(RepeatFixture):
    def test_the_old_tags_spelling_repeats_the_listed_tags_only(self):
        run = self.do_run(["repeat={n: 3, tags: [safety]}"])
        by = run.by_case()
        for cid in ("mostly-good", "late-good", "mostly-bad"):
            self.assertEqual(len(by[cid]["attempts"]), 3, cid)
        for cid in ("always-good", "always-bad", "tie", "one-error", "mostly-error", "half-error"):
            self.assertNotIn("attempts", by[cid], cid)
            self.assertNotIn("stability", by[cid], cid)
        self.assertEqual(run.summary["repeat"], {"n": 1, "per_tag": {"safety": 3}})  # {n: 3, tags: [safety]} is {n: 1, per_tag: {safety: 3}}

    def test_per_tag_repeats_the_listed_tags_more_than_the_rest(self):
        """C17: a case runs the largest per_tag count among its tags, else n."""
        run = self.do_run(["repeat={n: 2, per_tag: {safety: 4}}"])
        by = run.by_case()
        for cid in ("mostly-good", "late-good", "mostly-bad"):
            self.assertEqual(len(by[cid]["attempts"]), 4, cid)
        for cid in ("always-good", "always-bad", "tie", "one-error", "mostly-error", "half-error"):
            self.assertEqual(len(by[cid]["attempts"]), 2, cid)
        self.assertEqual(run.summary["repeat"], {"n": 2, "per_tag": {"safety": 4}})
        self.assertEqual(run.summary["n_attempts"], 3 * 4 + 6 * 2)
        self.assertEqual(run.summary["stability"]["n_repeated"], 9)

    def test_the_largest_per_tag_count_among_a_cases_tags_wins(self):
        run = self.do_run(["repeat={n: 2, per_tag: {safety: 4, stable: 3}}"], ids=["always-good", "mostly-good"])
        by = run.by_case()
        self.assertEqual(len(by["mostly-good"]["attempts"]), 4)  # safety 4
        self.assertEqual(len(by["always-good"]["attempts"]), 3)  # stable 3, not n = 2
        self.assertEqual(run.summary["repeat"], {"n": 2, "per_tag": {"safety": 4, "stable": 3}})

    def test_a_per_tag_count_below_n_lowers_the_attempts_of_its_cases(self):
        by = self.do_run(["repeat={n: 3, per_tag: {stable: 1}}"]).by_case()
        self.assertNotIn("attempts", by["always-good"])  # a count of 1: the case runs once
        self.assertEqual(len(by["mostly-good"]["attempts"]), 3)

    def test_without_per_tag_every_case_is_repeated(self):
        run = self.do_run(["repeat.n=2"])
        self.assertTrue(all(len(r["attempts"]) == 2 for r in run.results))
        self.assertEqual(run.summary["repeat"], {"n": 2, "per_tag": {}})
        self.assertEqual(run.summary["stability"]["n_repeated"], 9)

    def test_verdicts_follow_the_majority_rule(self):
        by = self.do_run(["repeat.n=3"]).by_case()
        expected = {  # attempts per case are the fixture's scripts, wrapped around to 3 attempts
            "always-good": ("pass", 1.0),  # good good good
            "always-bad": ("fail", 1.0),
            "mostly-good": ("pass", 2 / 3),  # good bad good
            "late-good": ("pass", 2 / 3),  # bad good good
            "mostly-bad": ("fail", 2 / 3),  # bad good bad
            "tie": ("pass", 2 / 3),  # good bad good
            "one-error": ("pass", 2 / 3),  # good good boom: 1 error of 3 is below half
            "mostly-error": ("error", 2 / 3),  # good boom boom: 2 errors of 3
            "half-error": ("pass", 2 / 3),  # good boom good: 1 error of 3
        }
        for cid, (outcome, stability) in expected.items():
            with self.subTest(case=cid):
                self.assertEqual(by[cid]["outcome"], outcome)
                self.assertEqual(by[cid]["passed"], outcome == "pass")
                self.assertAlmostEqual(by[cid]["stability"], stability)

    def test_two_attempts_ties_and_half_errors(self):
        by = self.do_run(["repeat.n=2"]).by_case()
        self.assertEqual(by["tie"]["outcome"], "fail")  # good, bad -> tie -> fail
        self.assertEqual(by["tie"]["stability"], 0.5)
        self.assertEqual((by["half-error"]["outcome"], by["half-error"]["error_source"]), ("error", "target"))  # good, boom -> errors == half
        self.assertEqual(by["one-error"]["outcome"], "pass")  # good, good
        self.assertEqual(by["mostly-good"]["outcome"], "fail")  # good, bad -> tie -> fail

    def test_top_level_output_comes_from_the_first_attempt_with_the_majority_outcome(self):
        by = self.do_run(["repeat.n=3"]).by_case()
        self.assertEqual(by["late-good"]["output"]["meta"]["attempt"], 1)  # bad, good, good -> first passing attempt is #1
        self.assertEqual(by["mostly-good"]["output"]["meta"]["attempt"], 0)
        self.assertEqual(by["mostly-bad"]["output"]["meta"]["attempt"], 0)  # bad, good, bad -> first failing is #0
        self.assertEqual([j["passed"] for j in by["late-good"]["judges"]], [True])  # judges of that same attempt
        self.assertEqual(by["late-good"]["output"]["sha"], by["late-good"]["attempts"][1]["output_sha"])
        self.assertNotEqual(by["late-good"]["attempts"][0]["output_sha"], by["late-good"]["attempts"][1]["output_sha"])

    def test_the_attempt_index_is_passed_to_the_target_as_a_param(self):
        log = self.dir / "calls.log"
        (self.dir / "target.py").write_text(
            "def answer(input, params):\n"
            f"    with open({str(log)!r}, 'a', encoding='utf-8') as fh:\n"
            "        fh.write(f\"{input} {params.get('_attempt', 'absent')}\\n\")\n"
            "    return 'good'\n",
            encoding="utf-8",
        )
        self.do_run(["repeat={n: 1, per_tag: {safety: 3}}"])
        calls = [line.split() for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sorted(a for cid, a in calls if cid == "mostly-good"), ["0", "1", "2"])  # 0-based
        self.assertEqual([a for cid, a in calls if cid == "always-good"], ["absent"])  # not repeated: no `_attempt`

    def test_records_round_trip_through_disk(self):
        run = self.do_run(["repeat={n: 3, tags: [safety]}"])
        loaded = load_run(run.run_dir)
        self.assertEqual(loaded.results, run.results)
        self.assertEqual(loaded.by_case()["late-good"]["attempts"], run.by_case()["late-good"]["attempts"])
        self.assertFalse((run.run_dir / "results.partial.jsonl").exists())

    def test_progress_marks_each_attempt(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.do_run(["repeat={n: 1, per_tag: {safety: 3}}"], progress=True)
        text = err.getvalue()
        self.assertIn("[attempt 1/3]", text)
        self.assertIn("[attempt 3/3]", text)
        self.assertRegex(text, r"\[\s*\d+/15\]")  # 6 single-attempt cases + 3 safety cases x 3 attempts
        self.assertNotIn("always-good [attempt", text)

    def test_repeat_tags_matching_nothing_warns_and_runs_once(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            run = self.do_run(["repeat={n: 1, per_tag: {nope: 3}}"])
        self.assertIn("match no selected case", err.getvalue())
        self.assertTrue(all("attempts" not in r for r in run.results))
        self.assertEqual(run.summary["stability"], {"n_repeated": 0, "flip_rate": None})


class OverrideTests(RepeatFixture):
    def test_the_int_override_sets_n_and_keeps_the_per_tag_counts(self):
        """C17: `run --repeat N` sets `n` and keeps `per_tag`."""
        run = self.do_run(["repeat={n: 3, per_tag: {safety: 5}}"], repeat=2)
        by = run.by_case()
        self.assertEqual({len(r["attempts"]) for cid, r in by.items() if cid not in ("mostly-good", "late-good", "mostly-bad")}, {2})
        self.assertEqual({len(by[cid]["attempts"]) for cid in ("mostly-good", "late-good", "mostly-bad")}, {5})
        self.assertEqual(run.summary["repeat"], {"n": 2, "per_tag": {"safety": 5}})

    def test_the_int_override_turns_a_config_without_per_tag_into_n_on_every_case(self):
        run = self.do_run(["repeat={n: 3}"], repeat=2)
        self.assertTrue(all(len(r["attempts"]) == 2 for r in run.results))
        self.assertEqual(run.summary["repeat"], {"n": 2, "per_tag": {}})

    def test_override_of_one_switches_repetition_off(self):
        run = self.do_run(["repeat={n: 3}"], repeat=1)
        self.assertTrue(all("attempts" not in r for r in run.results))
        self.assertEqual(run.summary["repeat"], {"n": 1, "per_tag": {}})  # C11: always the effective protocol, n == 1 = every case once

    def test_override_of_one_keeps_the_per_tag_cases_repeated(self):
        run = self.do_run(["repeat={n: 3, per_tag: {safety: 3}}"], repeat=1)
        self.assertEqual({r["case_id"] for r in run.results if "attempts" in r}, {"mostly-good", "late-good", "mostly-bad"})
        self.assertEqual(run.summary["repeat"], {"n": 1, "per_tag": {"safety": 3}})

    def test_override_accepts_a_dict_that_replaces_the_whole_protocol(self):
        run = self.do_run(["repeat={n: 3, per_tag: {safety: 5}}"], repeat={"n": 3, "tags": ["stable"]})
        self.assertEqual({r["case_id"] for r in run.results if "attempts" in r}, {"always-good", "always-bad"})
        self.assertEqual(run.summary["repeat"], {"n": 1, "per_tag": {"stable": 3}})
        run = self.do_run(repeat={"per_tag": {"stable": 2}})
        self.assertEqual(run.summary["repeat"], {"n": 1, "per_tag": {"stable": 2}})

    def test_invalid_override(self):
        with self.assertRaises(ValueError):
            self.do_run(repeat=0)
        with self.assertRaises(ValueError):
            self.do_run(repeat=500)

    def test_cli_repeat_flag(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["run", "-c", str(self.dir / "eval.yaml"), "--out", str(self.dir / "cli"), "--quiet", "--repeat", "3"])
        self.assertEqual(code, 0, err.getvalue())  # the fixture's gate allows errors (max_error_rate: 1.0)
        summary = json.loads((self.dir / "cli" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["repeat"], {"n": 3, "per_tag": {}})
        self.assertEqual(summary["stability"]["n_repeated"], 9)
        self.assertIn("Stability:", out.getvalue())
        self.assertIn("flip rate", out.getvalue())

    def test_cli_repeat_must_be_positive(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["run", "-c", str(self.dir / "eval.yaml"), "--repeat", "0"]), 2)
            self.assertEqual(main(["run", "-c", str(self.dir / "eval.yaml"), "--repeat", "x"]), 2)
        self.assertIn("--repeat", err.getvalue())


class ProtocolInSummaryTests(RepeatFixture):
    """C11, C17: summary.repeat is the EFFECTIVE protocol `{"n", "per_tag"}` (config and --repeat merged), keys sorted, always present."""

    def test_per_tag_keys_are_stored_sorted(self):
        run = self.do_run(["repeat={n: 3, per_tag: {stable: 4, flaky: 5, safety: 2}}"])
        self.assertEqual(list(run.summary["repeat"]["per_tag"]), ["flaky", "safety", "stable"])
        self.assertEqual(run.summary["repeat"], {"n": 3, "per_tag": {"flaky": 5, "safety": 2, "stable": 4}})

    def test_the_old_tags_spelling_is_normalised_in_the_summary(self):
        run = self.do_run(["repeat={n: 3, tags: [stable, flaky, safety]}"])
        self.assertEqual(run.summary["repeat"], {"n": 1, "per_tag": {"flaky": 3, "safety": 3, "stable": 3}})

    def test_no_repeat_is_n_one_and_per_tag_is_written_even_then(self):
        run = self.do_run()
        self.assertEqual(run.summary["repeat"], {"n": 1, "per_tag": {}})
        self.assertEqual(json.loads((self.dir / "out" / "summary.json").read_text(encoding="utf-8"))["repeat"], {"n": 1, "per_tag": {}})

    def test_the_cli_override_sets_n_and_keeps_per_tag(self):
        run = self.do_run(["repeat={n: 3, tags: [safety]}"], repeat=2)
        self.assertEqual(run.summary["repeat"], {"n": 2, "per_tag": {"safety": 3}})

    def test_a_dict_override_keeps_its_per_tag_sorted(self):
        run = self.do_run(repeat={"n": 2, "tags": ["stable", "core"]})
        self.assertEqual(run.summary["repeat"], {"n": 1, "per_tag": {"core": 2, "stable": 2}})

    def test_the_declared_protocol_survives_a_filter_that_matches_no_repeated_case(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):  # the "matches no selected case" warning
            run = self.do_run(["repeat={n: 1, per_tag: {safety: 3}}"], ids=["always-good"])
        self.assertIn("match no selected case", err.getvalue())
        self.assertTrue(all("attempts" not in r for r in run.results))
        self.assertEqual(run.summary["repeat"], {"n": 1, "per_tag": {"safety": 3}})  # what the suite declares, so two runs of it compare as one protocol


class AttemptErrorTests(RepeatFixture):
    """C10: a crash in ONE attempt of a repeated case is invisible in `outcome` when the majority passes."""

    def test_records_count_their_errored_attempts(self):
        by = self.do_run(["repeat.n=3"]).by_case()
        self.assertEqual((by["one-error"]["outcome"], by["one-error"]["n_attempt_errors"]), ("pass", 1))  # good good boom: hidden by the majority
        self.assertEqual((by["mostly-error"]["outcome"], by["mostly-error"]["n_attempt_errors"]), ("error", 2))
        self.assertEqual((by["half-error"]["outcome"], by["half-error"]["n_attempt_errors"]), ("pass", 1))  # good boom good
        self.assertEqual(by["always-good"]["n_attempt_errors"], 0)
        self.assertEqual(by["mostly-good"]["n_attempt_errors"], 0)  # a FAILED attempt is not an errored one

    def test_a_case_that_ran_once_counts_zero_or_one(self):
        by = self.do_run().by_case()
        self.assertTrue(all(r["n_attempt_errors"] == int(r["outcome"] == "error") for r in by.values()))
        self.assertEqual(sum(r["n_attempt_errors"] for r in by.values()), 0)  # without `_attempt` every scripted case behaves as attempt 0: none crashes

    def test_summary_counters(self):
        s = self.do_run(["repeat.n=3"]).summary
        self.assertEqual(s["n_attempts"], 27)  # 9 cases x 3
        self.assertEqual(s["n_attempt_errors"], 4)  # one-error 1, mostly-error 2, half-error 1
        self.assertEqual((s["n_attempt_target_errors"], s["n_attempt_judge_errors"]), (4, 0))
        self.assertAlmostEqual(s["attempt_error_rate"], 4 / 27)
        self.assertEqual((s["n_errors"], s["n_target_errors"]), (1, 1))  # the case-level counters see only mostly-error
        self.assertAlmostEqual(s["error_rate"], 1 / 9)

    def test_a_run_without_repeats_has_attempt_counters_equal_to_case_counters(self):
        s = self.do_run().summary
        self.assertEqual((s["n_attempts"], s["n_attempt_errors"], s["n_attempt_target_errors"]), (s["n_cases"], s["n_errors"], s["n_target_errors"]))
        self.assertAlmostEqual(s["attempt_error_rate"], s["error_rate"])

    def test_counters_survive_the_save_and_load_round_trip(self):
        run = self.do_run(["repeat.n=3"])
        loaded = load_run(self.dir / "out")
        self.assertEqual(loaded.summary["n_attempt_errors"], 4)
        self.assertEqual(sum(r["n_attempt_errors"] for r in loaded.results), 4)


class AttemptCountersOfSummarizeTests(unittest.TestCase):
    """`summarize` on hand-built records: the counters do not depend on the runner."""

    @staticmethod
    def repeated(cid, attempts, outcome):
        rec = make_record(cid, outcome == "pass")
        rec["outcome"] = outcome
        rec["error_source"] = "target" if outcome == "error" else None
        rec["passed"] = outcome == "pass"
        rec["attempts"] = [{"outcome": o, "passed": o == "pass", "output_sha": "x", **({"error_source": src} if src else {})} for o, src in attempts]
        return rec

    def test_target_and_judge_attempt_errors_are_counted_apart(self):
        records = [
            self.repeated("a", [("pass", None), ("error", "target"), ("pass", None)], "pass"),
            self.repeated("b", [("pass", None), ("error", "judge"), ("pass", None)], "pass"),
            self.repeated("c", [("error", "target"), ("error", "judge")], "error"),
            make_record("d", True),
        ]
        s = summarize(records)
        self.assertEqual(s["n_attempts"], 3 + 3 + 2 + 1)
        self.assertEqual((s["n_attempt_errors"], s["n_attempt_target_errors"], s["n_attempt_judge_errors"]), (4, 2, 2))
        self.assertAlmostEqual(s["attempt_error_rate"], 4 / 9)
        self.assertEqual((s["n_errors"], s["n_passed"]), (1, 3))

    def test_an_errored_attempt_without_a_source_counts_as_a_judge_error(self):
        s = summarize([self.repeated("a", [("pass", None), ("error", None)], "error")])
        self.assertEqual((s["n_attempt_target_errors"], s["n_attempt_judge_errors"]), (0, 1))  # the conservative reading, as for a case

    def test_single_attempt_errors_use_the_case_error_source(self):
        errored = make_record("t", False, error="boom")
        s = summarize([errored, make_record("ok", True)])
        self.assertEqual((s["n_attempts"], s["n_attempt_errors"], s["n_attempt_target_errors"], s["n_attempt_judge_errors"]), (2, 1, 1, 0))

    def test_nothing_to_summarise(self):
        s = summarize([])
        self.assertEqual((s["n_attempts"], s["n_attempt_errors"], s["attempt_error_rate"]), (0, 0, 0.0))


class StabilitySummaryTests(unittest.TestCase):
    def test_flip_rate_is_the_share_of_repeated_cases_whose_attempts_disagree(self):
        def repeated(cid, outcomes):
            rec = make_record(cid, outcomes[0] == "pass")
            rec["outcome"] = max(set(outcomes), key=outcomes.count)
            rec["passed"] = rec["outcome"] == "pass"
            rec["attempts"] = [{"outcome": o, "passed": o == "pass", "output_sha": "x"} for o in outcomes]
            return rec

        records = [
            repeated("steady", ["pass", "pass", "pass"]),
            repeated("flips", ["pass", "fail", "pass"]),
            repeated("errors", ["pass", "error", "pass"]),
            repeated("steady-fail", ["fail", "fail"]),
            make_record("single", True),
        ]
        stability = summarize(records)["stability"]
        self.assertEqual(stability["n_repeated"], 4)
        self.assertEqual(stability["flip_rate"], 0.5)  # `flips` and `errors` disagree, the other two do not

    def test_nothing_repeated_gives_none(self):
        self.assertEqual(summarize([make_record("a", True)])["stability"], {"n_repeated": 0, "flip_rate": None})
        self.assertEqual(summarize([])["stability"], {"n_repeated": 0, "flip_rate": None})

    def test_judge_spend_counts_every_attempt_but_system_cost_is_per_request(self):
        rec = make_record("a", True, cost=1.0)
        rec["judges"][0]["cost_usd"] = 0.25  # the representative attempt's judges
        rec["attempts"] = [
            {"outcome": "pass", "passed": True, "output_sha": "x", "cost_usd": 1.0, "judge_cost_usd": 0.25},
            {"outcome": "fail", "passed": False, "output_sha": "y", "cost_usd": 3.0, "judge_cost_usd": 0.5},
        ]
        cost = summarize([rec, make_record("b", True, cost=2.0)])["cost"]
        self.assertAlmostEqual(cost["judge_total_usd"], 0.75)  # what the evaluation cost: both attempts of `a`
        self.assertAlmostEqual(cost["system_mean_usd"], 1.5)  # cost per request comes from the representative outputs


if __name__ == "__main__":
    unittest.main()
