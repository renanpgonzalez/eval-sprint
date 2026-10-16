import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from evalsprint import __version__
from evalsprint.config import DEFAULT_GATE, SuiteConfig
from evalsprint.dataset import describe, load_cases
from evalsprint.judges.llm_judge import Rubric
from evalsprint.pricing import PriceTable
from evalsprint.runner import RunResult, _pass_rule, _redact, load_run, make_run_id, run_suite, save_run, summarize
from evalsprint.stats import wilson_ci
from evalsprint.types import JudgeResult, Usage, case_sha
from tests import SUITE_DIR, TEMPLATES, make_record

HAIKU = PriceTable().models["claude-haiku-4-5"]


def haiku_cost(inp: int, out: int) -> float:
    return (inp * HAIKU["input"] + out * HAIKU["output"]) / 1e6


EXPECTED_ORDER = ["billing-001", "billing-002", "shipping-001", "refund-001", "oos-001", "checks-001", "edge-untagged", "crash-001"]
EXPECTED_PASSED = {"billing-001", "shipping-001", "refund-001", "oos-001", "checks-001", "edge-untagged"}


class SuiteFixture(unittest.TestCase):
    """Copies the fixture suite to a temp dir so runs/ never pollutes the repo."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "suite"
        shutil.copytree(SUITE_DIR, self.dir, ignore=shutil.ignore_patterns("__pycache__"))
        self.config_path = self.dir / "eval.yaml"

    def config(self, overrides=None) -> SuiteConfig:
        return SuiteConfig.load(self.config_path, overrides)

    def do_run(self, overrides=None, **kw) -> RunResult:
        kw.setdefault("progress", False)
        kw.setdefault("out_dir", self.dir / "out")
        return run_suite(self.config(overrides), **kw)


class EndToEndTests(SuiteFixture):
    def test_results_follow_dataset_order_and_verdicts(self):
        run = self.do_run()
        self.assertEqual([r["case_id"] for r in run.results], EXPECTED_ORDER)
        self.assertEqual({r["case_id"] for r in run.results if r["passed"]}, EXPECTED_PASSED)

    def test_judge_applicability_end_to_end(self):
        by_case = self.do_run().by_case()
        names = lambda cid: [j["judge"] for j in by_case[cid]["judges"]]  # noqa: E731
        self.assertEqual(names("billing-001"), ["no_apology_loop", "quality"])  # expected_behavior: answer
        self.assertEqual(names("oos-001"), ["no_apology_loop"])  # abstain -> LLM judge skipped
        self.assertEqual(names("checks-001"), ["no_apology_loop", "case_checks"])  # auto-added for per-case checks
        self.assertEqual(names("edge-untagged"), ["no_apology_loop"])

    def test_llm_judge_verdicts_and_details(self):
        by_case = self.do_run().by_case()
        quality = {j["judge"]: j for j in by_case["billing-002"]["judges"]}["quality"]
        self.assertFalse(quality["passed"])
        self.assertFalse(quality["details"]["criteria"]["grounded"]["pass"])
        self.assertTrue(quality["details"]["criteria"]["polite"]["pass"])
        self.assertAlmostEqual(quality["score"], 0.5)
        self.assertFalse(by_case["billing-002"]["passed"])

    def test_target_error_is_captured_and_short_circuits_judges(self):
        rec = self.do_run().by_case()["crash-001"]
        self.assertFalse(rec["passed"])
        self.assertIn("RuntimeError: upstream exploded", rec["output"]["error"])
        self.assertEqual(rec["output"]["text"], "")
        self.assertEqual([j["reasoning"] for j in rec["judges"]], ["target error"])
        self.assertEqual(rec["n_attempt_errors"], 1)  # a case that ran once: its one attempt errored
        ok = self.do_run().by_case()["billing-001"]
        self.assertEqual(ok["n_attempt_errors"], 0)

    def test_record_shape_matches_contract(self):
        rec = self.do_run().results[0]
        self.assertEqual(set(rec), {"case_id", "case", "case_sha", "output", "judges", "outcome", "error_source", "passed", "n_attempt_errors", "target_retries_used"})  # no attempts/stability: not repeated
        self.assertEqual(rec["n_attempt_errors"], 0)
        self.assertEqual(rec["target_retries_used"], 0)
        self.assertEqual(rec["outcome"], "pass")
        self.assertIsNone(rec["error_source"])
        self.assertEqual(rec["case_sha"], case_sha(rec["case"]))  # the whole stored case dict except its id
        self.assertEqual(set(rec["case"]), {"id", "input", "reference", "expected_behavior", "tags", "checks", "metadata"})
        self.assertEqual(set(rec["output"]), {"text", "sha", "model", "usage", "cost_usd", "latency_s", "meta", "error"})
        self.assertEqual(set(rec["output"]["usage"]), {"input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"})
        self.assertEqual(set(rec["judges"][0]), {"judge", "passed", "score", "reasoning", "details", "cost_usd", "usage", "error"})
        self.assertEqual(rec["output"]["meta"]["context"], ["Settings > Billing lets you download invoices."])
        self.assertEqual(len(rec["output"]["sha"]), 12)

    def test_unknown_case_fields_reach_metadata(self):
        self.assertEqual(self.do_run().by_case()["shipping-001"]["case"]["metadata"], {"must_include": "tracking"})

    def test_summary_fields_match_contract(self):
        run = self.do_run(label="baseline v1")
        s = run.summary
        expected_keys = {
            "run_id", "suite", "label", "created_at", "evalsprint_version", "target", "judges", "judges_sha", "judges_config", "pass_rule", "gate", "dataset",
            "n_cases", "n_passed", "n_failed", "n_scored", "pass_rate", "pass_rate_ci", "n_errors", "n_target_errors", "n_judge_errors",
            "error_rate", "n_attempts", "n_attempt_errors", "n_attempt_target_errors", "n_attempt_judge_errors", "attempt_error_rate",
            "n_target_retries", "n_oversize", "placeholder", "stability", "by_tag", "by_judge", "cost", "latency", "tokens", "models", "duration_s", "repeat",
            "target_retries", "selection",
        }
        self.assertEqual(set(s), expected_keys)
        self.assertEqual(s["repeat"], {"n": 1, "per_tag": {}})  # the effective protocol is always recorded: n == 1 means every case ran once
        self.assertEqual(s["target_retries"], 1)  # C22: the configured value (the default), next to the n_target_retries that were used
        self.assertIs(type(s["target_retries"]), int)
        self.assertEqual(s["selection"], {"limited": False, "n_selected": 8, "n_config": 8, "limit": None, "tags": None, "ids": None})  # C26
        self.assertIs(s["placeholder"], True)  # the fixture's LLM judge runs on the mock provider
        self.assertEqual((s["n_oversize"], s["n_target_retries"]), (0, 1))  # crash-001 raises on every call: its one retry is counted
        self.assertEqual(s["suite"], "fixture-support")
        self.assertEqual(s["label"], "baseline v1")
        self.assertTrue(s["run_id"].endswith("-baseline-v1"))
        self.assertEqual(s["evalsprint_version"], __version__)
        self.assertEqual(set(s["target"]), {"callable", "params", "source_sha"})
        self.assertEqual((s["target"]["callable"], s["target"]["params"]), ("target.py:answer", {"model": "claude-haiku-4-5"}))
        self.assertRegex(s["target"]["source_sha"], r"^[0-9a-f]{12}$")  # the sha of target.py: a changed adapter changes the resume fingerprint
        self.assertEqual(s["judges"], ["no_apology_loop", "quality", "case_checks"])
        self.assertEqual(s["pass_rule"], "all")
        self.assertEqual(s["gate"]["max_pass_rate_drop"], 0.05)
        self.assertEqual(s["gate"]["alpha"], 0.05)  # defaults merged
        self.assertEqual(s["dataset"], "cases.jsonl")  # relative to the eval.yaml directory, never an absolute path
        self.assertGreater(s["duration_s"], 0)
        self.assertRegex(s["created_at"], r"^\d{4}-\d\d-\d\dT")

    def test_summary_numbers(self):
        s = self.do_run().summary
        self.assertEqual((s["n_cases"], s["n_passed"], s["n_failed"], s["n_scored"], s["n_errors"]), (8, 6, 1, 7, 1))
        self.assertEqual((s["n_target_errors"], s["n_judge_errors"]), (1, 0))
        self.assertAlmostEqual(s["pass_rate"], 6 / 7)  # the crashed case is an error, not a failure: it leaves the denominator
        self.assertAlmostEqual(s["error_rate"], 0.125)
        self.assertEqual(s["stability"], {"n_repeated": 0, "flip_rate": None})
        lo, hi = s["pass_rate_ci"]
        self.assertLess(lo, 6 / 7)
        self.assertGreater(hi, 6 / 7)
        self.assertEqual([lo, hi], list(wilson_ci(6, 7)))  # computed on the 7 scored cases, not on 8
        self.assertEqual(s["models"], {"claude-haiku-4-5": 7, "unknown": 1})

        self.assertEqual(s["by_tag"]["billing"]["n"], 2)  # billing-001 pass, billing-002 fail; crash-001 errored and is not scored
        self.assertEqual(s["by_tag"]["billing"]["n_passed"], 1)
        self.assertEqual(s["by_tag"]["core"], {"n": 2, "n_passed": 2, "pass_rate": 1.0, "ci": s["by_tag"]["core"]["ci"]})
        self.assertIn("(untagged)", s["by_tag"])
        self.assertEqual(s["by_tag"]["(untagged)"]["n"], 1)
        self.assertEqual(list(s["by_tag"]), sorted(s["by_tag"]))

        q = s["by_judge"]["quality"]
        self.assertEqual((q["n"], q["n_passed"], q["n_errors"]), (4, 3, 0))
        self.assertAlmostEqual(q["mean_score"], (1 + 0.5 + 1 + 1) / 4)
        self.assertAlmostEqual(q["cost_usd"], 4 * haiku_cost(400, 60))
        a = s["by_judge"]["no_apology_loop"]
        self.assertEqual((a["n"], a["n_passed"], a["n_errors"]), (7, 7, 0))  # the crashed case's skipped judge row is not a judge result
        self.assertEqual(s["by_judge"]["case_checks"]["n"], 1)

    def test_summary_cost_latency_tokens(self):
        run = self.do_run()
        s = run.summary
        priced = [r for r in run.results if r["output"]["cost_usd"] is not None]
        self.assertEqual(len(priced), 7)
        self.assertEqual(s["cost"]["n_priced"], 7)
        total = sum(r["output"]["cost_usd"] for r in priced)
        self.assertAlmostEqual(s["cost"]["system_total_usd"], total)
        self.assertAlmostEqual(s["cost"]["system_mean_usd"], total / 7)
        self.assertAlmostEqual(s["cost"]["system_per_1k_usd"], total / 7 * 1000)
        self.assertAlmostEqual(s["cost"]["judge_total_usd"], 4 * haiku_cost(400, 60))
        first = run.results[0]["output"]
        self.assertAlmostEqual(first["cost_usd"], haiku_cost(first["usage"]["input_tokens"], first["usage"]["output_tokens"]))

        self.assertEqual(s["tokens"]["input"], sum(r["output"]["usage"]["input_tokens"] for r in run.results))
        self.assertEqual(s["tokens"]["output"], sum(r["output"]["usage"]["output_tokens"] for r in run.results))
        self.assertEqual((s["tokens"]["cache_read"], s["tokens"]["cache_write"]), (0, 0))
        lat = s["latency"]
        self.assertEqual(set(lat), {"mean_s", "p50_s", "p95_s", "max_s"})
        self.assertLessEqual(lat["p50_s"], lat["p95_s"])
        self.assertLessEqual(lat["p95_s"], lat["max_s"])
        self.assertAlmostEqual(lat["max_s"], max(r["output"]["latency_s"] for r in run.results))

    def test_files_written_and_round_trip_through_load_run(self):
        run = self.do_run()
        self.assertEqual(run.run_dir, self.dir / "out")
        self.assertTrue((run.run_dir / "results.jsonl").exists())
        self.assertTrue((run.run_dir / "summary.json").exists())
        lines = (run.run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 8)
        self.assertEqual([json.loads(l)["case_id"] for l in lines], EXPECTED_ORDER)

        loaded = load_run(run.run_dir)
        self.assertEqual(loaded.results, run.results)
        self.assertEqual(loaded.summary, json.loads(json.dumps(run.summary)))
        self.assertEqual(loaded.run_id, run.run_id)
        self.assertEqual(loaded.by_case()["billing-001"]["case_id"], "billing-001")
        self.assertEqual(list(loaded.by_case()), EXPECTED_ORDER)

    def test_load_run_accepts_summary_json_path_and_str(self):
        run = self.do_run()
        self.assertEqual(load_run(run.run_dir / "summary.json").run_id, run.run_id)
        self.assertEqual(load_run(str(run.run_dir)).run_id, run.run_id)

    def test_default_run_dir_is_runs_dir_plus_run_id(self):
        cfg = self.config()
        run = run_suite(cfg, label="default-dir", progress=False)
        self.assertEqual(run.run_dir, cfg.runs_dir / run.run_id)
        self.assertTrue((run.run_dir / "summary.json").exists())

    def test_non_ascii_survives_the_round_trip(self):
        (self.dir / "cases.jsonl").write_text('{"id": "u-1", "input": "hello"}\n', encoding="utf-8")
        (self.dir / "target.py").write_text('def answer(input, params):\n    return "Café 日本語 \U0001f69a"\n', encoding="utf-8")
        run = self.do_run()
        self.assertEqual(load_run(run.run_dir).results[0]["output"]["text"], "Café 日本語 \U0001f69a")


class FilteringTests(SuiteFixture):
    def test_tags_ids_limit(self):
        self.assertEqual([r["case_id"] for r in self.do_run(tags=["billing"]).results], ["billing-001", "billing-002", "crash-001"])
        self.assertEqual([r["case_id"] for r in self.do_run(ids=["oos-001", "refund-001"]).results], ["refund-001", "oos-001"])  # dataset order
        self.assertEqual([r["case_id"] for r in self.do_run(limit=2).results], ["billing-001", "billing-002"])

    def test_summary_reflects_the_filtered_set(self):
        s = self.do_run(tags=["core"]).summary
        self.assertEqual((s["n_cases"], s["n_passed"], s["pass_rate"]), (2, 2, 1.0))

    def test_case_checks_judge_only_added_when_selected_cases_have_checks(self):
        self.assertNotIn("case_checks", self.do_run(ids=["billing-001"]).summary["judges"])
        self.assertIn("case_checks", self.do_run(ids=["checks-001"]).summary["judges"])

    def test_empty_selection_is_an_error(self):
        # regression: an empty run has pass_rate None and used to crash compare/report later
        with self.assertRaises(ValueError) as ctx:
            self.do_run(tags=["no-such-tag"])
        self.assertIn("no cases selected", str(ctx.exception))
        self.assertFalse((self.dir / "out").exists())


class SelectionRecordedTests(SuiteFixture):
    """C26: `summary.selection` says which part of the suite a run covers, so a baseline made from a subset can be told apart.

    Shape: {"limited": bool, "n_selected": int, "n_config": int, "limit": int | None, "tags": [str] | None, "ids": [str] | None}.
    `limited` is `n_selected < n_config`: a selector that still covers every case does not limit the run.
    """

    def selection(self, **kw) -> dict:
        return self.do_run(**kw).summary["selection"]

    def test_a_full_run_is_not_limited(self):
        self.assertEqual(self.selection(), {"limited": False, "n_selected": 8, "n_config": 8, "limit": None, "tags": None, "ids": None})

    def test_limit_is_recorded(self):
        self.assertEqual(self.selection(limit=3), {"limited": True, "n_selected": 3, "n_config": 8, "limit": 3, "tags": None, "ids": None})

    def test_tags_are_recorded(self):
        self.assertEqual(
            self.selection(tags=["billing", "core"]),
            {"limited": True, "n_selected": 4, "n_config": 8, "limit": None, "tags": ["billing", "core"], "ids": None},
        )

    def test_ids_are_recorded(self):
        self.assertEqual(
            self.selection(ids=["oos-001", "refund-001"]),
            {"limited": True, "n_selected": 2, "n_config": 8, "limit": None, "tags": None, "ids": ["oos-001", "refund-001"]},
        )

    def test_selectors_combine(self):
        self.assertEqual(
            self.selection(tags=["billing"], limit=2),
            {"limited": True, "n_selected": 2, "n_config": 8, "limit": 2, "tags": ["billing"], "ids": None},
        )

    def test_a_selector_that_covers_every_case_does_not_limit_the_run(self):
        self.assertEqual(self.selection(limit=8), {"limited": False, "n_selected": 8, "n_config": 8, "limit": 8, "tags": None, "ids": None})
        self.assertFalse(self.selection(limit=500)["limited"])  # more than the suite holds
        everything = [r["case_id"] for r in self.do_run().results]
        sel = self.selection(ids=everything)
        self.assertEqual((sel["limited"], sel["n_selected"], sel["ids"]), (False, 8, everything))

    def test_empty_selectors_mean_no_selector(self):
        # the API takes [] and 0 as "not given" (filter_cases does too): the summary must not claim a selection that did not happen
        self.assertEqual(self.selection(tags=[], ids=[], limit=0), {"limited": False, "n_selected": 8, "n_config": 8, "limit": None, "tags": None, "ids": None})

    def test_n_config_counts_the_dataset_not_the_selection(self):
        with open(self.dir / "cases.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": "extra-001", "input": "one more", "tags": ["extra"]}) + "\n")
        sel = self.selection(tags=["extra"])
        self.assertEqual((sel["n_selected"], sel["n_config"], sel["limited"]), (1, 9, True))

    def test_repeating_does_not_change_the_counts(self):
        # n_selected counts cases, not attempts
        run = self.do_run(limit=2, repeat=3)
        self.assertEqual(run.summary["selection"], {"limited": True, "n_selected": 2, "n_config": 8, "limit": 2, "tags": None, "ids": None})
        self.assertEqual(run.summary["n_cases"], 2)
        self.assertEqual(run.summary["n_attempts"], 6)

    def test_it_does_not_change_the_results_or_the_run_fingerprint(self):
        full = self.do_run(out_dir=self.dir / "full")
        part = self.do_run(limit=2, out_dir=self.dir / "part")
        meta = lambda run: json.loads((run.run_dir / "run_meta.json").read_text(encoding="utf-8"))["fingerprint"]  # noqa: E731
        self.assertEqual(meta(full), meta(part))  # a subset of the same suite is the same protocol: `--resume` may still reuse its records
        verdicts = lambda run: [(r["case_id"], r["case_sha"], r["outcome"]) for r in run.results]  # noqa: E731
        self.assertEqual(verdicts(part), verdicts(full)[:2])

    def test_it_survives_the_disk_round_trip_as_plain_json(self):
        run = self.do_run(tags=["billing", "core"], limit=3)
        on_disk = json.loads((run.run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["selection"], {"limited": True, "n_selected": 3, "n_config": 8, "limit": 3, "tags": ["billing", "core"], "ids": None})
        self.assertEqual(load_run(run.run_dir).summary["selection"], on_disk["selection"])

    def test_a_resumed_run_records_its_own_selection(self):
        finished = self.do_run(out_dir=self.dir / "first")
        self.assertFalse(finished.summary["selection"]["limited"])
        subset = self.do_run(resume_from=finished.run_dir, out_dir=self.dir / "second", ids=["billing-001", "billing-002"])
        self.assertEqual(
            subset.summary["selection"],
            {"limited": True, "n_selected": 2, "n_config": 8, "limit": None, "tags": None, "ids": ["billing-001", "billing-002"]},
        )
        self.assertEqual(subset.summary["resumed_from"]["reused_records"], 2)
        whole = self.do_run(resume_from=subset.run_dir, out_dir=self.dir / "third")
        self.assertEqual(whole.summary["selection"], {"limited": False, "n_selected": 8, "n_config": 8, "limit": None, "tags": None, "ids": None})

    def test_the_cli_run_passes_its_flags_so_no_cli_change_is_needed(self):
        import contextlib
        import io

        from evalsprint.cli import main

        out_dir = self.dir / "cli-out"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["run", "-c", str(self.config_path), "--out", str(out_dir), "--quiet", "--no-report", "--tags", "billing,core", "--limit", "3"])
        self.assertEqual(code, 0)
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["selection"], {"limited": True, "n_selected": 3, "n_config": 8, "limit": 3, "tags": ["billing", "core"], "ids": None})
        full = self.dir / "cli-full"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            main(["run", "-c", str(self.config_path), "--out", str(full), "--quiet", "--no-report"])
        self.assertFalse(json.loads((full / "summary.json").read_text(encoding="utf-8"))["selection"]["limited"])


class PassRuleTests(SuiteFixture):
    def test_all(self):
        run = self.do_run(["pass_rule=all"])
        self.assertEqual(run.summary["pass_rule"], "all")
        self.assertFalse(run.by_case()["billing-002"]["passed"])

    def test_any(self):
        # billing-002 fails the LLM judge but passes no_apology_loop -> passes under "any"
        run = self.do_run(["pass_rule=any"])
        self.assertTrue(run.by_case()["billing-002"]["passed"])
        self.assertFalse(run.by_case()["crash-001"]["passed"])  # target errors never pass
        self.assertEqual(run.summary["n_passed"], 7)

    def test_named_single_judge(self):
        run = self.do_run(["pass_rule=no_apology_loop"])
        self.assertTrue(run.by_case()["billing-002"]["passed"])
        run = self.do_run(["pass_rule=quality"])
        self.assertFalse(run.by_case()["billing-002"]["passed"])
        # cases the named judge does not apply to cannot pass
        self.assertFalse(run.by_case()["oos-001"]["passed"])
        self.assertTrue(run.by_case()["billing-001"]["passed"])

    def test_named_list_requires_all_named_judges(self):
        run = self.do_run(["pass_rule=[no_apology_loop, quality]"])
        self.assertFalse(run.by_case()["billing-002"]["passed"])
        self.assertTrue(run.by_case()["billing-001"]["passed"])
        # oos-001: `quality` does not apply (expected_behavior=abstain), so only no_apology_loop is selected
        self.assertTrue(run.by_case()["oos-001"]["passed"])

    def test_pass_rule_unit(self):
        ok = JudgeResult("a", True)
        bad = JudgeResult("b", False)
        self.assertFalse(_pass_rule("all", []))
        self.assertFalse(_pass_rule("any", []))
        self.assertTrue(_pass_rule("all", [ok, ok]))
        self.assertFalse(_pass_rule("all", [ok, bad]))
        self.assertTrue(_pass_rule("any", [bad, ok]))
        self.assertFalse(_pass_rule("any", [bad, bad]))
        self.assertTrue(_pass_rule("a", [ok, bad]))
        self.assertFalse(_pass_rule("b", [ok, bad]))
        self.assertFalse(_pass_rule("zzz", [ok, ok]))  # unknown judge name never passes
        self.assertTrue(_pass_rule(["a"], [ok, bad]))
        self.assertFalse(_pass_rule(["a", "b"], [ok, bad]))
        self.assertTrue(_pass_rule(["a", "zzz"], [ok, bad]))  # names that did not run are ignored


class ConcurrencyTests(SuiteFixture):
    def test_results_identical_across_concurrency_levels(self):
        runs = {}
        for n in (1, 2, 8):
            runs[n] = self.do_run([f"concurrency={n}", "target.params.jitter=true"], out_dir=self.dir / f"out{n}")
        def stable(run):  # a crashed target has no reported latency: wall-clock is inherently noisy
            return [dict(r, output=dict(r["output"], latency_s=None)) if r["output"]["error"] else r for r in run.results]

        for n in (2, 8):
            self.assertEqual(stable(runs[n]), stable(runs[1]), f"concurrency={n}")
            for key in ("n_cases", "n_passed", "pass_rate", "by_tag", "by_judge", "cost", "tokens", "models"):
                self.assertEqual(runs[n].summary[key], runs[1].summary[key], key)

    def test_dataset_order_kept_even_when_completion_order_differs(self):
        run = self.do_run(["concurrency=8", "target.params.jitter=true"])
        self.assertEqual([r["case_id"] for r in run.results], EXPECTED_ORDER)

    def test_zero_or_negative_concurrency_is_a_config_error(self):
        for value in ("0", "-1"):
            with self.subTest(concurrency=value), self.assertRaises(ValueError) as caught:
                self.do_run([f"concurrency={value}"])
            self.assertIn("'concurrency' must be at least 1", str(caught.exception))
        self.assertFalse((self.dir / "out").exists())  # rejected before a run directory was made

    def test_progress_output_goes_to_stderr(self):
        import contextlib
        import io

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.do_run(progress=True)
        text = err.getvalue()
        self.assertIn("PASS billing-001", text)
        self.assertIn("FAIL billing-002", text)
        self.assertIn("ERR  crash-001", text)


class SecretRedactionTests(SuiteFixture):
    def test_secret_params_never_reach_summary_json(self):
        # regression: ${API_KEY} in target.params was written in clear to summary.json, which is committed as the baseline
        with mock.patch.dict(os.environ, {"ES_LEAKY": "sk-live-123"}):
            run = self.do_run(["target.params.api_key=${ES_LEAKY}", "target.params.max_tokens=400", "target.params.api_key_env=MY_KEY"])
        params = run.summary["target"]["params"]
        self.assertEqual(params["api_key"], "***")
        self.assertEqual(params["max_tokens"], 400)  # not a secret
        self.assertEqual(params["api_key_env"], "MY_KEY")  # a variable NAME is useful and harmless
        self.assertNotIn("sk-live-123", (run.run_dir / "summary.json").read_text())
        # the adapter still received the real value (only the stored copy is masked)
        self.assertEqual(self.config(["target.params.api_key=abc"]).target["params"]["api_key"], "abc")

    def test_redact_unit(self):
        data = {
            "api_key": "k", "API-KEY": "k", "client_secret": "s", "auth_token": "t", "token": "t", "Authorization": "Bearer x",
            "password": "p", "max_tokens": 5, "prompt": "v1", "empty_token": "", "none_key": None, "api_key_env": "X",
            "nested": {"access_key": "a", "list": [{"secret": "z"}, "plain"]},
        }
        out = _redact(data)
        for k in ("api_key", "API-KEY", "client_secret", "auth_token", "token", "Authorization", "password"):
            self.assertEqual(out[k], "***", k)
        self.assertEqual((out["max_tokens"], out["prompt"], out["empty_token"], out["api_key_env"]), (5, "v1", "", "X"))
        self.assertEqual(out["nested"], {"access_key": "***", "list": [{"secret": "***"}, "plain"]})
        self.assertEqual(data["api_key"], "k")  # input untouched


class ConfigOverridesInRunTests(SuiteFixture):
    def test_target_param_override_changes_output(self):
        run = self.do_run(["target.params.prefix='[v2] '"])
        self.assertTrue(run.results[0]["output"]["text"].startswith("[v2] "))
        self.assertEqual(run.summary["target"]["params"]["prefix"], "[v2] ")

    def test_model_override_changes_cost(self):
        cheap = self.do_run(["target.params.model=gpt-4o-mini"], out_dir=self.dir / "a")
        base = self.do_run(out_dir=self.dir / "b")
        self.assertLess(cheap.summary["cost"]["system_total_usd"], base.summary["cost"]["system_total_usd"])
        self.assertEqual(cheap.summary["models"], {"gpt-4o-mini": 7, "unknown": 1})

    def test_pricing_override_from_config(self):
        run = self.do_run(["pricing.claude-haiku-4-5.input=1000", "pricing.claude-haiku-4-5.output=0"])
        first = run.results[0]["output"]
        self.assertAlmostEqual(first["cost_usd"], first["usage"]["input_tokens"] * 1000 / 1e6)


class TemplatesEndToEndTests(unittest.TestCase):
    """The starting-point templates must load and run offline exactly as shipped."""

    JUDGE_SIM = (
        "import json\n"
        "def judge(system, messages, model):\n"
        "    ids = [l[2:].split(':')[0].split(' ')[0] for l in system.splitlines() if l.startswith('- ') and ':' in l]\n"
        "    return json.dumps({'criteria': {i: {'reasoning': 'ok', 'pass': True} for i in ids}})\n"
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        for name in ("eval.yaml", "rubric.yaml", "target_adapter.py"):
            shutil.copy(TEMPLATES / name, self.dir / name)
        shutil.copy(TEMPLATES / "cases.example.jsonl", self.dir / "cases.jsonl")
        (self.dir / "sim.py").write_text(self.JUDGE_SIM, encoding="utf-8")

    def llm_index(self):
        judges = yaml.safe_load((self.dir / "eval.yaml").read_text())["judges"]
        return next(i for i, j in enumerate(judges) if j["type"] == "llm")

    def test_eval_yaml_loads_with_documented_defaults(self):
        cfg = SuiteConfig.load(self.dir / "eval.yaml")
        self.assertEqual(cfg.gate, DEFAULT_GATE)  # the template documents the defaults, so they must be the defaults
        self.assertEqual(cfg.pass_rule, "all")
        self.assertEqual(cfg.target["callable"], "target_adapter.py:offline_stub")
        self.assertEqual(cfg.judges[self.llm_index()]["provider"], "anthropic")  # ${JUDGE_PROVIDER:-anthropic}
        with mock.patch.dict(os.environ, {"JUDGE_PROVIDER": "openai", "JUDGE_MODEL": "gpt-5.4"}):
            judge = SuiteConfig.load(self.dir / "eval.yaml").judges[self.llm_index()]
        self.assertEqual((judge["provider"], judge["model"]), ("openai", "gpt-5.4"))

    def test_rubric_and_cases_load(self):
        rubric = Rubric.load("rubric.yaml", self.dir)
        self.assertEqual(rubric.ids, ["correct", "grounded", "safe_and_in_scope", "concise_actionable"])
        self.assertFalse(rubric.criteria[-1]["critical"])
        for cid in rubric.ids:
            self.assertIn(cid, rubric.system_prompt())
        cases = load_cases(self.dir / "cases.jsonl")
        info = describe(cases)
        self.assertGreaterEqual(info["n_cases"], 8)
        self.assertEqual(set(info["expected_behavior"]), {"answer", "abstain", "refuse"})
        self.assertIsInstance(next(c for c in cases if c.id == "account-001").input, dict)
        self.assertIn("must_include", next(c for c in cases if c.id == "billing-001").metadata)

    def test_template_runs_offline_end_to_end(self):
        i = self.llm_index()
        with mock.patch.dict(os.environ, {"JUDGE_PROVIDER": "mock"}):
            cfg = SuiteConfig.load(self.dir / "eval.yaml", [f"judges.{i}.provider_options.responder=sim.py:judge"])
            run = run_suite(cfg, label="template-smoke", out_dir=self.dir / "out", progress=False)
        s = run.summary
        self.assertEqual(s["n_cases"], 10)
        self.assertEqual(s["n_errors"], 0)
        self.assertEqual(s["judges"], ["length", "no_boilerplate_leak", "not_empty", "quality", "case_checks"])
        self.assertEqual(s["by_judge"]["quality"]["n_errors"], 0)
        # the stub gives one canned answer, so only the per-case check on account-002 (needs a reset link) fails
        self.assertEqual({r["case_id"] for r in run.results if not r["passed"]}, {"account-002"})
        self.assertGreater(s["cost"]["system_total_usd"], 0)
        self.assertEqual(load_run(run.run_dir).summary["run_id"], run.run_id)


class SummarizeTests(unittest.TestCase):
    def test_empty(self):
        s = summarize([])
        self.assertEqual(s["n_cases"], 0)
        self.assertIsNone(s["pass_rate"])
        self.assertEqual(s["error_rate"], 0.0)
        self.assertEqual(s["by_tag"], {})
        self.assertIsNone(s["cost"]["system_mean_usd"])
        self.assertIsNone(s["cost"]["system_per_1k_usd"])
        self.assertIsNone(s["latency"]["p95_s"])
        self.assertIsNone(s["latency"]["max_s"])

    def test_counts_rates_and_tags(self):
        recs = [
            make_record("a", True, tags=["x", "y"]),
            make_record("b", False, tags=["x"]),
            make_record("c", True, tags=[]),
            make_record("d", False, tags=["y"], error="boom"),
        ]
        s = summarize(recs)  # legacy records (no `outcome`): derived from passed and the target error
        self.assertEqual((s["n_cases"], s["n_passed"], s["n_failed"], s["n_scored"], s["n_errors"]), (4, 2, 1, 3, 1))
        self.assertAlmostEqual(s["pass_rate"], 2 / 3)  # the errored case is not a failure
        self.assertEqual(s["error_rate"], 0.25)
        self.assertEqual(s["by_tag"]["x"]["n"], 2)
        self.assertEqual(s["by_tag"]["x"]["n_passed"], 1)
        self.assertEqual((s["by_tag"]["y"]["n"], s["by_tag"]["y"]["pass_rate"]), (1, 1.0))  # `d` errored: not scored under `y`
        self.assertEqual(s["by_tag"]["(untagged)"]["pass_rate"], 1.0)
        self.assertEqual(list(s["by_tag"]), ["(untagged)", "x", "y"])
        lo, hi = s["by_tag"]["x"]["ci"]
        self.assertTrue(0 <= lo < 0.5 < hi <= 1)

    def test_by_judge_stats(self):
        j = lambda name, passed, score=None, cost=0.0, error=None: {  # noqa: E731
            "judge": name, "passed": passed, "score": score, "reasoning": "", "details": {}, "cost_usd": cost, "usage": {}, "error": error,
        }
        recs = [
            make_record("a", True, judges=[j("q", True, 1.0, 0.01), j("h", True)]),
            make_record("b", False, judges=[j("q", False, 0.5, 0.02), j("h", True)]),
            make_record("c", False, judges=[j("q", False, None, 0.0, "parse error")]),
        ]
        by = summarize(recs)["by_judge"]
        self.assertEqual((by["q"]["n"], by["q"]["n_passed"], by["q"]["n_errors"]), (2, 1, 1))  # n = scored rows: the judge's own error is left out
        self.assertEqual(by["q"]["pass_rate"], 0.5)
        self.assertAlmostEqual(by["q"]["mean_score"], 0.75)  # None scores excluded
        self.assertAlmostEqual(by["q"]["cost_usd"], 0.03)
        self.assertEqual((by["h"]["n"], by["h"]["pass_rate"]), (2, 1.0))
        self.assertIsNone(by["h"]["mean_score"])
        self.assertEqual(set(by["q"]), {"n", "n_passed", "pass_rate", "ci", "mean_score", "n_errors", "cost_usd"})

    def test_cost_latency_tokens_models(self):
        recs = [
            make_record("a", True, cost=0.01, latency=1.0, model="m1", usage={"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 2, "cache_write_tokens": 1}),
            make_record("b", True, cost=0.03, latency=3.0, model="m1", usage={"input_tokens": 20, "output_tokens": 15, "cache_read_tokens": 0, "cache_write_tokens": 0}),
            make_record("c", False, cost=None, latency=None, model=None),
        ]
        s = summarize(recs)
        self.assertAlmostEqual(s["cost"]["system_total_usd"], 0.04)
        self.assertAlmostEqual(s["cost"]["system_mean_usd"], 0.02)  # mean over priced cases only
        self.assertAlmostEqual(s["cost"]["system_per_1k_usd"], 20.0)
        self.assertEqual(s["cost"]["n_priced"], 2)
        self.assertEqual(s["tokens"], {"input": 30, "output": 20, "cache_read": 2, "cache_write": 1})
        self.assertEqual(s["models"], {"m1": 2, "unknown": 1})
        self.assertAlmostEqual(s["latency"]["mean_s"], 2.0)
        self.assertAlmostEqual(s["latency"]["p50_s"], 2.0)
        self.assertAlmostEqual(s["latency"]["p95_s"], 2.9)
        self.assertEqual(s["latency"]["max_s"], 3.0)

    def test_summarize_is_json_serialisable(self):
        json.dumps(summarize([make_record("a", True, tags=["t"], cost=0.1, latency=1.0)]))


class SaveRunRegressionTests(unittest.TestCase):
    def test_non_json_meta_does_not_crash_save(self):
        # regression: a client adapter returning e.g. a Path/datetime in meta lost the whole run at save time
        from datetime import datetime

        rec = make_record("a", True)
        rec["output"]["meta"] = {"when": datetime(2026, 1, 2, 3, 4, 5), "path": Path("/tmp/x"), "obj": object.__class__}
        with tempfile.TemporaryDirectory() as tmp:
            save_run(tmp, {"run_id": "r"}, [rec])
            loaded = load_run(tmp)
        self.assertEqual(loaded.results[0]["output"]["meta"]["when"], "2026-01-02 03:04:05")
        self.assertEqual(loaded.results[0]["output"]["meta"]["path"], str(Path("/tmp/x")))  # default=str: "\\tmp\\x" on Windows

    def test_save_creates_nested_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = save_run(Path(tmp) / "a" / "b" / "run", {"run_id": "r"}, [make_record("a", True)])
            self.assertTrue((out / "summary.json").exists())

    def test_run_result_helpers(self):
        rr = RunResult(run_dir=None, summary={"run_id": "abc"}, results=[make_record("a", True), make_record("b", False)])
        self.assertEqual(rr.run_id, "abc")
        self.assertEqual(list(rr.by_case()), ["a", "b"])
        self.assertEqual(RunResult(None, {}, []).run_id, "")


class RunIdTests(unittest.TestCase):
    def test_format(self):
        self.assertRegex(make_run_id(), r"^\d{8}-\d{6}-\d{6}$")  # microseconds: two runs in the same second differ
        self.assertRegex(make_run_id("Prompt v2 (final)!"), r"^\d{8}-\d{6}-\d{6}-Prompt-v2-final$")

    def test_label_is_slugged_and_truncated(self):
        rid = make_run_id("x" * 100)
        self.assertEqual(len(rid.split("-", 3)[3]), 40)
        self.assertNotIn("/", make_run_id("a/b\\c"))

    def test_empty_label_gives_plain_timestamp(self):
        self.assertRegex(make_run_id(""), r"^\d{8}-\d{6}-\d{6}$")

    def test_id_comes_from_the_given_start_time(self):
        from datetime import datetime, timezone

        start = datetime(2026, 9, 29, 8, 7, 6, 54321, tzinfo=timezone.utc)
        self.assertEqual(make_run_id("pr-7", start), "20260929-080706-054321-pr-7")


class FlakyJudgeFixture(SuiteFixture):
    """The fixture suite with a judge model that goes down for chosen cases."""

    RESPONDER = (
        "import json, re\n"
        "DOWN_FOR = {down!r}\n"
        "def judge_responder(system, messages, model):\n"
        "    content = messages[-1]['content']\n"
        "    match = re.search(r'<input(?:_[0-9a-f]+)?>\\n?(.*?)\\n?</input(?:_[0-9a-f]+)?>', content, re.S)\n"
        "    if any(text in match.group(1) for text in DOWN_FOR):\n"
        "        raise RuntimeError('HTTP 401 invalid x-api-key')\n"
        "    out = re.search(r'<ai_output(?:_[0-9a-f]+)?>\\n?(.*?)\\n?</ai_output(?:_[0-9a-f]+)?>', content, re.S).group(1).lower()\n"
        "    grounded = 'cannot help' not in out and \"don't know\" not in out\n"
        "    polite = '!!!' not in out\n"
        "    return {{'text': json.dumps({{'criteria': {{'grounded': {{'reasoning': 'ok', 'pass': grounded}}, 'polite': {{'reasoning': 'ok', 'pass': polite}}}}}}),\n"
        "            'usage': {{'input_tokens': 400, 'output_tokens': 60}}}}\n"
    )

    def interrupting_judge(self, on_input):
        (self.dir / "stop_judge.py").write_text(
            "def check(case, output):\n"
            f"    if case.input == {on_input!r}:\n"
            "        raise KeyboardInterrupt\n"
            "    return True\n",
            encoding="utf-8",
        )
        return ["judges.1={type: custom, name: stopper, callable: 'stop_judge.py:check'}"]

    def judge_down_for(self, *inputs):
        (self.dir / "flaky_judge.py").write_text(self.RESPONDER.format(down=list(inputs)), encoding="utf-8")
        return ["judges.1.provider_options.responder=flaky_judge.py:judge_responder"]


class OutcomeAccountingTests(FlakyJudgeFixture):
    """C1: pass / fail / error, and errors stay out of every pass rate."""

    def test_judge_error_makes_the_case_an_error_not_a_failure(self):
        run = self.do_run(self.judge_down_for("How do I download my invoice?"))  # billing-001
        rec = run.by_case()["billing-001"]
        self.assertEqual((rec["outcome"], rec["error_source"], rec["passed"]), ("error", "judge", False))
        quality = next(j for j in rec["judges"] if j["judge"] == "quality")
        self.assertIn("401", quality["error"])
        self.assertEqual(next(j for j in rec["judges"] if j["judge"] == "no_apology_loop")["passed"], True)  # the other judge still ran

    def test_summary_counters_and_rates(self):
        s = self.do_run(self.judge_down_for("How do I download my invoice?")).summary
        # 8 cases: billing-001 judge error, crash-001 target error, billing-002 fails, 5 pass
        self.assertEqual((s["n_cases"], s["n_passed"], s["n_failed"], s["n_scored"]), (8, 5, 1, 6))
        self.assertEqual((s["n_errors"], s["n_target_errors"], s["n_judge_errors"]), (2, 1, 1))
        self.assertAlmostEqual(s["error_rate"], 2 / 8)
        self.assertAlmostEqual(s["pass_rate"], 5 / 6)
        self.assertEqual(s["pass_rate_ci"], list(wilson_ci(5, 6)))  # on the scored cases

    def test_by_tag_covers_scored_cases_only(self):
        s = self.do_run(self.judge_down_for("How do I download my invoice?")).summary
        # billing: billing-001 (judge error), billing-002 (fail), crash-001 (target error) -> only billing-002 is scored
        self.assertEqual((s["by_tag"]["billing"]["n"], s["by_tag"]["billing"]["n_passed"]), (1, 0))
        self.assertEqual(s["by_tag"]["core"]["n"], 1)  # billing-001 (error) is gone; shipping-001 remains

    def test_by_judge_excludes_target_errors_and_counts_only_its_own_errors(self):
        s = self.do_run(self.judge_down_for("How do I download my invoice?")).summary
        quality = s["by_judge"]["quality"]
        self.assertEqual(quality["n_errors"], 1)
        self.assertEqual((quality["n"], quality["n_passed"]), (3, 2))  # 4 answer-cases minus the errored one; billing-002 fails
        self.assertAlmostEqual(quality["pass_rate"], 2 / 3)
        apology = s["by_judge"]["no_apology_loop"]
        self.assertEqual((apology["n"], apology["n_errors"]), (7, 0))  # crash-001's skipped row is not a result of this judge
        self.assertEqual(apology["n_passed"], 7)

    def test_judge_errors_keep_the_cost_that_was_spent(self):
        # billing-001's judge call fails BEFORE any tokens are reported, so only the other three calls are billed here;
        # the errored result of a judge that did spend (via `.usage` / `.cost_usd` on the exception) keeps it:
        from evalsprint.judges import run_judges
        from evalsprint.judges.base import Judge
        from evalsprint.types import Case, TargetOutput

        class Broke(Exception):
            usage = Usage(300, 40)
            cost_usd = 0.0123

        class Spender(Judge):
            def evaluate(self, case, output):
                raise Broke("parse failed after two calls")

        (res,) = run_judges([Spender("spender")], Case(id="c", input="q"), TargetOutput(text="x"))
        self.assertIsNotNone(res.error)
        self.assertEqual((res.usage, res.cost_usd), (Usage(300, 40), 0.0123))
        rec = make_record("a", False, judges=[{"judge": "spender", "passed": False, "score": None, "reasoning": "", "details": {}, "cost_usd": 0.0123, "usage": {}, "error": "Broke: x"}])
        by = summarize([rec])["by_judge"]["spender"]
        self.assertAlmostEqual(by["cost_usd"], 0.0123)  # errored rows are billed
        self.assertEqual((by["n"], by["n_errors"]), (0, 1))
        self.assertAlmostEqual(summarize([rec])["cost"]["judge_total_usd"], 0.0123)

    def test_exception_without_spend_info_has_zero_cost(self):
        from evalsprint.judges import run_judges
        from evalsprint.judges.base import Judge
        from evalsprint.types import Case, TargetOutput

        class Down(Judge):
            def evaluate(self, case, output):
                raise ConnectionError("network unreachable")

        (res,) = run_judges([Down("down")], Case(id="c", input="q"), TargetOutput(text="x"))
        self.assertEqual((res.cost_usd, res.usage), (0.0, Usage()))
        self.assertIn("ConnectionError", res.error)

    def test_target_error_record(self):
        rec = self.do_run().by_case()["crash-001"]
        self.assertEqual((rec["outcome"], rec["error_source"], rec["passed"]), ("error", "target", False))

    def test_a_total_outage_leaves_nothing_scored(self):
        run = self.do_run(self.judge_down_for(""))  # "" is in every input: the judge model is down for all cases
        s = run.summary
        quality_cases = [r for r in run.results if any(j["judge"] == "quality" for j in r["judges"])]
        self.assertTrue(quality_cases)
        self.assertTrue(all(r["outcome"] == "error" and r["error_source"] in ("judge", "target") for r in quality_cases))
        self.assertEqual(s["by_judge"]["quality"]["n"], 0)
        self.assertIsNone(s["by_judge"]["quality"]["pass_rate"])

    def test_all_cases_errored_means_no_pass_rate(self):
        (self.dir / "cases.jsonl").write_text('{"id": "a", "input": "BOOM"}\n{"id": "b", "input": "BOOM"}\n', encoding="utf-8")
        s = self.do_run().summary
        self.assertEqual((s["n_scored"], s["n_errors"], s["error_rate"]), (0, 2, 1.0))
        self.assertIsNone(s["pass_rate"])
        self.assertEqual(s["by_judge"], {})  # no judge produced a result

    def test_a_bug_in_the_runner_side_of_one_case_becomes_an_error_record(self):
        from evalsprint import runner

        real = runner.run_judges

        def flaky(judges, case, output):
            if case.id == "billing-001":
                raise RuntimeError("bug")
            return real(judges, case, output)

        with mock.patch.object(runner, "run_judges", flaky):
            run = self.do_run()
        rec = run.by_case()["billing-001"]
        self.assertEqual((rec["outcome"], rec["error_source"]), ("error", "judge"))
        self.assertIn("internal error: RuntimeError: bug", rec["judges"][0]["error"])
        self.assertEqual(len(run.results), 8)  # the other cases were not lost

    def test_case_sha_identifies_the_content_of_the_case(self):
        first = self.do_run().by_case()
        lines = (self.dir / "cases.jsonl").read_text(encoding="utf-8").replace("Go to Settings > Billing and click Download.", "Use the Billing page.")
        (self.dir / "cases.jsonl").write_text(lines, encoding="utf-8")
        second = self.do_run(out_dir=self.dir / "out2").by_case()
        self.assertNotEqual(first["billing-001"]["case_sha"], second["billing-001"]["case_sha"])  # reference edited
        self.assertEqual(first["billing-002"]["case_sha"], second["billing-002"]["case_sha"])
        self.assertEqual(first["billing-002"]["case_sha"], case_sha(first["billing-002"]["case"]))

    def test_editing_a_judge_facing_extra_field_changes_the_case_sha(self):
        # shipping-001 carries `must_include: tracking`, which the judge is told to look for; weakening it must show up as an edited case
        first = self.do_run().by_case()
        (self.dir / "cases.jsonl").write_text((self.dir / "cases.jsonl").read_text(encoding="utf-8").replace('"must_include": "tracking"', '"must_include": "anything"'), encoding="utf-8")
        second = self.do_run(out_dir=self.dir / "out2").by_case()
        self.assertNotEqual(first["shipping-001"]["case_sha"], second["shipping-001"]["case_sha"])
        self.assertEqual(first["billing-002"]["case_sha"], second["billing-002"]["case_sha"])
        edited_tags = (self.dir / "cases.jsonl").read_text(encoding="utf-8").replace('"tags": ["billing"]}', '"tags": ["billing", "vip"]}', 1)
        (self.dir / "cases.jsonl").write_text(edited_tags, encoding="utf-8")
        third = self.do_run(out_dir=self.dir / "out3").by_case()
        self.assertNotEqual(second["billing-002"]["case_sha"], third["billing-002"]["case_sha"])  # tags select judges, so they count too

    def test_outcome_of_reads_records_with_and_without_the_new_fields(self):
        from evalsprint.runner import outcome_of

        self.assertEqual(outcome_of({"outcome": "pass", "passed": True}), ("pass", None))
        self.assertEqual(outcome_of({"outcome": "error", "error_source": "judge", "judges": []}), ("error", "judge"))
        self.assertEqual(outcome_of({"outcome": "error", "output": {"error": "x"}, "judges": []}), ("error", "target"))  # source missing: derived
        self.assertEqual(outcome_of({"passed": True, "output": {}, "judges": []}), ("pass", None))  # legacy
        self.assertEqual(outcome_of({"passed": False, "output": {}, "judges": []}), ("fail", None))
        self.assertEqual(outcome_of({"passed": False, "output": {"error": "x"}, "judges": [{"error": "x"}]}), ("error", "target"))
        self.assertEqual(outcome_of({"passed": False, "output": {}, "judges": [{"error": "x"}]}), ("error", "judge"))
        self.assertEqual(outcome_of({"outcome": "pass", "error_source": "judge"}), ("pass", None))


class SummarizeOutcomeTests(unittest.TestCase):
    def rec(self, cid, outcome, source=None, tags=None, judges=None):
        r = make_record(cid, outcome == "pass", tags=tags, judges=judges)
        r["outcome"], r["error_source"] = outcome, source
        if source == "target":
            r["output"]["error"] = "boom"
        return r

    def test_counters_from_explicit_outcomes(self):
        s = summarize(
            [
                self.rec("a", "pass", tags=["x"]), self.rec("b", "pass", tags=["x"]), self.rec("c", "fail", tags=["x"]),
                self.rec("d", "error", "judge", tags=["x"]), self.rec("e", "error", "target", tags=["y"]),
            ]
        )
        self.assertEqual((s["n_passed"], s["n_failed"], s["n_scored"], s["n_errors"], s["n_target_errors"], s["n_judge_errors"]), (2, 1, 3, 2, 1, 1))
        self.assertAlmostEqual(s["pass_rate"], 2 / 3)
        self.assertAlmostEqual(s["error_rate"], 2 / 5)
        self.assertEqual(s["by_tag"]["x"]["n"], 3)
        self.assertNotIn("y", s["by_tag"])  # its only case errored

    def test_by_judge_rows_of_target_errored_cases_are_dropped(self):
        j = lambda name, passed, error=None: {"judge": name, "passed": passed, "score": None, "reasoning": "", "details": {}, "cost_usd": 0.0, "usage": {}, "error": error}  # noqa: E731
        s = summarize(
            [
                self.rec("a", "pass", judges=[j("q", True)]),
                self.rec("b", "error", "target", judges=[j("q", False, "boom")]),  # placeholder written by run_judges
                self.rec("c", "error", "judge", judges=[j("q", False, "401")]),
            ]
        )
        q = s["by_judge"]["q"]
        self.assertEqual((q["n"], q["n_passed"], q["n_errors"]), (1, 1, 1))  # only c's own error counts for this judge
        self.assertEqual(q["pass_rate"], 1.0)

    def test_hand_written_records_without_outcome_still_summarise(self):
        s = summarize([make_record("a", True), make_record("b", False), make_record("c", False, error="boom")])
        self.assertEqual((s["n_scored"], s["n_errors"]), (2, 1))


class SaveRunAtomicTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "run"

    def test_no_temp_files_are_left_behind(self):
        save_run(self.dir, {"run_id": "r"}, [make_record("a", True)])
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["results.jsonl", "summary.json"])

    def test_a_failure_while_writing_keeps_the_previous_files_intact(self):
        save_run(self.dir, {"run_id": "old"}, [make_record("old", True)])
        before = {name: (self.dir / name).read_bytes() for name in ("results.jsonl", "summary.json")}
        with mock.patch("evalsprint.runner.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_run(self.dir, {"run_id": "new"}, [make_record("new", False)])
        self.assertEqual({name: (self.dir / name).read_bytes() for name in before}, before)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["results.jsonl", "summary.json"])  # and no half-written temp file

    def test_summary_is_written_last_so_its_presence_means_a_complete_run(self):
        order = []
        real = os.replace

        def spy(src, dst):
            order.append(Path(dst).name)
            return real(src, dst)

        with mock.patch("evalsprint.runner.os.replace", spy):
            save_run(self.dir, {"run_id": "r"}, [make_record("a", True)])
        self.assertEqual(order, ["results.jsonl", "summary.json"])

    def test_lone_surrogates_anywhere_in_a_record_or_the_summary_cannot_lose_the_run(self):
        rec = make_record("a", True, text="ok \ud83d")
        rec["judges"][0]["reasoning"] = "because \ude00"
        rec["output"]["meta"] = {"traceback": "x\ud800"}
        save_run(self.dir, {"run_id": "r", "label": "l\ud83d"}, [rec])
        raw = (self.dir / "results.jsonl").read_bytes().decode("utf-8")  # strict decode
        self.assertNotIn("\ud83d", raw)
        loaded = load_run(self.dir)
        self.assertEqual(loaded.results[0]["output"]["text"], "ok \ufffd")
        self.assertEqual(loaded.summary["label"], "l\ufffd")

    def test_files_use_unix_newlines_on_every_platform(self):
        save_run(self.dir, {"run_id": "r"}, [make_record("a", True), make_record("b", False)])
        self.assertNotIn(b"\r\n", (self.dir / "results.jsonl").read_bytes())
        self.assertNotIn(b"\r\n", (self.dir / "summary.json").read_bytes())

    def test_truncated_results_file_names_the_file_and_line(self):
        save_run(self.dir, {"run_id": "r"}, [make_record("a", True), make_record("b", True)])
        text = (self.dir / "results.jsonl").read_text(encoding="utf-8")
        (self.dir / "results.jsonl").write_text(text[: len(text) - 40], encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            load_run(self.dir)
        self.assertIn("results.jsonl:2", str(ctx.exception))
        self.assertIn("truncated", str(ctx.exception))


class IncrementalResultsTests(FlakyJudgeFixture):
    def test_finished_cases_survive_an_interrupted_run(self):
        overrides = ["concurrency=1"] + self.interrupting_judge("I want a refund for last month.")  # refund-001 is the 4th case
        with self.assertRaises(KeyboardInterrupt):
            self.do_run(overrides)
        partial = self.dir / "out" / "results.partial.jsonl"
        lines = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["case_id"] for r in lines], ["billing-001", "billing-002", "shipping-001"])  # paid work kept, in completion order
        self.assertTrue(all(r["attempt"] is None and r["outcome"] in ("pass", "fail", "error") for r in lines))
        self.assertFalse((self.dir / "out" / "summary.json").exists())  # an interrupted run is not a run
        self.assertFalse((self.dir / "out" / "results.jsonl").exists())

    def test_the_run_stops_spending_after_an_interrupt(self):
        calls = self.dir / "calls.log"
        (self.dir / "target.py").write_text(
            "def answer(input, params):\n"
            f"    open({str(calls)!r}, 'a', encoding='utf-8').write(str(input) + '\\n')\n"
            "    return 'ok'\n",
            encoding="utf-8",
        )
        with self.assertRaises(KeyboardInterrupt):
            self.do_run(["concurrency=1"] + self.interrupting_judge("Where is my order?"))  # 3rd of 8
        self.assertLessEqual(len(calls.read_text(encoding="utf-8").splitlines()), 4)  # not all 8

    def test_partial_file_is_removed_on_success_and_stale_ones_are_replaced(self):
        out = self.dir / "out"
        out.mkdir()
        (out / "results.partial.jsonl").write_text('{"stale": true}\n', encoding="utf-8")
        run = self.do_run()
        self.assertFalse((run.run_dir / "results.partial.jsonl").exists())
        self.assertEqual(sorted(p.name for p in run.run_dir.iterdir()), ["results.jsonl", "run_meta.json", "summary.json"])

    def test_repeated_attempts_are_each_appended(self):
        (self.dir / "spy_judge.py").write_text("def check(case, output):\n    return True\n", encoding="utf-8")
        lines = []
        real_unlink = Path.unlink

        def keep_partial(path, *a, **kw):  # capture the partial file just before it is deleted
            if path.name == "results.partial.jsonl":
                lines.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
            return real_unlink(path, *a, **kw)

        with mock.patch.object(Path, "unlink", keep_partial):
            self.do_run(["repeat={n: 3, tags: [core]}"])
        core = [r for r in lines if r["case_id"] in ("billing-001", "shipping-001")]
        self.assertEqual(sorted((r["case_id"], r["attempt"]) for r in core), [(c, k) for c in ("billing-001", "shipping-001") for k in range(3)])
        self.assertEqual(len(lines), 6 + 2 * 3)


class RunIdAndDirectoryTests(SuiteFixture):
    def test_run_id_matches_the_directory_and_has_microseconds(self):
        run = run_suite(self.config(), label="pr-7", progress=False)
        self.assertEqual(run.run_dir.name, run.run_id)
        self.assertRegex(run.run_id, r"^\d{8}-\d{6}-\d{6}-pr-7$")

    def test_three_runs_with_the_same_label_make_three_directories(self):
        cfg = self.config()
        ids = [run_suite(cfg, label="pr-7", progress=False, limit=2).run_id for _ in range(3)]
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(len(list(cfg.runs_dir.iterdir())), 3)

    def test_an_existing_run_directory_is_never_overwritten(self):
        from datetime import datetime, timezone

        from evalsprint import runner

        fixed = datetime(2026, 9, 29, 12, 0, 0, 123456, tzinfo=timezone.utc)
        real = runner.make_run_id
        cfg = self.config()
        with mock.patch.object(runner, "make_run_id", lambda label=None, now=None: real(label, fixed)):
            first = run_suite(cfg, label="x", progress=False, limit=2)
            first_summary = (first.run_dir / "summary.json").read_bytes()
            second = run_suite(cfg, label="x", progress=False, limit=3)
            third = run_suite(cfg, label="x", progress=False, limit=4)
        self.assertEqual(first.run_id, "20260929-120000-123456-x")
        self.assertEqual((second.run_id, third.run_id), ("20260929-120000-123456-x-2", "20260929-120000-123456-x-3"))
        self.assertEqual((first.run_dir / "summary.json").read_bytes(), first_summary)  # the first run is untouched
        self.assertEqual([load_run(r.run_dir).summary["n_cases"] for r in (first, second, third)], [2, 3, 4])
        self.assertEqual([load_run(r.run_dir).summary["run_id"] for r in (first, second, third)], [first.run_id, second.run_id, third.run_id])

    def test_parallel_runs_with_the_same_label_do_not_collide(self):
        from concurrent.futures import ThreadPoolExecutor

        cfg = self.config()
        with ThreadPoolExecutor(max_workers=4) as pool:
            runs = list(pool.map(lambda _: run_suite(cfg, label="ci", progress=False, limit=2), range(4)))
        self.assertEqual(len({r.run_id for r in runs}), 4)
        self.assertEqual(len(list(cfg.runs_dir.iterdir())), 4)
        for r in runs:
            self.assertEqual(load_run(r.run_dir).summary["run_id"], r.run_id)
            self.assertEqual(len(load_run(r.run_dir).results), 2)

    def test_the_run_id_is_taken_at_the_start_of_the_run(self):
        from evalsprint import runner

        events = []
        real = runner.make_run_id

        def spy(label=None, now=None):
            events.append("run_id")
            return real(label, now)

        (self.dir / "target.py").write_text(f"def answer(input, params):\n    open({str(self.dir / 'events')!r}, 'a').write('target\\n')\n    return 'ok'\n", encoding="utf-8")
        with mock.patch.object(runner, "make_run_id", spy):
            self.do_run(limit=2)
        events += (self.dir / "events").read_text().split()
        self.assertEqual(events[0], "run_id")  # before any target call
        self.assertEqual(events.count("run_id"), 1)

    def test_created_at_is_the_start_time_and_duration_covers_the_run(self):
        s = self.do_run().summary
        self.assertRegex(s["created_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")
        stamp = s["run_id"][:15]  # YYYYMMDD-HHMMSS of the same start instant
        self.assertEqual(s["created_at"].replace("-", "").replace(":", "").replace("T", "-")[:15], stamp)

    def test_explicit_out_dir_is_used_as_given_and_may_be_reused(self):
        out = self.dir / "chosen"
        a = self.do_run(out_dir=out, limit=2)
        b = self.do_run(out_dir=out, limit=3)
        self.assertEqual((a.run_dir, b.run_dir), (out, out))
        self.assertEqual(load_run(out).summary["n_cases"], 3)
        self.assertNotEqual(a.run_id, b.run_id)  # still a fresh id each time

    def test_config_errors_leave_no_directory_behind(self):
        cfg = self.config(["pass_rule=nope"])
        with self.assertRaises(ValueError):
            run_suite(cfg, progress=False)
        self.assertFalse(cfg.runs_dir.exists())


class SummaryMetadataTests(SuiteFixture):
    def test_dataset_is_relative_to_the_config_directory(self):
        self.assertEqual(self.do_run().summary["dataset"], "cases.jsonl")

    def test_dataset_outside_the_config_directory_stays_relative_never_absolute(self):
        data = self.dir.parent / "shared" / "cases.jsonl"
        data.parent.mkdir()
        shutil.copy(self.dir / "cases.jsonl", data)
        ref = self.do_run([f"dataset=../shared/cases.jsonl"]).summary["dataset"]
        self.assertEqual(ref, "../shared/cases.jsonl")
        self.assertFalse(Path(ref).is_absolute())

    def test_tag_groups_reach_the_summary(self):
        groups = {"kind": ["billing", "shipping"], "priority": ["core"]}
        self.assertEqual(self.do_run(["tag_groups={kind: [billing, shipping], priority: [core]}"]).summary["tag_groups"], groups)
        self.assertNotIn("tag_groups", self.do_run().summary)

    def test_summary_json_is_valid_json_with_the_new_counters(self):
        run = self.do_run()
        on_disk = json.loads((run.run_dir / "summary.json").read_text(encoding="utf-8"))
        for key in ("n_failed", "n_scored", "n_target_errors", "n_judge_errors", "stability"):
            self.assertIn(key, on_disk)


class SurrogateEndToEndTests(SuiteFixture):
    def test_a_lone_surrogate_in_case_and_output_no_longer_loses_the_run(self):
        (self.dir / "cases.jsonl").write_text(
            '{"id": "a", "input": "hello \\ud83d", "reference": "x", "tags": ["t"]}\n{"id": "b", "input": "fine", "reference": "x", "tags": ["t"]}\n', encoding="utf-8"
        )
        (self.dir / "target.py").write_text('def answer(input, params):\n    return {"text": "ok \\ud83d", "model": "claude-haiku-4-5", "usage": {"input_tokens": 5, "output_tokens": 5}}\n', encoding="utf-8")
        run = self.do_run(["judges=[{type: min_chars, name: len, value: 1}]"])
        self.assertEqual([r["outcome"] for r in run.results], ["pass", "pass"])
        self.assertEqual(run.results[0]["case"]["input"], "hello \ufffd")
        self.assertEqual(run.results[0]["output"]["text"], "ok \ufffd")
        (run.run_dir / "results.jsonl").read_bytes().decode("utf-8")  # strict decode of what was saved
        self.assertEqual(load_run(run.run_dir).results, run.results)


class ConfigGuardTests(SuiteFixture):
    def test_pass_rule_naming_an_unknown_judge_is_an_error_with_a_suggestion(self):
        with self.assertRaises(ValueError) as ctx:
            self.do_run(["pass_rule=qualty"])
        self.assertIn("did you mean 'quality'", str(ctx.exception))
        self.do_run(["pass_rule=[quality, no_apology_loop]"])  # known names, and the automatic case_checks judge, are fine
        self.do_run(["pass_rule=case_checks"])

    def test_a_suite_without_any_judge_is_an_error(self):
        (self.dir / "cases.jsonl").write_text('{"id": "a", "input": "hello"}\n', encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            self.do_run(["judges=[]"])
        self.assertIn("no judges", str(ctx.exception))

    def test_behaviors_are_enforced_before_anything_runs(self):
        with self.assertRaises(ValueError) as ctx:
            self.do_run(["behaviors=[answer, refuse]"])  # the fixture has abstain cases
        self.assertIn("abstain", str(ctx.exception))
        self.assertFalse((self.dir / "out").exists())

    def test_extended_redaction_reaches_summary_json(self):
        secrets = {
            "secret_key": "sk1", "private_key": "pk1", "signing_key": "sg1", "bearer": "br1", "jwt": "jw1", "cookie": "ck1", "session_id": "ss1",
            "dsn": "https://abc@sentry.example/1", "connection_string": "Server=x;Password=hunter2", "password": "pw1", "credentials": "cr1",
        }
        overrides = [f"target.params.{k}={v}" for k, v in secrets.items() if k != "dsn" and k != "connection_string"]
        overrides += ["target.params.dsn='https://abc@sentry.example/1'", "target.params.connection_string='Server=x;Password=hunter2'"]
        overrides += ["target.params.endpoint='https://svc:hunter2@api.example.com/v1'", "target.params.max_tokens=400", "target.params.api_key_env=MY_KEY"]
        run = self.do_run(overrides)
        text = (run.run_dir / "summary.json").read_text(encoding="utf-8")
        for leaked in ("sk1", "pk1", "sg1", "br1", "jw1", "ck1", "ss1", "sentry.example", "hunter2", "pw1", "cr1"):
            self.assertNotIn(leaked, text)
        params = run.summary["target"]["params"]
        self.assertEqual([params[k] for k in secrets], ["***"] * len(secrets))
        self.assertEqual((params["max_tokens"], params["api_key_env"]), (400, "MY_KEY"))
        self.assertEqual(params["endpoint"], "https://***@api.example.com/v1")


class ReusedOutDirTests(FlakyJudgeFixture):
    """`run --out DIR` onto a directory that already holds a finished run must not leave that run readable as this one."""

    def finished_run_in(self, out):
        run = self.do_run(out_dir=out, limit=3)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["results.jsonl", "run_meta.json", "summary.json"])
        return run

    def test_an_interrupted_rerun_leaves_no_stale_results_behind(self):
        out = self.dir / "out"
        self.finished_run_in(out)
        with self.assertRaises(KeyboardInterrupt):
            self.do_run(["concurrency=1"] + self.interrupting_judge("Where is my order?"), out_dir=out)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["results.partial.jsonl", "run_meta.json"])  # the old summary/results are gone
        with self.assertRaises(FileNotFoundError):
            load_run(out)  # nothing to mistake for this run

    def test_the_old_run_is_removed_before_the_first_case_starts(self):
        out = self.dir / "out"
        seen = self.dir / "seen.log"
        (self.dir / "target.py").write_text(  # written before the first run: the loader caches a module per path
            "import os\n"
            "def answer(input, params):\n"
            f"    open({str(seen)!r}, 'a').write(str(os.path.exists({str(out / 'summary.json')!r})) + ',' + str(os.path.exists({str(out / 'results.jsonl')!r})) + '\\n')\n"
            "    return 'ok'\n",
            encoding="utf-8",
        )
        self.finished_run_in(out)
        seen.unlink()
        self.do_run(["concurrency=1"], out_dir=out, limit=2)  # one thread: concurrent appends to seen.log overwrite each other's line on Windows
        self.assertEqual(seen.read_text(encoding="utf-8").split(), ["False,False", "False,False"])

    def test_a_clean_rerun_replaces_the_old_run(self):
        out = self.dir / "out"
        first = self.finished_run_in(out)
        second = self.do_run(out_dir=out, limit=5)
        self.assertEqual((first.summary["n_cases"], load_run(out).summary["n_cases"]), (3, 5))
        self.assertEqual(len(load_run(out).results), 5)

    def test_files_that_are_not_run_files_are_left_alone(self):
        out = self.dir / "out"
        self.finished_run_in(out)
        (out / "notes.txt").write_text("keep me", encoding="utf-8")
        (out / "calibration.json").write_text("{}", encoding="utf-8")
        self.do_run(out_dir=out, limit=2)
        self.assertEqual((out / "notes.txt").read_text(encoding="utf-8"), "keep me")
        self.assertTrue((out / "calibration.json").exists())

    def test_somebody_elses_summary_json_is_never_deleted(self):
        out = self.dir / "out"
        out.mkdir()
        (out / "summary.json").write_text('{"project": "my accounting export"}', encoding="utf-8")
        calls = self.dir / "calls.log"
        (self.dir / "target.py").write_text(f"def answer(input, params):\n    open({str(calls)!r}, 'a').write('x')\n    return 'ok'\n", encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            self.do_run(out_dir=out)
        self.assertIn("summary.json that is not an evalsprint run file", str(caught.exception))
        self.assertEqual(json.loads((out / "summary.json").read_text(encoding="utf-8")), {"project": "my accounting export"})
        self.assertFalse(calls.exists())  # refused before anything was spent

    def test_somebody_elses_results_jsonl_is_never_deleted(self):
        out = self.dir / "out"
        out.mkdir()
        (out / "results.jsonl").write_text("this is not json\n", encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            self.do_run(out_dir=out)
        self.assertIn("results.jsonl that is not an evalsprint run file", str(caught.exception))
        self.assertEqual((out / "results.jsonl").read_text(encoding="utf-8"), "this is not json\n")

    def test_a_config_error_deletes_nothing(self):
        out = self.dir / "out"
        self.finished_run_in(out)
        with self.assertRaises(ValueError):
            self.do_run(["pass_rule=nope"], out_dir=out)  # rejected before the directory is touched
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["results.jsonl", "run_meta.json", "summary.json"])
        self.assertEqual(len(load_run(out).results), 3)

    def test_results_only_from_a_crash_between_the_two_saves_is_replaced(self):
        out = self.dir / "out"
        self.finished_run_in(out)
        (out / "summary.json").unlink()  # what a crash between the two atomic writes leaves behind
        run = self.do_run(out_dir=out, limit=2)
        self.assertEqual(len(load_run(out).results), 2)
        self.assertEqual(run.summary["n_cases"], 2)


class PartialNewerThanSummaryTests(unittest.TestCase):
    """A directory where results.partial.jsonl is newer than summary.json holds a run that is being (or was)
    overwritten: summary.json and results.jsonl are from an older run and must not be read."""

    def make_dir(self, tmp, partial_offset):
        out = Path(tmp) / "run"
        save_run(out, {"run_id": "old"}, [make_record("a", True)])
        partial = out / "results.partial.jsonl"
        partial.write_text('{"case_id": "a"}\n', encoding="utf-8")
        summary_time = (out / "summary.json").stat().st_mtime
        os.utime(partial, (summary_time + partial_offset, summary_time + partial_offset))
        return out

    def test_load_run_refuses_it(self):
        from evalsprint.runner import check_run_files

        with tempfile.TemporaryDirectory() as tmp:
            out = self.make_dir(tmp, +30)
            for call in (load_run, check_run_files):
                with self.subTest(call=call.__name__), self.assertRaises(ValueError) as caught:
                    call(out)
                self.assertIn("results.partial.jsonl is newer than summary.json", str(caught.exception))
                self.assertIn("run it again", str(caught.exception))
            with self.assertRaises(ValueError):
                load_run(out / "summary.json")  # a path to the summary is the same directory

    def test_an_older_partial_file_does_not_block_a_finished_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_run(self.make_dir(tmp, -30)).summary["run_id"], "old")

    def test_a_finished_run_without_a_partial_file_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = save_run(Path(tmp) / "run", {"run_id": "r"}, [make_record("a", True)])
            self.assertEqual(load_run(out).summary["run_id"], "r")


class JudgesFingerprintTests(SuiteFixture):
    """summary.judges_sha / judges_config: changing what the judges check must be visible in the baseline diff."""

    def run_sha(self, overrides=None, **kw):
        kw.setdefault("out_dir", self.dir / f"out-{len(list(self.dir.glob('out-*')))}")
        return self.do_run(overrides, limit=2, **kw).summary["judges_sha"]

    def edit_config(self, mutate):
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        mutate(data)
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    def test_shape_and_the_documented_formula(self):
        import hashlib

        s = self.do_run().summary
        self.assertRegex(s["judges_sha"], r"^[0-9a-f]{12}$")
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        rubric = (self.dir / "rubric.yaml").read_text(encoding="utf-8")
        blob = json.dumps({"judges": data["judges"], "rubrics": {"rubric.yaml": rubric}, "pass_rule": "all"}, sort_keys=True, default=str)
        self.assertEqual(s["judges_sha"], hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12])
        self.assertEqual(s["judges_config"], data["judges"])  # the list a reviewer reads when the sha differs
        stored = load_run(self.dir / "out").summary
        self.assertEqual((stored["judges_sha"], stored["judges_config"]), (s["judges_sha"], s["judges_config"]))

    def test_it_ignores_everything_that_is_not_a_judge_setting(self):
        base = self.run_sha()
        self.assertEqual(self.run_sha(["target.params.prefix=x", "concurrency=1", "gate.max_pass_rate_drop=0.5"], label="other"), base)
        self.assertEqual(self.run_sha(repeat=2), base)

    def test_a_rubric_file_edit_changes_it(self):
        base = self.run_sha()
        rubric = self.dir / "rubric.yaml"
        rubric.write_text(rubric.read_text(encoding="utf-8") + "\n# a criterion was quietly relaxed\n", encoding="utf-8")
        self.assertNotEqual(self.run_sha(), base)

    def test_line_endings_of_the_rubric_do_not_change_it(self):
        base = self.run_sha()
        rubric = self.dir / "rubric.yaml"
        rubric.write_bytes(rubric.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))  # a Windows checkout
        self.assertEqual(self.run_sha(), base)

    def test_a_utf8_bom_added_by_an_editor_does_not_change_it(self):
        base = self.run_sha()
        rubric = self.dir / "rubric.yaml"
        raw = rubric.read_bytes()
        rubric.write_bytes(b"\xef\xbb\xbf" + raw)
        self.assertEqual(self.run_sha(), base)  # the bytes changed, the rubric did not
        rubric.write_bytes(b"\xef\xbb\xbf" + raw.replace(b"\n", b"\r\n"))  # Notepad: BOM and CRLF together
        self.assertEqual(self.run_sha(), base)
        rubric.write_bytes(b"\xef\xbb\xbf" + raw + b"# edited\n")
        self.assertNotEqual(self.run_sha(), base)  # a real edit is still seen through the BOM

    def test_an_inline_rubric_is_hashed_by_content(self):
        rubric = yaml.safe_load((self.dir / "rubric.yaml").read_text(encoding="utf-8"))
        self.edit_config(lambda data: data["judges"][1].update(rubric=rubric))
        base = self.run_sha()
        self.assertEqual(self.run_sha(), base)
        rubric["criteria"][0]["question"] = "Is it roughly fine?"
        self.edit_config(lambda data: data["judges"][1].update(rubric=rubric))
        self.assertNotEqual(self.run_sha(), base)

    def test_judge_model_settings_and_pass_rule_change_it(self):
        base = self.run_sha()
        self.assertNotEqual(self.run_sha(["judges.1.model=claude-sonnet-5-5"]), base)
        self.assertNotEqual(self.run_sha(["judges.0.values=nothing"]), base)  # a heuristic judge's setting
        self.assertNotEqual(self.run_sha(["pass_rule=quality"]), base)
        self.assertNotEqual(self.run_sha(["judges.1.when_behavior=abstain"]), base)  # scoping which cases the judge sees

    def test_it_is_computed_after_env_expansion(self):
        self.edit_config(lambda data: data["judges"][1].update(model="${JUDGE_MODEL:-claude-haiku-4-5}"))
        default = self.run_sha()
        self.assertEqual(default, self.run_sha())
        with mock.patch.dict(os.environ, {"JUDGE_MODEL": "claude-haiku-4-5"}):
            self.assertEqual(self.run_sha(), default)  # same value once expanded
        with mock.patch.dict(os.environ, {"JUDGE_MODEL": "claude-sonnet-5-5"}):
            self.assertNotEqual(self.run_sha(), default)
            self.assertEqual(self.do_run(limit=1, out_dir=self.dir / "o").summary["judges_config"][1]["model"], "claude-sonnet-5-5")

    def test_secrets_are_masked_in_the_stored_copy_and_do_not_move_the_sha(self):
        from evalsprint.runner import judges_fingerprint

        def cfgs(key):
            return [{"type": "llm", "name": "q", "model": "m", "rubric": {"criteria": []}, "provider_options": {"api_key": key, "base_url": "https://user:pw@host/v1"}}]

        sha_a, shown = judges_fingerprint(cfgs("sk-ant-rotated-1234567890"), "all", self.dir)
        sha_b, _ = judges_fingerprint(cfgs("sk-ant-other-abcdefghij"), "all", self.dir)
        self.assertEqual(sha_a, sha_b)  # rotating a key is not a judge change
        text = json.dumps(shown)
        self.assertNotIn("sk-ant-rotated", text)
        self.assertNotIn("user:pw", text)
        self.assertIn("host/v1", text)
        self.assertNotEqual(judges_fingerprint([dict(cfgs("k")[0], provider_options={"base_url": "https://other/v1"})], "all", self.dir)[0], sha_a)
        self.assertEqual(cfgs("real")[0]["provider_options"]["api_key"], "real")  # the config itself is not modified

    def test_a_suite_without_llm_judges_still_has_one(self):
        from evalsprint.runner import judges_fingerprint

        sha, shown = judges_fingerprint([{"type": "contains", "name": "c", "values": ["x"]}], "all", self.dir)
        self.assertRegex(sha, r"^[0-9a-f]{12}$")
        self.assertEqual(shown, [{"type": "contains", "name": "c", "values": ["x"]}])
        self.assertRegex(judges_fingerprint([], "any", self.dir)[0], r"^[0-9a-f]{12}$")

    def test_it_travels_with_the_baseline_copy(self):
        from evalsprint.cli import main

        run = self.do_run(limit=2)  # the fixture's 8th case crashes on purpose: a full run would be refused as a baseline
        self.assertTrue(run.summary["placeholder"])  # and its LLM judge runs on the mock provider: --force copies it anyway (C18)
        dest = self.dir / "baseline"
        self.assertEqual(main(["baseline", str(run.run_dir), "--to", str(dest)]), 3)
        self.assertFalse(dest.exists())
        self.assertEqual(main(["baseline", str(run.run_dir), "--to", str(dest), "--force"]), 0)
        self.assertEqual(load_run(dest).summary["judges_sha"], run.summary["judges_sha"])


class CustomJudgeCodeInFingerprintTests(SuiteFixture):
    """C11 / W7: the code of a custom judge decides pass or fail, so editing it must change judges_sha
    (the compare gate then reports a judge change instead of a quiet 'improvement')."""

    JUDGE = "def check(case, output):\n    return len(output.text) > 3\n"

    def setUp(self):
        super().setUp()
        self.judge_file = self.dir / "my_judge.py"
        self.judge_file.write_text(self.JUDGE, encoding="utf-8")
        self.cfg = [{"type": "custom", "name": "mine", "callable": "my_judge.py:check"}]

    def sha(self, cfgs=None, warnings=None):
        from evalsprint.runner import judges_fingerprint

        return judges_fingerprint(self.cfg if cfgs is None else cfgs, "all", self.dir, warnings)[0]

    def test_editing_the_judge_file_changes_the_sha(self):
        base = self.sha()
        self.assertEqual(self.sha(), base)
        self.judge_file.write_text(self.JUDGE.replace("> 3", "> 0"), encoding="utf-8")  # a threshold quietly relaxed
        self.assertNotEqual(self.sha(), base)

    def test_a_comment_is_an_edit_too(self):
        base = self.sha()
        self.judge_file.write_text(self.JUDGE + "# harmless?\n", encoding="utf-8")
        self.assertNotEqual(self.sha(), base)  # the kit cannot tell harmless from harmful: review decides

    def test_line_endings_and_a_bom_do_not_change_it(self):
        base = self.sha()
        raw = self.judge_file.read_bytes().replace(b"\r\n", b"\n")  # write_text already wrote CRLF on Windows: start from LF
        self.judge_file.write_bytes(raw.replace(b"\n", b"\r\n"))
        self.assertEqual(self.sha(), base)
        self.judge_file.write_bytes(b"\xef\xbb\xbf" + raw.replace(b"\n", b"\r\n"))
        self.assertEqual(self.sha(), base)

    def test_each_custom_judge_is_hashed_under_its_own_name(self):
        other = self.dir / "other_judge.py"
        other.write_text("def check(case, output):\n    return True\n", encoding="utf-8")
        both = self.cfg + [{"type": "custom", "name": "other", "callable": "other_judge.py:check"}]
        base = self.sha(both)
        other.write_text("def check(case, output):\n    return False\n", encoding="utf-8")
        self.assertNotEqual(self.sha(both), base)
        self.assertNotEqual(self.sha(both), self.sha())

    def test_the_documented_formula(self):
        import hashlib

        blob = json.dumps({"judges": self.cfg, "rubrics": {"custom:mine": self.JUDGE}, "pass_rule": "all"}, sort_keys=True, default=str)
        self.assertEqual(self.sha(), hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12])

    def test_a_module_function_spec_hashes_the_module_source(self):
        import sys

        (self.dir / "kit_custom_pkg_judge.py").write_text(self.JUDGE, encoding="utf-8")
        cfg = [{"type": "custom", "name": "mine", "callable": "kit_custom_pkg_judge:check"}]
        warnings = []
        with mock.patch.object(sys, "path", [str(self.dir)] + sys.path):
            base = self.sha(cfg, warnings)
            (self.dir / "kit_custom_pkg_judge.py").write_text(self.JUDGE.replace("> 3", "> 9"), encoding="utf-8")
            changed = self.sha(cfg, warnings)
        self.assertNotEqual(base, changed)
        self.assertEqual(warnings, [])  # resolved: nothing to warn about

    def test_a_spec_that_names_no_source_file_is_hashed_as_text_and_warns(self):
        warnings = []
        cfg = [{"type": "custom", "name": "builtin", "callable": "math:floor"}]  # a compiled module: no .py to hash
        base = self.sha(cfg, warnings)
        self.assertEqual(self.sha(cfg), base)
        self.assertEqual(len(warnings), 1)
        self.assertIn("'builtin'", warnings[0])
        self.assertIn("will NOT change judges_sha", warnings[0])
        self.assertNotEqual(self.sha([{"type": "custom", "name": "builtin", "callable": "math:ceil"}]), base)  # the spec string itself is hashed

    def test_a_missing_file_does_not_crash_the_fingerprint(self):
        self.judge_file.unlink()
        warnings = []
        self.assertRegex(self.sha(warnings=warnings), r"^[0-9a-f]{12}$")  # build_judges reports the missing file; this must not hide it
        self.assertEqual(len(warnings), 1)

    def test_a_run_shows_the_change_in_its_summary(self):
        overrides = ["judges.0={type: custom, name: mine, callable: 'my_judge.py:check'}"]
        first = self.do_run(overrides, limit=2, out_dir=self.dir / "a")
        self.judge_file.write_text(self.JUDGE.replace("> 3", "> 0"), encoding="utf-8")
        second = self.do_run(overrides, limit=2, out_dir=self.dir / "b")
        self.assertNotEqual(first.summary["judges_sha"], second.summary["judges_sha"])
        self.assertEqual(first.summary["judges_config"], second.summary["judges_config"])  # the config text did not change: only the code did

    def test_an_unresolvable_spec_warns_on_stderr_before_the_run(self):
        import contextlib
        import io

        overrides = ["judges.0={type: custom, name: floor, callable: 'math:floor'}"]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.do_run(overrides, limit=1, out_dir=self.dir / "c")
        self.assertIn("warning: custom judge 'floor'", err.getvalue())


class CustomJudgeDataFilesTests(SuiteFixture):
    """C20: `data_files: [path, ...]` of a custom judge (relative to eval.yaml) are hashed into `judges_sha`: a judge that reads a
    word list or a policy file changes when that file does, and a path that does not exist fails at configuration time."""

    JUDGE = "def check(case, output):\n    return len(output.text) > 3\n"

    def setUp(self):
        super().setUp()
        (self.dir / "my_judge.py").write_text(self.JUDGE, encoding="utf-8")
        (self.dir / "banned.txt").write_text("refund\nlawsuit\n", encoding="utf-8")
        (self.dir / "data").mkdir()
        (self.dir / "data" / "policy.json").write_text('{"max": 3}', encoding="utf-8")
        self.cfg = [{"type": "custom", "name": "mine", "callable": "my_judge.py:check", "data_files": ["banned.txt", "data/policy.json"]}]

    def sha(self, cfgs=None):
        from evalsprint.runner import judges_fingerprint

        return judges_fingerprint(self.cfg if cfgs is None else cfgs, "all", self.dir)[0]

    def test_editing_a_data_file_changes_the_sha(self):
        base = self.sha()
        self.assertEqual(self.sha(), base)
        (self.dir / "banned.txt").write_text("refund\nlawsuit\nchargeback\n", encoding="utf-8")
        self.assertNotEqual(self.sha(), base)

    def test_every_listed_file_counts(self):
        base = self.sha()
        (self.dir / "data" / "policy.json").write_text('{"max": 4}', encoding="utf-8")
        self.assertNotEqual(self.sha(), base)

    def test_line_endings_and_a_bom_do_not_change_it_but_binary_bytes_do(self):
        base = self.sha()
        (self.dir / "banned.txt").write_bytes(b"\xef\xbb\xbfrefund\r\nlawsuit\r\n")
        self.assertEqual(self.sha(), base)
        (self.dir / "banned.txt").write_bytes(b"\xff\xfe\x00\x01")  # not UTF-8: hashed as bytes
        changed = self.sha()
        self.assertNotEqual(changed, base)
        (self.dir / "banned.txt").write_bytes(b"\xff\xfe\x00\x02")
        self.assertNotEqual(self.sha(), changed)

    def test_a_judge_without_data_files_keeps_the_sha_it_had_before_this_key_existed(self):
        import hashlib

        plain = [{"type": "custom", "name": "mine", "callable": "my_judge.py:check"}]
        blob = json.dumps({"judges": plain, "rubrics": {"custom:mine": self.JUDGE}, "pass_rule": "all"}, sort_keys=True, default=str)
        self.assertEqual(self.sha(plain), hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12])  # baselines made before C20 stay valid
        self.assertNotEqual(self.sha(), self.sha(plain))

    def test_the_paths_are_resolved_against_the_config_directory_not_the_cwd(self):
        here = os.getcwd()
        os.chdir(tempfile.gettempdir())
        self.addCleanup(os.chdir, here)
        self.assertRegex(self.sha(), r"^[0-9a-f]{12}$")
        (self.dir / "banned.txt").write_text("changed\n", encoding="utf-8")
        self.assertNotEqual(self.sha(), self.sha([{**self.cfg[0], "data_files": ["data/policy.json"]}]))

    def test_a_run_shows_the_change_in_its_summary_and_the_code_did_not_change(self):
        overrides = ["judges.0={type: custom, name: mine, callable: 'my_judge.py:check', data_files: [banned.txt]}"]
        first = self.do_run(overrides, limit=2, out_dir=self.dir / "a")
        (self.dir / "banned.txt").write_text("something else\n", encoding="utf-8")
        second = self.do_run(overrides, limit=2, out_dir=self.dir / "b")
        self.assertNotEqual(first.summary["judges_sha"], second.summary["judges_sha"])
        self.assertEqual(first.summary["judges_config"], second.summary["judges_config"])

    def test_the_judge_keeps_its_data_files(self):
        from evalsprint.judges import build_judges

        (judge,) = build_judges(self.cfg, self.dir, PriceTable(), [])
        self.assertEqual(judge.data_files, ["banned.txt", "data/policy.json"])

    def test_a_single_path_as_a_string_is_a_list_of_one(self):
        from evalsprint.judges import build_judges

        single = [{"type": "custom", "name": "mine", "callable": "my_judge.py:check", "data_files": "banned.txt"}]
        (judge,) = build_judges(single, self.dir, PriceTable(), [])
        self.assertEqual(judge.data_files, ["banned.txt"])
        base = self.sha(single)
        (self.dir / "banned.txt").write_text("edited\n", encoding="utf-8")
        self.assertNotEqual(self.sha(single), base)

    def test_a_missing_file_is_a_configuration_error_that_names_it(self):
        from evalsprint.judges import build_judges

        bad = [{"type": "custom", "name": "mine", "callable": "my_judge.py:check", "data_files": ["banned.txt", "nope.txt"]}]
        with self.assertRaises(ValueError) as ctx:
            build_judges(bad, self.dir, PriceTable(), [])
        self.assertIn("'nope.txt' not found", str(ctx.exception))
        self.assertIn("relative to the eval.yaml directory", str(ctx.exception))

    def test_a_directory_or_a_wrong_shape_is_an_error(self):
        from evalsprint.judges import build_judges

        for value in (["data"], {"a": 1}, [1], [""], 5):
            bad = [{"type": "custom", "name": "mine", "callable": "my_judge.py:check", "data_files": value}]
            with self.assertRaises(ValueError, msg=repr(value)):
                build_judges(bad, self.dir, PriceTable(), [])

    def test_only_custom_judges_take_the_key(self):
        from evalsprint.judges import build_judges

        with self.assertRaises(ValueError) as ctx:
            build_judges([{"type": "contains", "name": "c", "values": ["x"], "data_files": ["banned.txt"]}], self.dir, PriceTable(), [])
        self.assertIn("data_files", str(ctx.exception))

    def test_describe_fails_on_a_missing_file_before_any_run(self):
        import contextlib
        import io

        from evalsprint.cli import main

        cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        cfg["judges"] = [{"type": "custom", "name": "mine", "callable": "my_judge.py:check", "data_files": ["missing.txt"]}]
        path = self.dir / "broken.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = main(["describe", "-c", str(path)])
        self.assertEqual(code, 2)
        self.assertIn("'missing.txt' not found", err.getvalue())


if __name__ == "__main__":
    unittest.main()
