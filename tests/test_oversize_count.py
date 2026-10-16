"""`summary.n_oversize` (contract C18): the number of judge results that refused an output above `max_output_chars` without calling the model.

An oversize output is a FAIL (not an error) decided for free: the count tells a reviewer that some failures are the size cap, not quality.
"""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from evalsprint.config import SuiteConfig
from evalsprint.runner import load_run, run_suite, summarize

TARGET = '''\
def answer(input, params):
    long = input.startswith("long")
    return {"text": ("x" * 500 if long else "a short answer"), "model": "claude-haiku-4-5", "usage": {"input_tokens": 10, "output_tokens": 5}}
'''

JUDGE = '''\
import json


def responder(system, messages, model):
    return {"text": json.dumps({"criteria": {"ok": {"reasoning": "fine", "pass": True}}}), "usage": {"input_tokens": 10, "output_tokens": 5}}
'''


class OversizeFixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "target.py").write_text(TARGET, encoding="utf-8")
        (self.dir / "judge_sim.py").write_text(JUDGE, encoding="utf-8")
        cases = [{"id": "s1", "input": "short one"}, {"id": "l1", "input": "long one"}, {"id": "l2", "input": "long two"}, {"id": "s2", "input": "short two"}]
        (self.dir / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")

    def config(self, cap=100, extra=None):
        judge = {
            "type": "llm", "name": "quality", "provider": "mock", "model": "claude-haiku-4-5",
            "provider_options": {"responder": "judge_sim.py:responder"},
            "rubric": {"name": "r", "task": "t", "criteria": [{"id": "ok", "question": "Is it fine?"}]},
            "max_output_chars": cap,
        }
        data = {"suite": "oversize", "dataset": "cases.jsonl", "concurrency": 1, "target": {"callable": "target.py:answer"}, "judges": [judge], **(extra or {})}
        path = self.dir / "eval.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return SuiteConfig.load(path)

    def run_it(self, cap=100, extra=None, **kw):
        return run_suite(self.config(cap, extra), progress=False, out_dir=self.dir / "out", **kw)


class CountTests(OversizeFixture):
    def test_each_refused_output_is_counted(self):
        run = self.run_it()
        self.assertEqual(run.summary["n_oversize"], 2)
        by_case = run.by_case()
        self.assertTrue(by_case["l1"]["judges"][0]["details"]["oversize"])
        self.assertEqual((by_case["l1"]["outcome"], by_case["l1"]["passed"]), ("fail", False))  # a fail, not an error
        self.assertEqual(run.summary["n_errors"], 0)
        self.assertEqual(by_case["l1"]["judges"][0]["cost_usd"], 0.0)  # decided without calling the model

    def test_a_cap_above_every_output_counts_nothing(self):
        self.assertEqual(self.run_it(cap=10_000).summary["n_oversize"], 0)

    def test_a_suite_without_llm_judges_counts_nothing(self):
        (self.dir / "eval.yaml").write_text(
            yaml.safe_dump({"suite": "x", "dataset": "cases.jsonl", "target": {"callable": "target.py:answer"}, "judges": [{"type": "min_chars", "name": "m", "value": 1}]}), encoding="utf-8"
        )
        run = run_suite(SuiteConfig.load(self.dir / "eval.yaml"), progress=False, out_dir=self.dir / "plain")
        self.assertEqual(run.summary["n_oversize"], 0)

    def test_it_survives_the_save_and_load_round_trip(self):
        run = self.run_it()
        self.assertEqual(load_run(run.run_dir).summary["n_oversize"], 2)
        self.assertEqual(json.loads((Path(run.run_dir) / "summary.json").read_text(encoding="utf-8"))["n_oversize"], 2)

    def test_a_repeated_case_counts_every_attempt(self):
        run = self.run_it(repeat=3)
        self.assertEqual(run.summary["n_oversize"], 2 * 3)  # two long cases, three attempts each
        entry = run.by_case()["l1"]["attempts"][0]
        self.assertEqual(entry["n_oversize"], 1)
        self.assertEqual(run.by_case()["s1"]["attempts"][0]["n_oversize"], 0)


class SummarizeUnitTests(unittest.TestCase):
    def record(self, oversize, cid="c"):
        judges = [{"judge": "q", "passed": False, "score": 0.0, "details": {"oversize": True} if oversize else {"criteria": {}}, "cost_usd": 0.0, "error": None}]
        return {"case_id": cid, "outcome": "fail", "error_source": None, "passed": False, "n_attempt_errors": 0, "judges": judges,
                "output": {"text": "x", "cost_usd": 0.0, "latency_s": 0.1, "usage": {}}, "case": {"tags": []}}

    def test_only_a_truthy_oversize_flag_counts(self):
        s = summarize([self.record(True, "a"), self.record(False, "b"), self.record(True, "c")])
        self.assertEqual(s["n_oversize"], 2)

    def test_a_record_written_before_c18_counts_zero(self):
        r = self.record(False)
        r["judges"][0].pop("details")
        self.assertEqual(summarize([r])["n_oversize"], 0)

    def test_attempt_entries_win_over_the_representative_judges(self):
        r = self.record(True, "rep")
        r["attempts"] = [
            {"outcome": "fail", "n_oversize": 1}, {"outcome": "fail", "n_oversize": 0}, {"outcome": "fail", "n_oversize": 1},
        ]
        self.assertEqual(summarize([r])["n_oversize"], 2)


if __name__ == "__main__":
    unittest.main()
