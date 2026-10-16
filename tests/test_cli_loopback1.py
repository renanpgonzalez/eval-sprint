"""Loop-back 1, phase B: the CLI side of C22 and C26.

* `baseline` warns, and still copies, when the run was limited (`summary.selection.limited`, written by the runner) and when the summary does not record
  its selection at all (a run of an older evalsprint). Exit codes do not change.
* A summary.json with text where a number belongs: `report` (and `compare`, `baseline`) end with a one-line error that names the field instead of
  `TypeError: bad operand type for abs()`. Exit codes do not change: a summary the command copes with is still fine.
* The help of `label --previous-labels` states its condition (C24 wording).
* The `Target retries` line of `run` reads `summary.target_retries`.
"""

import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from evalsprint import runner
from evalsprint.cli import _non_numeric_field, main
from evalsprint.config import SuiteConfig

TARGET = '''\
def answer(input, params):
    if params.get("down"):
        raise RuntimeError("upstream down")
    return {"text": "answer to " + input, "model": "claude-haiku-4-5", "usage": {"input_tokens": 100, "output_tokens": 20}}
'''

N_CASES = 6


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


class Fixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "suite"
        self.dir.mkdir()
        (self.dir / "target.py").write_text(TARGET, encoding="utf-8")
        rows = [{"id": f"c{i}", "input": f"q{i}", "tags": ["a" if i < 3 else "b"]} for i in range(N_CASES)]
        (self.dir / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.config = self.dir / "eval.yaml"
        self.config.write_text(
            yaml.safe_dump({
                "suite": "loopback1", "dataset": "cases.jsonl", "concurrency": 1, "target": {"callable": "target.py:answer"},
                "judges": [{"type": "contains", "name": "says_answer", "values": ["answer"]}],
            }),
            encoding="utf-8",
        )

    def make_run(self, name: str, *extra: str) -> Path:
        out = self.dir / name
        code, _, err = cli("run", "-c", self.config, "--out", out, "--quiet", "--no-report", *extra)
        self.assertIn(code, (0, 3), err)
        return out

    @staticmethod
    def read_summary(run_dir: Path) -> dict:
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    def edit_summary(self, run_dir: Path, edit) -> Path:
        summary = self.read_summary(run_dir)
        edit(summary)
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        return run_dir

    def copy_run(self, source: Path, name: str) -> Path:
        import shutil

        target = self.dir / name
        shutil.copytree(source, target)
        return target


class BaselineSelectionTests(Fixture):
    def baseline(self, run: Path, *extra: str):
        return cli("baseline", run, "--to", self.dir / "bl", *extra)

    def warnings(self, err: str) -> list[str]:
        return [line for line in err.splitlines() if line.startswith("warning:")]

    def test_a_full_run_is_copied_without_a_word(self):
        code, out, err = self.baseline(self.make_run("full"))
        self.assertEqual((code, err), (0, ""))
        self.assertIn("copied to", out)

    def test_a_limit_warns_names_the_counts_and_still_copies(self):
        run = self.make_run("limited", "--limit", "2")
        code, out, err = self.baseline(run)
        self.assertEqual(code, 0)
        self.assertIn("copied to", out)
        [warning] = self.warnings(err)
        self.assertEqual(err.strip().count("\n"), 0, err)  # one line
        self.assertIn(f"run {self.read_summary(run)['run_id']} was limited (--limit 2)", warning)
        self.assertIn("covers only 2 of the 6 cases the config selects", warning)
        self.assertIn("the other 4 show up as new cases", warning)
        self.assertEqual(len((self.dir / "bl" / "results.jsonl").read_text(encoding="utf-8").splitlines()), 2)  # copied as it is: the warning refuses nothing

    def test_tags_are_named(self):
        _, _, err = self.baseline(self.make_run("tagged", "--tags", "a"))
        self.assertIn("(--tags a)", err)
        self.assertIn("covers only 3 of the 6 cases", err)

    def test_every_selector_is_named(self):
        _, _, err = self.baseline(self.make_run("both", "--tags", "a,b", "--limit", "4"))
        self.assertIn("(--limit 4, --tags a,b)", err)

    def test_ids_given_to_the_api_are_named(self):
        config = SuiteConfig.load(self.config)
        run = runner.run_suite(config, out_dir=self.dir / "ids", ids=["c1", "c4"], progress=False)
        code, _, err = self.baseline(run.run_dir)
        self.assertEqual(code, 0)
        self.assertIn("(ids c1,c4)", err)
        self.assertIn("covers only 2 of the 6 cases", err)

    def test_a_long_list_of_ids_is_cut(self):
        run = self.make_run("full")
        self.edit_summary(run, lambda s: s.update(selection={"limited": True, "n_selected": 8, "n_config": 20, "limit": None, "tags": None, "ids": [f"case-{i}" for i in range(8)]}))
        _, _, err = self.baseline(run)
        self.assertIn("ids case-0,case-1,case-2,case-3,case-4 (+3 more)", err)

    def test_a_selector_that_covers_every_case_does_not_warn(self):
        for name, extra in (("limit", ("--limit", "6")), ("limit-more", ("--limit", "50")), ("tags", ("--tags", "a,b"))):
            with self.subTest(selector=name):
                run = self.make_run(name, *extra)
                self.assertIs(self.read_summary(run)["selection"]["limited"], False)
                code, _, err = self.baseline(run)
                self.assertEqual((code, err), (0, ""))

    def test_a_summary_without_selection_says_it_does_not_record_it(self):
        run = self.edit_summary(self.make_run("old"), lambda s: s.pop("selection"))
        code, out, err = self.baseline(run)
        self.assertEqual(code, 0)
        self.assertIn("copied to", out)
        [warning] = self.warnings(err)
        self.assertEqual(err.strip().count("\n"), 0, err)
        self.assertIn("does not record which cases it covers", warning)
        self.assertIn("older evalsprint", warning)

    def test_a_selection_that_cannot_be_read_counts_as_not_recorded(self):
        for bad in ("everything", None, 7, [], {"limited": "yes"}, {"n_selected": 2}):
            with self.subTest(selection=bad):
                run = self.edit_summary(self.make_run(f"bad{abs(hash(repr(bad)))}"), lambda s: s.update(selection=bad))
                code, _, err = self.baseline(run)
                self.assertEqual(code, 0, err)
                self.assertIn("does not record which cases it covers", err)

    def test_values_that_come_from_the_artifact_stay_on_one_line(self):
        run = self.make_run("hostile")
        hostile = {"limited": True, "n_selected": "2\nerror: INJECTED", "n_config": [1], "limit": "7\x1b[31m\nINJECTED", "tags": ["x\nINJECTED", 3, None], "ids": ["i\rINJECTED"]}
        self.edit_summary(run, lambda s: s.update(selection=hostile))
        code, _, err = self.baseline(run)
        self.assertEqual(code, 0, err)
        self.assertEqual(err.strip().count("\n"), 0, err)
        self.assertNotIn("\x1b", err)
        self.assertIn("was limited", err)

    def test_exit_codes_do_not_change(self):
        # an unhealthy run is still refused (3) and nothing is copied, limited or not; --force copies it and says both things
        run = self.make_run("down", "--limit", "3", "--set", "target.params.down=true")
        code, out, err = self.baseline(run)
        self.assertEqual(code, 3, err)
        self.assertIn("refusing to use run", err)
        self.assertFalse((self.dir / "bl").exists())
        code, out, err = self.baseline(run, "--force")
        self.assertEqual(code, 0, err)
        self.assertIn("--force: this run is unhealthy", err)
        self.assertIn("was limited (--limit 3)", err)
        self.assertTrue((self.dir / "bl" / "summary.json").is_file())

    def test_the_warning_is_not_printed_when_the_baseline_is_refused(self):
        run = self.make_run("down2", "--limit", "3", "--set", "target.params.down=true")
        _, _, err = self.baseline(run)
        self.assertNotIn("was limited", err)

    def test_compare_against_a_limited_baseline_is_what_the_warning_describes(self):
        # the engineering critic's case: a baseline of 2 of 6 cases and a full candidate pair only those 2; the other 4 are "new cases"
        base = self.make_run("sub", "--limit", "2")
        full = self.make_run("full")
        out_json = self.dir / "cmp.json"
        code, _, _ = cli("compare", base, full, "--json", out_json)
        self.assertEqual(code, 0)
        comparison = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(comparison["n_paired"], 2)
        self.assertEqual(len(comparison["unpaired"]["only_candidate"]), 4)
        self.assertEqual(self.read_summary(base)["selection"], {"limited": True, "n_selected": 2, "n_config": 6, "limit": 2, "tags": None, "ids": None})


class NonNumericFieldTests(unittest.TestCase):
    def setUp(self):
        self.summary = {
            "n_cases": 3, "pass_rate": 0.5, "pass_rate_ci": [0.2, 0.8], "error_rate": 0.0, "duration_s": 1.5, "target_retries": 1,
            "stability": {"n_repeated": 0, "flip_rate": None},
            "by_tag": {"a": {"n": 3, "n_passed": 1, "pass_rate": 0.33, "ci": [0.1, 0.7]}},
            "by_judge": {"j": {"n": 3, "n_passed": 1, "pass_rate": 0.33, "ci": [0.1, 0.7], "mean_score": 0.4, "n_errors": 0, "cost_usd": 0.0}},
            "cost": {"system_total_usd": 0.1, "system_mean_usd": None, "system_per_1k_usd": None, "judge_total_usd": 0.0, "n_priced": 0},
            "latency": {"mean_s": None, "p50_s": None, "p95_s": None, "max_s": None},
            "tokens": {"input": 1, "output": 2, "cache_read": 0, "cache_write": 0},
            "models": {"claude-haiku-4-5": 3},
            "repeat": {"n": 1, "per_tag": {"safety": 5}},
            "selection": {"limited": False, "n_selected": 3, "n_config": 3, "limit": None, "tags": None, "ids": None},
        }

    def test_a_summary_as_the_runner_writes_it_has_none(self):
        self.assertIsNone(_non_numeric_field(self.summary))
        self.assertIsNone(_non_numeric_field({}))  # an older summary lacks fields: fine

    def test_it_names_the_field_with_its_dotted_path(self):
        cases = [
            (("pass_rate",), "x", "pass_rate"),
            (("n_cases",), "3", "n_cases"),
            (("by_tag", "a", "n"), "3", "by_tag.a.n"),
            (("by_judge", "j", "cost_usd"), "1", "by_judge.j.cost_usd"),
            (("cost", "judge_total_usd"), [1], "cost.judge_total_usd"),
            (("models", "claude-haiku-4-5"), "3", "models.claude-haiku-4-5"),
            (("repeat", "per_tag", "safety"), "5", "repeat.per_tag.safety"),
            (("selection", "n_selected"), "2", "selection.n_selected"),
            (("stability", "flip_rate"), "0.5", "stability.flip_rate"),
        ]
        for path, value, name in cases:
            with self.subTest(field=name):
                forged = json.loads(json.dumps(self.summary))
                node = forged
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
                self.assertEqual(_non_numeric_field(forged), (name, value))

    def test_a_pair_names_the_bad_end(self):
        forged = dict(self.summary, pass_rate_ci=[0.2, "x"])
        self.assertEqual(_non_numeric_field(forged), ("pass_rate_ci[1]", "x"))
        self.assertEqual(_non_numeric_field(dict(self.summary, pass_rate_ci="0.2-0.8")), ("pass_rate_ci", "0.2-0.8"))
        self.assertEqual(_non_numeric_field(dict(self.summary, pass_rate_ci=[0.1])), ("pass_rate_ci", [0.1]))

    def test_null_is_a_value_that_does_not_exist_and_a_bool_is_not_a_number(self):
        self.assertIsNone(_non_numeric_field(dict(self.summary, pass_rate=None, pass_rate_ci=None)))
        self.assertEqual(_non_numeric_field(dict(self.summary, n_cases=True)), ("n_cases", True))

    def test_a_section_that_is_not_an_object_is_named(self):
        self.assertEqual(_non_numeric_field(dict(self.summary, by_tag=[1, 2])), ("by_tag", [1, 2]))
        self.assertEqual(_non_numeric_field(dict(self.summary, cost="free")), ("cost", "free"))


class ForgedSummaryTests(Fixture):
    def setUp(self):
        super().setUp()
        self.base = self.make_run("base")
        self.cand = self.make_run("cand")

    def forged(self, name: str, edit) -> Path:
        return self.edit_summary(self.copy_run(self.cand, name), edit)

    def test_report_names_the_field_in_one_line(self):
        cases = {
            "pass_rate": lambda s: s.update(pass_rate="x<b>"),
            "n_cases": lambda s: s.update(n_cases="6"),
            "by_tag.a.n": lambda s: s["by_tag"]["a"].update(n="3"),
            "by_tag.a.pass_rate": lambda s: s["by_tag"]["a"].update(pass_rate="1"),
            "cost.judge_total_usd": lambda s: s["cost"].update(judge_total_usd="0"),
            "latency.p95_s": lambda s: s["latency"].update(p95_s="fast"),
        }
        for field, edit in cases.items():
            run = self.forged("f_" + field.replace(".", "_"), edit)
            for how, argv in (("with a baseline", ("report", run, "--baseline", self.base, "-o", self.dir / "r.html")), ("alone", ("report", run, "-o", self.dir / "r2.html"))):
                with self.subTest(field=field, report=how):
                    code, out, err = cli(*argv)
                    self.assertEqual(code, 2, err)  # the exit code report had for these summaries
                    self.assertEqual(err.strip().count("\n"), 0, err)
                    self.assertTrue(err.startswith("error: "), err)
                    self.assertIn(f"the field '{field}' should be a number", err)
                    self.assertIn("summary.json", err)
                    self.assertNotIn("TypeError", err)
                    self.assertNotIn("Traceback", err)
                    self.assertNotIn("Unknown format code", err)
                    self.assertEqual(out, "")

    def test_the_engineering_critics_hostile_value_stays_inert_and_short(self):
        hostile = '</script><img src=x onerror=alert("X")>' + "A" * 500
        run = self.forged("hostile", lambda s: s.update(pass_rate=hostile))
        code, _, err = cli("report", run, "--baseline", self.base, "-o", self.dir / "r.html")
        self.assertEqual(code, 2)
        self.assertEqual(err.strip().count("\n"), 0)
        self.assertLess(len(err), 400)
        self.assertIn("the field 'pass_rate'", err)

    def test_report_names_the_first_summary_that_has_a_bad_field_baseline_included(self):
        bad_base = self.edit_summary(self.copy_run(self.base, "bad_base"), lambda s: s.update(n_cases="6"))
        # the baseline's summary is not read by report, so a forged baseline is tolerated as before ...
        code, _, err = cli("report", self.cand, "--baseline", bad_base, "-o", self.dir / "r.html")
        self.assertEqual((code, err), (0, ""))
        # ... and when the report fails for the candidate's field, the candidate is the one named
        run = self.forged("both", lambda s: s.update(pass_rate="x"))
        _, _, err = cli("report", run, "--baseline", bad_base, "-o", self.dir / "r.html")
        self.assertIn("both/summary.json", err)
        self.assertIn("'pass_rate'", err)

    def test_a_summary_report_copes_with_is_still_fine(self):
        for field, edit in {
            "n_oversize": lambda s: s.update(n_oversize="x"),
            "duration_s": lambda s: s.update(duration_s="long"),
            "error_rate": lambda s: s.update(error_rate="0"),
            "target_retries": lambda s: s.update(target_retries="one"),
            "selection.n_selected": lambda s: s["selection"].update(n_selected="all"),
        }.items():
            with self.subTest(field=field):
                run = self.forged("ok_" + field.replace(".", "_"), edit)
                code, out, err = cli("report", run, "-o", self.dir / "ok.html")
                self.assertEqual((code, err), (0, ""))  # the exit code did not change: only a crash gets the new message
                self.assertIn("Report written", out)

    def test_other_failures_keep_their_own_message(self):
        # a gate setting that is not valid is already named by the gate; the diagnosis must not replace it
        run = self.forged("gate", lambda s: s["gate"].update(alpha="x"))
        code, _, err = cli("report", run, "--baseline", self.base, "-o", self.dir / "r.html")
        self.assertEqual(code, 2)
        self.assertIn("gate.alpha must be a number", err)
        self.assertNotIn("the field", err)
        code, _, err = cli("compare", self.base, run)
        self.assertEqual(code, 2)
        self.assertIn("gate.alpha must be a number", err)

    def test_compare_decides_from_the_records_so_a_forged_number_changes_nothing(self):
        for field, edit in {"pass_rate": lambda s: s.update(pass_rate="x"), "n_cases": lambda s: s.update(n_cases="6"), "by_tag.a.n": lambda s: s["by_tag"]["a"].update(n="3")}.items():
            with self.subTest(field=field):
                run = self.forged("c_" + field.replace(".", "_"), edit)
                plain = cli("compare", self.base, self.cand)
                forged = cli("compare", self.base, run)
                self.assertEqual(forged[0], plain[0])  # the same exit code, the same markdown but for the run ids
                self.assertEqual(forged[2], "")
                scrub = lambda text: re.sub(r"`[^`]+`", "`id`", text)  # noqa: E731
                self.assertEqual(scrub(forged[1]), scrub(plain[1]))

    def test_a_crash_inside_compare_names_the_forged_field(self):
        run = self.forged("crash", lambda s: s.update(pass_rate="x"))
        boom = TypeError("bad operand type for abs(): 'str'")
        with mock.patch("evalsprint.compare.compare_runs", side_effect=boom):
            code, out, err = cli("compare", self.base, run)
        self.assertEqual(code, 2)  # the exit code of any failure of compare: 1 stays "the gate failed"
        self.assertEqual(err.strip().count("\n"), 0)
        self.assertIn("the field 'pass_rate' should be a number", err)
        self.assertEqual(out, "")
        with mock.patch("evalsprint.compare.compare_runs", side_effect=boom):
            code, _, err = cli("compare", self.base, self.cand)  # nothing forged: the message is the one it always was
        self.assertEqual(code, 2)
        self.assertIn("TypeError: bad operand type for abs()", err)

    def test_the_baseline_command_names_the_field_too(self):
        run = self.forged("health", lambda s: s.update(error_rate="0"))
        code, out, err = cli("baseline", run, "--to", self.dir / "bl")
        self.assertEqual(code, 2)
        self.assertEqual(err.strip().count("\n"), 0)
        self.assertIn("the field 'error_rate' should be a number", err)
        self.assertFalse((self.dir / "bl").exists())


class RelabelHelpTests(unittest.TestCase):
    SENTENCE = (
        "Skip every case that already has a label for its current output. Only the changed outputs need a label when the earlier labels covered every "
        "unchanged case; otherwise the export also draws never-labeled unchanged cases."
    )

    def test_the_help_of_previous_labels_states_its_condition(self):
        code, out, _ = cli("label", "--help")
        self.assertEqual(code, 0)
        self.assertIn(self.SENTENCE, " ".join(out.split()))

    def test_the_overview_no_longer_promises_only_the_changed_outputs(self):
        code, out, _ = cli("--help")
        self.assertEqual(code, 0)
        flat = " ".join(out.split())
        self.assertNotIn("only the outputs that changed since those labels", flat)
        self.assertIn("--previous-labels labels.jsonl skip the cases that already have a label for their current output", flat)


class TargetRetriesLineTests(Fixture):
    def test_the_line_reads_the_summary_not_the_config(self):
        real = runner.run_suite

        def with_summary(*args, **kwargs):
            result = real(*args, **kwargs)
            result.summary["target_retries"] = 4  # what the run recorded, whatever the config says now
            result.summary["n_target_retries"] = 2
            return result

        with mock.patch("evalsprint.cli.run_suite", side_effect=with_summary):
            code, out, err = cli("run", "-c", self.config, "--out", self.dir / "o", "--quiet", "--no-report")
        self.assertEqual(code, 0, err)
        self.assertRegex(out, r"Target retries:\s+2 call\(s\) repeated after a crash \(target_retries: 4\)")

    def test_a_real_run_prints_the_configured_value(self):
        marker = self.dir / "flaky"
        (self.dir / "target.py").write_text(
            "import os\n"
            "def answer(input, params):\n"
            f"    seen = {str(marker)!r} + input\n"
            "    if input == 'q1' and not os.path.exists(seen):\n"
            "        open(seen, 'w').close()\n"
            "        raise TimeoutError('once')\n"
            "    return {'text': 'answer to ' + input}\n",
            encoding="utf-8",
        )
        code, out, err = cli("run", "-c", self.config, "--out", self.dir / "o", "--quiet", "--no-report", "--set", "target_retries=3")
        self.assertEqual(code, 0, err)
        self.assertRegex(out, r"Target retries:\s+1 call\(s\) repeated after a crash \(target_retries: 3\)")
        self.assertEqual(self.read_summary(self.dir / "o")["target_retries"], 3)


if __name__ == "__main__":
    unittest.main()
