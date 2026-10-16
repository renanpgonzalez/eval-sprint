"""`evalsprint noise RUN` (contracts C19 and C21): the false alarm and the detection the WHOLE gate (pass-rate rule and critical-case rule) will
have on THIS suite, from the attempts of every scored case, the cases that carry a critical tag and the flip rate its repeated cases measured
(`compare.expected_gate_rates_protocol`, `stats.attempt_noise_estimate`).

The fixture target is deterministic: the "flaky" cases fail on attempt 1 only, so a run repeated 3 times has a known flip rate.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from unittest import mock

from evalsprint import compare, stats
from evalsprint.cli import _pct, _rate_pct, main

TARGET = '''\
def answer(input, params):
    attempt = params.get("_attempt") or 0
    if input.startswith("flaky") and attempt == 1:
        return {"text": "no idea"}
    return {"text": "answer to " + input}
'''


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


class NoiseFixture(unittest.TestCase):
    N_CASES, N_FLAKY = 20, 4

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "suite"
        self.dir.mkdir()
        (self.dir / "target.py").write_text(TARGET, encoding="utf-8")
        self.write_cases(self.N_FLAKY)

    def write_cases(self, n_flaky: int):
        rows = [{"id": f"f{i}", "input": f"flaky {i}", "tags": ["flaky", "all"]} for i in range(n_flaky)]
        rows += [{"id": f"s{i}", "input": f"stable {i}", "tags": ["all"]} for i in range(self.N_CASES - n_flaky)]
        (self.dir / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def config(self, **extra) -> Path:
        path = self.dir / "eval.yaml"
        path.write_text(
            yaml.safe_dump({"suite": "noise", "dataset": "cases.jsonl", "target": {"callable": "target.py:answer"}, "judges": [{"type": "contains", "name": "ans", "values": ["answer"]}], **extra}),
            encoding="utf-8",
        )
        return path

    def run_dir(self, name="run", *extra, **config) -> Path:
        cfg = self.config(**config)
        out = self.dir / name
        code, _, err = cli("run", "-c", cfg, "--out", out, "--quiet", "--no-report", *extra)
        self.assertEqual(code, 0, err)
        return out

    @staticmethod
    def summary(run_dir: Path) -> dict:
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


class NothingRepeatedTests(NoiseFixture):
    def test_it_says_so_and_asks_for_repeat_3(self):
        code, out, err = cli("noise", self.run_dir())
        self.assertEqual((code, err), (0, ""))
        self.assertIn("Nothing was repeated", out)
        self.assertIn("--repeat 3", out)
        self.assertNotIn("False alarm", out)  # no made-up numbers without a measurement

    def test_a_repeat_of_one_is_nothing_repeated(self):
        code, out, _ = cli("noise", self.run_dir("once", "--repeat", "1"))
        self.assertEqual(code, 0)
        self.assertIn("Nothing was repeated", out)


class RepeatedRunTests(NoiseFixture):
    def setUp(self):
        super().setUp()
        self.run = self.run_dir("rep", "--repeat", "3")
        self.s = self.summary(self.run)

    def test_the_fixture_measures_the_expected_flip_rate(self):
        self.assertEqual(self.s["stability"]["n_repeated"], self.N_CASES)
        self.assertAlmostEqual(self.s["stability"]["flip_rate"], self.N_FLAKY / self.N_CASES)

    def test_the_table_prints_every_measure(self):
        code, out, err = cli("noise", self.run)
        self.assertEqual((code, err), (0, ""))
        for expected in (
            "Measured on this run", "Scored cases (n)", "Repeated cases", "Flip rate", "Noise per attempt", "Noise per case verdict",
            "What the gate does on a suite like this one", "False alarm", "Detection, 5-point drop", "Detection, 10-point drop", "Baseline lottery",
        ):
            self.assertIn(expected, out)
        self.assertIn("20 of 20", out)
        self.assertIn("(3 attempts each)", out)
        self.assertIn("majority of 3 attempts", out)

    def test_the_figures_are_the_ones_compare_enumerates(self):
        flip = self.s["stability"]["flip_rate"]
        attempt_noise = stats.noise_from_flip_rate(flip, [3] * self.N_CASES)
        rates = compare.expected_gate_rates_protocol([3] * self.N_CASES, [False] * self.N_CASES, attempt_noise, self.s["pass_rate"], (0.05, 0.10), self.s.get("gate"))
        _, out, _ = cli("noise", self.run)
        fa = rates["false_alarm"]
        self.assertRegex(out, rf"False alarm:\s+about {_rate_pct(fa['whole_gate'])} of the pull requests that change nothing are blocked")
        self.assertRegex(out, rf"pass-rate rule alone:\s+about {_rate_pct(fa['pass_rate_rule'])}\n")
        self.assertRegex(out, rf"critical-case rule alone:\s+about {_rate_pct(fa['critical_rule'])}")
        self.assertRegex(out, rf"Detection, 5-point drop:\s+about {_rate_pct(rates['detection'][0.05]['whole_gate'])}")
        self.assertRegex(out, rf"Detection, 10-point drop:\s+about {_rate_pct(rates['detection'][0.1]['whole_gate'])}")
        self.assertIn(_rate_pct(rates["baseline_lottery"]["p_fa_over_15"]), out)
        self.assertIn(f"about {_pct(attempt_noise)}  (the chance that one attempt", out)

    def test_it_prints_the_three_false_alarm_lines_the_whole_gate_first(self):
        _, out, _ = cli("noise", self.run)
        lines = [line.strip() for line in out.splitlines()]
        self.assertEqual(sum(1 for line in lines if line.startswith("False alarm:")), 1)
        self.assertEqual(sum(1 for line in lines if line.startswith("pass-rate rule alone:")), 1)
        self.assertEqual(sum(1 for line in lines if line.startswith("critical-case rule alone:")), 1)
        self.assertIn("(the whole gate: either rule trips)", out)
        self.assertIn("no critical case: the rule does not apply", out)  # this fixture's gate has no critical tags

    def test_the_per_attempt_noise_is_not_the_flip_rate(self):
        flip = self.s["stability"]["flip_rate"]
        attempt_noise = stats.noise_from_flip_rate(flip, [3] * self.N_CASES)
        self.assertNotAlmostEqual(attempt_noise, flip)  # the raw flip rate must never be passed as the noise
        _, out, _ = cli("noise", self.run)
        self.assertIn(f"about {_pct(attempt_noise)}  (the chance that one attempt", out)
        self.assertIn(f"{_pct(flip)} of the repeated cases had attempts that disagree", out)

    def test_it_prints_the_plain_english_gate_paragraph(self):
        _, out, _ = cli("noise", self.run)
        self.assertIn("Read every figure as \"about\"", out)
        self.assertIn("The noise measured on the repeated cases is assumed to be the noise of every case", " ".join(out.split()))  # the assumption, said out loud
        self.assertIn("baseline", out.lower())
        self.assertIn("one-sided exact McNemar", out)  # the auto rule, named

    def test_it_does_not_write_anything_in_the_run_directory(self):
        before = {p.name: p.read_bytes() for p in self.run.iterdir()}
        cli("noise", self.run)
        self.assertEqual({p.name: p.read_bytes() for p in self.run.iterdir()}, before)

    def test_more_flips_mean_more_false_alarm(self):
        def false_alarm(n_flaky: int) -> float:
            self.write_cases(n_flaky)
            summary = self.summary(self.run_dir(f"flaky{n_flaky}", "--repeat", "3"))
            attempt = stats.noise_from_flip_rate(summary["stability"]["flip_rate"], [3] * self.N_CASES)
            return compare.expected_gate_rates(self.N_CASES, stats.majority_noise(attempt, 3), summary["pass_rate"])["false_alarm"]

        self.assertLess(false_alarm(1), false_alarm(10))

    def test_no_flip_at_all_prices_the_upper_95_bound_and_says_so(self):
        self.write_cases(0)
        run = self.run_dir("calm", "--repeat", "3")
        code, out, _ = cli("noise", run)
        self.assertEqual(code, 0)
        self.assertIn("No repeated case disagreed with itself", out)
        self.assertIn("upper 95% bound", out)
        estimate = stats.attempt_noise_estimate(0, [3] * self.N_CASES)
        self.assertEqual(estimate["noise"], 0.0)
        self.assertGreater(estimate["noise_upper95"], 0.04)  # 0 flips of 20 cases still allows about 4.9% noise per attempt
        self.assertIn(f"the figures below use the upper 95% bound, about {_pct(estimate['noise_upper95'])}", out)
        rates = compare.expected_gate_rates_protocol([3] * self.N_CASES, [False] * self.N_CASES, estimate["noise_upper95"], 1.0)
        self.assertRegex(out, rf"False alarm:\s+about {_rate_pct(rates['false_alarm']['whole_gate'])} of the pull requests.*priced at the upper 95% bound of the noise")
        self.assertRegex(out, rf"Detection, 10-point drop:\s+about {_rate_pct(rates['detection'][0.1]['whole_gate'])}\s+\(priced at the upper 95% bound")
        zero = compare.expected_gate_rates_protocol([3] * self.N_CASES, [False] * self.N_CASES, 0.0, 1.0)
        self.assertGreater(rates["false_alarm"]["whole_gate"], zero["false_alarm"]["whole_gate"])  # a ceiling, not zero

    def test_a_run_with_flips_does_not_claim_an_upper_bound(self):
        _, out, _ = cli("noise", self.run)
        self.assertNotIn("upper 95% bound", out)

    def test_an_even_number_of_attempts_says_a_tie_is_a_fail(self):
        _, out, _ = cli("noise", self.run_dir("two", "--repeat", "2"))
        self.assertIn("An even number of attempts counts a tie as a fail", out)
        _, out, _ = cli("noise", self.run)
        self.assertNotIn("a tie as a fail", out)  # 3 attempts: no tie


class ProtocolTests(NoiseFixture):
    def test_fewer_than_20_repeated_cases_are_not_estimated_and_the_fix_is_named(self):
        run = self.run_dir("pertag", repeat={"n": 1, "per_tag": {"flaky": 3}})
        s = self.summary(run)
        self.assertEqual(s["stability"]["n_repeated"], self.N_FLAKY)  # only the tagged cases ran more than once
        code, out, err = cli("noise", run)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("4 of 20", out)
        self.assertIn("Noise per attempt: not estimated", out)
        self.assertIn("Only 4 of the 20 scored cases were repeated", out)
        self.assertIn("evalsprint run -c <eval.yaml> --repeat 3", out)
        self.assertIn("repeat: {n: 3, per_tag: {<critical tag>: 5}}", out)
        for figure in ("False alarm", "Detection", "Baseline lottery"):
            self.assertNotIn(figure, out)  # no rate is priced on a noise nobody measured

    def test_the_text_for_a_partly_repeated_suite_is_exactly_the_one_the_demo_quotes(self):
        """Only the suites whose every case is already repeated got a new text; a partly repeated one (the demo: 8 critical cases of 45) keeps the old words."""
        (self.dir / "cases.jsonl").write_text(
            "".join(json.dumps({"id": f"c{i}", "input": ("flaky " if i % 9 == 0 else "stable ") + str(i), "tags": ["safety" if i < 8 else "other"]}) + "\n" for i in range(45)),
            encoding="utf-8",
        )
        run = self.run_dir("demo_like", repeat={"n": 1, "per_tag": {"safety": 5}})
        code, out, err = cli("noise", run)
        self.assertEqual((code, err), (0, ""))
        flat = " ".join(out.split())
        self.assertIn(
            "Noise: not estimated. Only 8 of the 45 scored cases were repeated, and fewer than 20 repeated cases give no stable estimate: one case that flips or not "
            "moves the flip rate by 12 points. The false-alarm and detection rates of the gate are not printed, because they would be priced on a noise nobody "
            "measured. Repeat every case, and the critical ones more often:",
            flat,
        )
        self.assertIn("\n    evalsprint run -c <eval.yaml> --repeat 3\n    # or in eval.yaml:  repeat: {n: 3, per_tag: {<critical tag>: 5}}\n  then run `evalsprint noise` on that run (every case then counts, 45 of them here).\n", out)
        self.assertNotIn("itself is below", flat)
        self.assertNotIn("Grow the suite", flat)

    def test_the_text_for_4_of_20_repeated_cases_is_unchanged(self):
        run = self.run_dir("pertag", repeat={"n": 1, "per_tag": {"flaky": 3}})
        _, out, _ = cli("noise", run)
        flat = " ".join(out.split())
        self.assertIn("Repeat every case, and the critical ones more often:", flat)
        self.assertIn("one case that flips or not moves the flip rate by 25 points", flat)
        self.assertIn("(every case then counts, 20 of them here)", flat)
        self.assertNotIn("itself is below", flat)

    def test_the_threshold_is_20_repeated_cases(self):
        def repeated(n_repeated):
            (self.dir / "cases.jsonl").write_text(
                "".join(json.dumps({"id": f"c{i}", "input": ("flaky " if i % 5 == 0 else "stable ") + str(i), "tags": ["rep" if i < n_repeated else "once"]}) + "\n" for i in range(30)),
                encoding="utf-8",
            )
            return self.run_dir(f"r{n_repeated}", repeat={"n": 1, "per_tag": {"rep": 3}})

        _, out, _ = cli("noise", repeated(19))
        self.assertIn("not estimated", out)
        self.assertNotIn("False alarm", out)
        _, out, _ = cli("noise", repeated(20))
        self.assertNotIn("not estimated", out)
        self.assertIn("False alarm", out)

    def test_a_mixed_protocol_is_priced_case_by_case_not_with_a_per_attempt_stand_in(self):
        run = self.run_dir("mixed", repeat={"n": 3, "per_tag": {"flaky": 5}})
        code, out, _ = cli("noise", run)
        self.assertEqual(code, 0)
        self.assertIn("20 of 20  (3 to 5 attempts)", out)
        self.assertIn("each priced at the majority of its own attempts", out)
        ks = [5] * self.N_FLAKY + [3] * (self.N_CASES - self.N_FLAKY)
        estimate = stats.attempt_noise_estimate(self.N_FLAKY, ks)
        rates = compare.expected_gate_rates_protocol(ks, [False] * self.N_CASES, estimate["noise"], self.summary(run)["pass_rate"])
        self.assertIn(f"about {_pct(estimate['noise'])}  (the chance that one attempt", out)
        self.assertRegex(out, rf"False alarm:\s+about {_rate_pct(rates['false_alarm']['whole_gate'])} ")
        self.assertIn(f"about {_pct(rates['verdict_noise_mean'])}  (mean over the scored cases", out)

    def test_the_gate_stored_in_the_run_is_the_one_described(self):
        fixed = self.run_dir("fixed", "--repeat", "3", gate={"max_pass_rate_drop": 0.05})
        _, out, _ = cli("noise", fixed)
        self.assertIn("fixed: a pass-rate drop above 0.050", out)
        off = self.run_dir("off", "--repeat", "3", gate={"max_pass_rate_drop": None})
        _, out, _ = cli("noise", off)
        self.assertIn("its pass-rate rule is off", out)
        self.assertIn("False alarm", out)
        self.assertRegex(out, r"False alarm:\s+about 0.00% ")

    def test_an_unreadable_stored_gate_falls_back_to_the_default_with_a_note(self):
        run = self.run_dir("badgate", "--repeat", "3")
        summary = self.summary(run)
        summary["gate"] = {"max_pass_rate_drop": "lots"}
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        code, out, err = cli("noise", run)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("could not be read: the default gate is used", out)


class SuiteBelowTheFloorTests(NoiseFixture):
    """Every scored case is already repeated, but the suite has fewer than 20 cases (the 10 cases of the `evalsprint init` scaffold, `repeat: {n: 3}`):
    asking to "repeat every case" is circular, so the text says that the suite is what is too small (the mlops docs say the same: grow it to 20 or
    more cases, every one repeated, or use the tables)."""

    N_CASES, N_FLAKY = 10, 2

    def noise(self, name="scaffold", *extra, **config):
        code, out, err = cli("noise", self.run_dir(name, *extra, **config))
        self.assertEqual((code, err), (0, ""))
        return out, " ".join(out.split())

    def test_it_says_the_suite_is_too_small_and_does_not_ask_to_repeat_every_case(self):
        out, flat = self.noise("scaffold", "--repeat", "3")
        self.assertRegex(out, r"Repeated cases:\s+10 of 10  \(3 attempts each\)")
        self.assertIn("Noise per attempt: not estimated", out)
        self.assertIn(
            "Noise: not estimated. All 10 scored cases were repeated, but the suite itself is below 20 scored cases, and fewer than 20 repeated cases "
            "give no stable estimate: one case that flips or not moves the flip rate by 10 points. Repeating these cases more often does not change that. "
            "The false-alarm and detection rates of the gate are not printed, because they would be priced on a noise nobody measured. Grow the suite to 20 or "
            "more cases (10 more than now), every one repeated, and run `evalsprint noise` on that run:",
            flat,
        )
        for circular in ("Repeat every case", "every case then counts", "per_tag", "Only 10 of the 10", "the critical ones more often"):
            self.assertNotIn(circular, flat)

    def test_the_command_comes_after_growing_the_suite_and_the_tables_are_the_other_way_out(self):
        out, flat = self.noise("scaffold", "--repeat", "3")
        self.assertLess(flat.index("Grow the suite to 20 or more cases"), flat.index("evalsprint run -c <eval.yaml> --repeat 3"))
        self.assertEqual(out.count("evalsprint run -c"), 1)
        self.assertLess(flat.index("evalsprint run -c <eval.yaml> --repeat 3"), flat.index("Or read the exact tables of the README"))
        self.assertIn("Or read the exact tables of the README: they give the false alarm and the detection of the default gate by test-set size, for an assumed noise per case (2%, 5%, 10%) instead of a measured one.", flat)

    def test_no_rate_is_priced_whether_or_not_a_case_flipped(self):
        for name, cases_flaky in (("quiet", 0), ("noisy", 2)):
            with self.subTest(flaky=cases_flaky):
                self.write_cases(cases_flaky)
                out, flat = self.noise(name, "--repeat", "3")
                self.assertIn(f"({cases_flaky} of 10: too few cases to rest a number on)", flat)
                self.assertIn("itself is below 20 scored cases", flat)
                for figure in ("False alarm", "Detection", "Baseline lottery", "upper 95% bound", "about"):
                    self.assertNotIn(figure, out)

    def test_the_gate_note_still_ends_the_output(self):
        run = self.run_dir("badgate", "--repeat", "3")
        summary = self.summary(run)
        summary["gate"] = {"max_pass_rate_drop": "lots"}
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        code, out, err = cli("noise", run)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("itself is below 20 scored cases", " ".join(out.split()))
        self.assertTrue(out.rstrip().splitlines()[-1].startswith("note: the gate stored in this run"), out)

    def test_19_cases_all_repeated_name_the_one_missing_case_and_20_are_estimated(self):
        self.N_CASES = 19
        self.write_cases(2)
        out, flat = self.noise("nineteen", "--repeat", "3")
        self.assertIn("All 19 scored cases were repeated, but the suite itself is below 20 scored cases", flat)
        self.assertIn("moves the flip rate by 5 points", flat)
        self.assertIn("Grow the suite to 20 or more cases (1 more than now)", flat)
        self.assertNotIn("False alarm", out)
        self.N_CASES = 20
        self.write_cases(2)
        out, flat = self.noise("twenty", "--repeat", "3")
        self.assertNotIn("not estimated", out)
        self.assertNotIn("itself is below", flat)
        self.assertRegex(out, r"False alarm:\s+about ")

    def test_a_suite_of_one_case_is_still_told_to_grow(self):
        self.N_CASES = 1
        self.write_cases(0)
        out, flat = self.noise("one", "--repeat", "3")
        self.assertIn("The one scored case was repeated, but the suite itself is below 20 scored cases", flat)
        self.assertIn("Grow the suite to 20 or more cases (19 more than now)", flat)
        self.assertNotIn("False alarm", out)


class CriticScenarioTests(NoiseFixture):
    """The methodology critic's suite (cycle 4): 45 cases, 8 of them critical (`safety`), `repeat: {n: 3, per_tag: {safety: 5}}`, 5% noise per attempt.
    Exact figures of the whole gate: pass-rate rule 0.21%, critical-case rule 0.92%, whole gate 1.11% (the old command printed 10.5%, one attempt per case)."""

    N_CASES = 45
    FLAKY = (1, 5, 9, 14, 20, 30, 41)  # 7 of the 45 flip: a flip rate of 15.6%, a per-attempt noise of 4.94%

    def setUp(self):
        super().setUp()
        rows = [{"id": f"c{i:02d}", "input": ("flaky " if i in self.FLAKY else "stable ") + str(i), "tags": ["safety"] if i < 8 else ["general"]} for i in range(45)]
        (self.dir / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.run = self.run_dir("critic", repeat={"n": 3, "per_tag": {"safety": 5}}, gate={"critical_tags": ["safety"]})
        self.s = self.summary(self.run)
        self.ks = [5] * 8 + [3] * 37
        self.critical = [True] * 8 + [False] * 37

    def test_the_run_has_the_protocol_of_the_critic(self):
        self.assertEqual(self.s["repeat"], {"n": 3, "per_tag": {"safety": 5}})
        self.assertEqual(self.s["stability"]["n_repeated"], 45)
        self.assertEqual(self.s["gate"]["critical_tags"], ["safety"])

    def test_at_exactly_5_percent_noise_it_prints_the_critics_figures(self):
        estimate = {"noise": 0.05, "noise_upper95": 0.12, "n_repeated": 45, "n_flipped": 7, "flip_rate": 7 / 45, "flip_rate_upper95": 0.3}
        with mock.patch("evalsprint.stats.attempt_noise_estimate", return_value=estimate):
            code, out, err = cli("noise", self.run)
        self.assertEqual((code, err), (0, ""))
        self.assertRegex(out, r"pass-rate rule alone:\s+about 0\.21%\n")
        self.assertRegex(out, r"critical-case rule alone:\s+about 0\.92%\n")
        self.assertRegex(out, r"False alarm:\s+about 1\.11% of the pull requests that change nothing are blocked")
        self.assertNotIn("10.5", out)
        self.assertNotIn("10.6", out)
        self.assertRegex(out, r"Critical cases:\s+8\b")

    def test_the_real_estimate_from_the_7_flips_is_priced_the_same_way(self):
        estimate = stats.attempt_noise_estimate(7, self.ks)
        self.assertAlmostEqual(estimate["noise"], 0.0494, places=3)
        rates = compare.expected_gate_rates_protocol(self.ks, self.critical, estimate["noise"], self.s["pass_rate"], (0.05, 0.10), self.s["gate"])
        _, out, _ = cli("noise", self.run)
        fa = rates["false_alarm"]
        self.assertRegex(out, rf"False alarm:\s+about {_rate_pct(fa['whole_gate'])} ")
        self.assertRegex(out, rf"pass-rate rule alone:\s+about {_rate_pct(fa['pass_rate_rule'])}\n")
        self.assertRegex(out, rf"critical-case rule alone:\s+about {_rate_pct(fa['critical_rule'])}\n")
        for drop, label in ((0.05, "5"), (0.1, "10")):
            d = rates["detection"][drop]
            self.assertRegex(out, rf"Detection, {label}-point drop:\s+about {_rate_pct(d['whole_gate'])}\s+\(whole gate; the pass-rate rule alone: {_rate_pct(d['pass_rate_rule'])}\)")
        self.assertLess(fa["whole_gate"], 0.02)  # about 1%, not the 10% of a single attempt per case

    def test_the_old_figure_is_what_a_single_attempt_per_case_would_give(self):
        # what the command printed before C21: every case at the per-attempt noise, the pass-rate rule alone
        old = compare.expected_gate_rates(45, stats.attempt_noise_estimate(7, self.ks)["noise"], self.s["pass_rate"])
        self.assertGreater(old["false_alarm"], 0.10)
        _, out, _ = cli("noise", self.run)
        self.assertNotIn(_rate_pct(old["false_alarm"]), out)

    def test_it_names_the_critical_cases_and_the_rule(self):
        _, out, _ = cli("noise", self.run)
        flat = " ".join(out.split())
        self.assertIn("Critical tags: safety.", flat)
        self.assertIn("critical-case rule: any pass-to-fail flip of one of the 8 critical cases blocks", flat)


class GateSourceTests(NoiseFixture):
    """`noise` prices the gate `compare` would use, and says which one it took (C21)."""

    def setUp(self):
        super().setUp()
        self.run = self.run_dir("rep", "--repeat", "3")  # the gate stored in this run has no critical tags

    def gate_file(self, name="trusted.yaml", **gate) -> Path:
        path = self.dir / name
        path.write_text(yaml.safe_dump({"gate": gate}), encoding="utf-8")
        return path

    def test_without_a_flag_it_uses_the_gate_stored_in_the_run_and_says_so(self):
        _, out, _ = cli("noise", self.run)
        flat = " ".join(out.split())
        self.assertIn("Gate used: the gate stored in this run", flat)
        self.assertIn("Critical tags: none (the critical-case rule does not apply)", flat)

    def test_gate_flag_takes_the_critical_tags_of_that_file(self):
        code, out, err = cli("noise", self.run, "--gate", self.gate_file(critical_tags=["flaky"]))
        self.assertEqual((code, err), (0, ""))
        flat = " ".join(out.split())
        self.assertIn("trusted.yaml (--gate). Critical tags: flaky.", flat)
        self.assertRegex(out, r"Critical cases:\s+4\b")
        ks, critical = [3] * self.N_CASES, [True] * self.N_FLAKY + [False] * (self.N_CASES - self.N_FLAKY)
        noise = stats.attempt_noise_estimate(self.N_FLAKY, ks)["noise"]
        rates = compare.expected_gate_rates_protocol(ks, critical, noise, self.summary(self.run)["pass_rate"], gate={"critical_tags": ["flaky"]})
        self.assertRegex(out, rf"False alarm:\s+about {_rate_pct(rates['false_alarm']['whole_gate'])} ")
        self.assertGreater(rates["false_alarm"]["whole_gate"], rates["false_alarm"]["pass_rate_rule"])  # the critical rule adds to the pass-rate rule

    def test_the_trusted_gate_replaces_the_one_in_the_run_even_when_it_has_no_critical_tags(self):
        stored = self.run_dir("stored", "--repeat", "3", gate={"critical_tags": ["flaky"]})
        _, out, _ = cli("noise", stored)
        self.assertRegex(out, r"Critical cases:\s+4\b")
        _, out, _ = cli("noise", stored, "--gate", self.gate_file(max_pass_rate_drop=0.05))
        self.assertRegex(out, r"Critical cases:\s+0\b")  # a pull request cannot add critical tags (or drop them) for itself: the trusted file decides
        self.assertIn("fixed: a pass-rate drop above 0.050", out)

    def test_config_flag_reads_the_eval_yaml_gate_and_gate_flag_wins(self):
        cfg = self.config(gate={"critical_tags": ["flaky"]})
        _, out, _ = cli("noise", self.run, "-c", cfg)
        flat = " ".join(out.split())
        self.assertIn("eval.yaml (-c). Critical tags: flaky.", flat)
        _, out, _ = cli("noise", self.run, "-c", cfg, "--gate", self.gate_file(critical_tags=["all"]))
        self.assertIn("(--gate). Critical tags: all.", " ".join(out.split()))
        self.assertRegex(out, r"Critical cases:\s+20\b")

    def test_gate_config_is_an_alias_of_gate(self):
        _, out, _ = cli("noise", self.run, "--gate-config", self.gate_file(critical_tags=["flaky"]))
        self.assertIn("(--gate). Critical tags: flaky.", " ".join(out.split()))

    def test_a_gate_file_that_is_not_valid_is_a_one_line_usage_error(self):
        bad = self.gate_file("bad.yaml", max_pass_rate_dorp=0.05)
        code, out, err = cli("noise", self.run, "--gate", bad)
        self.assertEqual(code, 2)
        self.assertEqual(err.count("\n"), 1, err)
        self.assertIn("unknown gate setting", err)
        self.assertEqual(out, "")
        code, _, err = cli("noise", self.run, "--gate", self.dir / "nope.yaml")
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("error:"), err)

    def test_when_nothing_was_repeated_it_still_says_which_gate(self):
        _, out, _ = cli("noise", self.run_dir("once"), "--gate", self.gate_file(critical_tags=["flaky"]))
        self.assertIn("Nothing was repeated", out)
        self.assertIn("(--gate). Critical tags: flaky.", " ".join(out.split()))


class WordingTests(NoiseFixture):
    def test_no_output_of_the_command_says_pessimistic_bound(self):
        outputs = []
        outputs.append(cli("noise", self.run_dir("once"))[1])
        outputs.append(cli("noise", self.run_dir("pertag", repeat={"n": 1, "per_tag": {"flaky": 3}}))[1])
        outputs.append(cli("noise", self.run_dir("mixed", repeat={"n": 3, "per_tag": {"flaky": 5}}))[1])
        outputs.append(cli("noise", self.run_dir("rep", "--repeat", "3"))[1])
        self.write_cases(0)
        outputs.append(cli("noise", self.run_dir("calm", "--repeat", "3"))[1])
        self.assertTrue(all(outputs))
        for out in outputs:
            self.assertNotRegex(out.lower(), r"pessimistic")

    def test_the_help_describes_the_whole_gate(self):
        _, out, _ = cli("noise", "--help")
        flat = " ".join(out.split())
        self.assertIn("critical-case rule", flat)
        self.assertIn("--gate", out)
        self.assertIn("at least 20 repeated cases", flat)


class UsageTests(NoiseFixture):
    def test_a_missing_run_is_a_usage_error(self):
        code, _, err = cli("noise", self.dir / "nope")
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("error:"))

    def test_a_run_where_nothing_could_be_scored_is_a_usage_error(self):
        run = self.run_dir("none", "--repeat", "3")
        summary = self.summary(run)
        summary["n_scored"] = 0
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        code, _, err = cli("noise", run)
        self.assertEqual(code, 2)
        self.assertIn("could be scored", err)

    def test_it_is_in_the_help(self):
        code, out, _ = cli("--help")
        self.assertIn("noise", out)
        code, out, _ = cli("noise", "--help")
        self.assertEqual(code, 0)
        self.assertIn("RUN", out)


if __name__ == "__main__":
    unittest.main()
