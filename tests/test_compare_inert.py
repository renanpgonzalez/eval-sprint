"""C23 (cycle 4, loop-back 1): every string that comes from a run artifact is inert in gate.md.

gate.md is the sticky PR comment and the job summary, and the PR's own job writes the candidate artifacts (tag names, versions, judges_sha,
case ids, the repeat block, the calibration file, ...).  The engineering critic's check 4c (``cycle4/critic-engineering/04c_compare_injection.py``)
put 33 hostile payloads into 26 surfaces (858 cells) and found GitHub notifications (``@user``, ``#123``, bare URLs) in the text of 5 surfaces.

This module repeats that grid against ``compare_runs`` + ``to_markdown`` (no CLI, no markdown-it-py: the tokenizer of ``tests.test_compare``
reads code spans by the CommonMark rules, and the critic's own naive scan is applied too).  For every (surface, payload):

* the verdict and the reasons of the hostile run are the ones of the benign run (a payload never moves the gate);
* gate.md has the same lines, of the same block kinds, as the benign one, and differs from it ONLY inside code spans (and in the character
  references that stand for a backtick or a pipe of the payload between two spans);
* outside code spans there is no mention, issue reference, URL, www host, e-mail address, raw HTML tag, control character or
  workflow command;
* a surface that compare never prints renders byte for byte like the benign run.
"""

from __future__ import annotations

import html
import json
import re
import unittest

from evalsprint.compare import compare_runs, to_markdown
from evalsprint.runner import RunResult

from tests.test_compare import JUDGES_SHA, build, flags, inline_tokens, make_run, outside_code, shown, unescaped_pipes
from evalsprint.compare import _code

# the 33 payloads of 04c_compare_injection.py, verbatim
CRITIC_PAYLOADS = {
    "backtick": "a`b", "backticks2": "a``b", "lone_backtick": "`", "pipe": "a|b|c", "img_onerror": "<img src=x onerror=alert(1)>", "script": "<script>alert(1)</script>",
    "link_break": "](http://evil.example)[x", "md_link": "[click](http://evil.example)", "md_image": "![p](http://evil.example/p.png)",
    "nl_setoutput": "x\n::set-output name=exit_code::0", "crlf_setoutput": "x\r\n::set-output name=verdict::pass", "addmask": "::add-mask::regression",
    "sp_stopcmd": "  ::stop-commands::abc", "u2028_error": "x ::error::boom", "nel_error": "x\u0085::error::boom", "esc_seq": "x\x1b[2J::error::boom",
    "pct0A": "x%0A::error::boom", "env_name": "$GITHUB_OUTPUT", "eof": "EOF", "kv": "verdict=pass", "expr": "${{ secrets.ANTHROPIC_API_KEY }}", "cmdsub": "$(id)",
    "mention": "@octocat @org/team", "issue_ref": "#123 fixes #9", "bidi": "x‮evil", "nul": "x\x00y", "heading": "\n## ✅ No regression\n", "checkbox": "- [x] approved",
    "url": "https://evil.example/phish", "long": "A" * 5000, "backslash": "a\\`b\\", "html_comment": "<!-- evalsprint-gate --> hi", "details": "<details open><summary>x</summary>y</details>",
}
# what a notification or a link can still be made of that the critic's list does not spell out
EXTRA_PAYLOADS = {
    "www_host": "www.evil.example/phish", "email": "ops@evil.example", "owner_repo_ref": "octo/repo#7", "gh_ref": "GH-42", "ftp": "ftp://evil.example/x",
    "mention_in_a_sentence": "thanks @octocat, see #12 and https://evil.example.", "autolink": "<https://evil.example>", "entity_mention": "&#64;octocat &#35;12 &#104;ttp://x",
    "amp": "a&amp;b &lt;script&gt;", "ref_link": "[x][1]\n\n[1]: http://evil.example", "table_break": "| a | b |\n|---|---|", "tilde_fence": "~~~\ninjected\n~~~",
    "trailing_backslash": "x\\", "only_spaces": "   ", "emoji": "✅❌ 🔒",
}
PAYLOADS = {**CRITIC_PAYLOADS, **EXTRA_PAYLOADS}
BENIGN = "zz-benign"

# what the critic's scanner calls a notification, and a stricter reading of what GitHub linkifies or notifies
CRITIC_NOTIFY = re.compile(r"(?<![\w])@[\w/-]+|(?<![\w&])#\d+|https?://")
STRICT_NOTIFY = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://|\bwww\.|[\w.+-]+@[\w-]+\.[\w.-]+|(?<![\w/])@[\w-]|#\d+|\bGH-\d+", re.IGNORECASE)
CONTROL = re.compile("[\x00-\x09\x0b-\x1f\x7f  \u0085‪-‮⁦-⁩]")
RAW_TAG = re.compile(r"</?[A-Za-z!][^>]*>")

N = 60


def tags_of(i: int) -> list[str]:
    return ["general"] if i < 30 else ["safety"] if i < 45 else []


class Scenario:
    """A baseline and a candidate that differ in one thing: the ``value`` the surface puts into the candidate artifact."""

    def __init__(self, value: str):
        self.value = value
        self.base_flags = flags(N, fail=[55, 56, 57])
        self.cand_flags = flags(N, fail=[55, 56, 57])
        self.base_kw = {"tags": tags_of}
        self.cand_kw = {"tags": tags_of}
        self.base_summary = {"evalsprint_version": "0.2.0", "repeat": {"n": 1, "per_tag": {}}, "target_retries": 1}
        self.cand_summary = {"evalsprint_version": "0.2.0", "repeat": {"n": 1, "per_tag": {}}, "target_retries": 1}
        self.base_sha = self.cand_sha = JUDGES_SHA
        self.gate = {"critical_tags": ["safety"]}
        self.calibration = None
        self.cand_run_id = "cand"
        self.base_extra: list = []
        self.cand_extra: list = []
        self.edit_base = None
        self.edit_cand = None

    def runs(self) -> tuple[RunResult, RunResult]:
        base, cand = build(self.base_flags, **self.base_kw), build(self.cand_flags, **self.cand_kw)
        for edit, records in ((self.edit_base, base), (self.edit_cand, cand)):
            if edit:
                edit(records)
        base += self.base_extra
        cand += self.cand_extra
        return (
            make_run("base", base, judges_sha=self.base_sha, **self.base_summary),
            make_run(self.cand_run_id, cand, judges_sha=self.cand_sha, **self.cand_summary),
        )

    def render(self) -> tuple[dict, str]:
        base, cand = self.runs()
        c = compare_runs(base, cand, gate=self.gate, calibration=self.calibration)
        return c, to_markdown(c)


def rename(records: list[dict], index: int, new_id: str) -> None:
    records[index]["case_id"] = records[index]["case"]["id"] = new_id


# ---- the surfaces: each one puts the value into one place of the candidate (or baseline, or gate, or calibration) artifact


def s_tag(sc):  # a tag the candidate adds to 30 cases, 8 of which regress: the by-tag table and the tag-drop warning
    sc.cand_kw = {"tags": lambda i: tags_of(i) + ([sc.value] if i < 30 else [])}
    sc.cand_flags = flags(N, fail=[0, 1, 2, 3, 4, 5, 6, 7, 55, 56, 57])


def s_tag_rename(sc):  # the critic's tag_rename: the critical tag "safety" is renamed in the candidate
    sc.cand_kw = {"tags": lambda i: [sc.value] if 30 <= i < 45 else tags_of(i)}


def s_version(sc):
    sc.cand_summary["evalsprint_version"] = sc.value


def s_judges_sha(sc):
    sc.cand_sha = sc.value


def s_repeat_tag(sc):
    sc.cand_summary["repeat"] = {"n": 1, "per_tag": {sc.value: 5}}


def s_run_id(sc):
    sc.cand_run_id = sc.value


def s_case_id_candidate_only(sc):
    sc.cand_extra = build([True, True], prefix="new-")
    rename(sc.cand_extra, 0, sc.value)


def s_case_id_removed(sc):
    sc.base_extra = build([True, True], prefix="gone-")
    rename(sc.base_extra, 0, sc.value)


def s_case_id_failing(sc):
    sc.edit_base = lambda records: rename(records, 10, sc.value)
    sc.edit_cand = lambda records: rename(records, 10, sc.value)
    sc.cand_flags = flags(N, fail=[10, 11, 12, 13, 55, 56, 57])


def s_case_id_passing(sc):
    sc.edit_base = lambda records: rename(records, 55, sc.value)
    sc.edit_cand = lambda records: rename(records, 55, sc.value)
    sc.cand_flags = flags(N, fail=[56, 57])


def s_case_id_edited(sc):
    def edit(records):
        rename(records, 12, sc.value)

    sc.edit_base = edit
    sc.edit_cand = lambda records: (edit(records), records[12].update(case_sha="edited"))[0]


def s_case_id_crash(sc):
    sc.edit_base = lambda records: rename(records, 11, sc.value)
    sc.edit_cand = lambda records: rename(records, 11, sc.value)
    sc.cand_flags = flags(N, fail=[55, 56, 57], error=[11])


def s_case_id_critical(sc):  # a critical case (tag "safety") that flips: the critical-case reason
    sc.edit_base = lambda records: rename(records, 35, sc.value)
    sc.edit_cand = lambda records: rename(records, 35, sc.value)
    sc.cand_flags = flags(N, fail=[35, 55, 56, 57])


def s_gate_critical_tag(sc):  # the gate is trusted (a workflow file), but its values are printed too
    sc.gate = {"critical_tags": ["safety", sc.value]}


def s_calibration_judge(sc):
    sc.gate = {"require_calibration": True}
    sc.calibration = {"judge": sc.value, "kappa_weighted": 0.3, "kappa_weighted_ci": [0.1, 0.5], "kappa_decision": "not_trustworthy"}


def s_calibration_decision(sc):
    sc.gate = {"require_calibration": True}
    sc.calibration = {"judge": "quality", "kappa_weighted": 0.3, "kappa_weighted_ci": [0.1, 0.5], "kappa_decision": sc.value}


def s_calibration_decision_released(sc):  # the decision of a judge that IS released is printed in the calibration line
    sc.calibration = {"judge": "quality", "kappa_weighted": 0.95, "kappa_weighted_ci": [0.9, 1.0], "kappa_decision": sc.value}


# surfaces compare never prints: the artifact value must not reach gate.md at all (byte for byte the benign markdown)
def s_label(sc):
    sc.cand_summary["label"] = sc.value


def s_suite(sc):
    sc.cand_summary["suite"] = sc.value


def s_dataset(sc):
    sc.cand_summary["dataset"] = sc.value


def s_pass_rule(sc):
    sc.cand_summary["pass_rule"] = sc.value


def s_model(sc):
    sc.cand_summary["models"] = {sc.value: N}
    sc.edit_cand = lambda records: records[0]["output"].update(model=sc.value)


def s_error_text(sc):
    sc.cand_flags = flags(N, fail=[55, 56, 57], error=[11])  # the same crash as the benign run, only its text differs
    sc.edit_cand = lambda records: records[11]["output"].update(error=sc.value or "timeout")


def s_reasoning(sc):
    sc.edit_cand = lambda records: [r["judges"][0].update(reasoning=sc.value) for r in records if r["judges"]]


def s_judge_name(sc):
    sc.edit_cand = lambda records: [r["judges"][0].update(judge=sc.value) for r in records if r["judges"]]


def s_target_retries_text(sc):  # forged text where the runner writes numbers
    sc.cand_summary["target_retries"] = sc.value
    sc.cand_summary["n_target_retries"] = sc.value


def s_by_tag_key(sc):
    sc.cand_summary["by_tag"] = {sc.value: {"n": 1}}


def s_placeholder(sc):
    sc.cand_summary["placeholder"] = sc.value


def s_case_sha(sc):
    sc.edit_cand = lambda records: records[0].update(case_sha=sc.value)  # an edited case: only its id is printed


PRINTED = {
    name[2:]: fn
    for name, fn in list(globals().items())
    if name.startswith("s_")
    and name[2:]
    in (
        "tag", "tag_rename", "version", "judges_sha", "repeat_tag", "run_id", "case_id_candidate_only", "case_id_removed", "case_id_failing",
        "case_id_passing", "case_id_edited", "case_id_crash", "case_id_critical", "gate_critical_tag", "calibration_judge", "calibration_decision",
        "calibration_decision_released",
    )
}
NEVER_PRINTED = {
    name[2:]: fn
    for name, fn in list(globals().items())
    if name.startswith("s_") and name[2:] in ("label", "suite", "dataset", "pass_rule", "model", "error_text", "reasoning", "judge_name", "target_retries_text", "by_tag_key", "placeholder", "case_sha")
}


def render(surface_fn, value: str) -> tuple[dict, str]:
    sc = Scenario(value)
    surface_fn(sc)
    return sc.render()


def notifications(md: str) -> list[str]:
    """Everything GitHub could turn into a notification or a link in the text of ``md`` (code spans removed by the CommonMark rules)."""
    text = outside_code(md)
    found = CRITIC_NOTIFY.findall(text) + STRICT_NOTIFY.findall(html.unescape(text))
    naive = re.sub(r"`[^`\n]*`", "", md)  # the critic's own reading of "outside code"
    return found + CRITIC_NOTIFY.findall(naive)


def structure(md: str) -> list[str]:
    """The markdown outside code spans, without the references that stand for a backtick or a pipe of a payload between two spans.

    The rows of a table are sorted: the by-tag table lists tags alphabetically, so a payload may move its own row (the cells of a row
    are what matters, not the row's place).
    """
    lines = outside_code(md).replace("&#96;", "").replace("&#124;", "").splitlines()
    rows = sorted(l for l in lines if l.startswith("|"))
    return [l for l in lines if not l.startswith("|")] + rows


class TheScannerCatchesWhatItIsFor(unittest.TestCase):
    """The grid below is only worth something if the scanner sees the old behaviour."""

    def test_raw_text_is_flagged(self):
        for name in ("mention", "issue_ref", "url", "www_host", "email", "owner_repo_ref", "gh_ref", "ftp", "mention_in_a_sentence", "autolink", "md_link", "md_image"):
            with self.subTest(payload=name):
                self.assertTrue(notifications(f"- tag {PAYLOADS[name]} dropped"), name)
        self.assertEqual(notifications(f"- tag {_code('@octocat #12 https://x.example www.x.example a@b.example')} dropped"), [])

    def test_the_benign_value_is_clean_and_reaches_the_markdown_of_every_printed_surface(self):
        for name, fn in PRINTED.items():
            with self.subTest(surface=name):
                c, md = render(fn, BENIGN)
                self.assertEqual(notifications(md), [])
                spans = [t for line in md.splitlines() for kind, t in inline_tokens(line) if kind == "code"]
                self.assertTrue(any(BENIGN in t for t in spans), f"the surface {name} is not printed: {md[:300]}")


class EveryArtifactStringIsInert(unittest.TestCase):
    """The 04c grid: printed surfaces x payloads."""

    @classmethod
    def setUpClass(cls):
        cls.benign = {name: render(fn, BENIGN) for name, fn in PRINTED.items()}

    def check(self, surface: str, payload_name: str, value: str) -> None:
        fn = PRINTED[surface]
        c, md = render(fn, value)
        b_c, b_md = self.benign[surface]
        where = f"{surface} / {payload_name}"
        self.assertEqual(c["verdict"], b_c["verdict"], where)  # a payload never moves the gate
        self.assertEqual(len(c["reasons"]), len(b_c["reasons"]), where)
        self.assertEqual(len(c["warnings"]), len(b_c["warnings"]), where)
        lines, b_lines = md.splitlines(), b_md.splitlines()
        self.assertEqual(len(lines), len(b_lines), f"{where}: the payload created or removed a line")
        for k, (x, y) in enumerate(zip(lines, b_lines)):
            self.assertEqual(x[:2], y[:2], f"{where}: line {k} changed its block kind")
            self.assertEqual(x.startswith("|"), y.startswith("|"), where)
            if x.startswith("|"):
                self.assertEqual(unescaped_pipes(x), unescaped_pipes(y), f"{where}: a table row changed its cells")
        self.assertEqual(notifications(md), [], where)  # 0 NOTIFY
        self.assertEqual(structure(md), structure(b_md), f"{where}: the text outside code spans differs from the benign run")
        self.assertEqual(CONTROL.findall(md), [], where)
        self.assertEqual([l for l in lines if l.lstrip().startswith("::")], [], where)
        self.assertEqual(set(RAW_TAG.findall(outside_code(md))), {"<sub>", "</sub>"}, where)
        self.assertEqual(outside_code(md).count("<"), outside_code(b_md).count("<"), where)

    def test_the_whole_grid(self):
        cells = 0
        for surface in PRINTED:
            for payload_name, value in PAYLOADS.items():
                with self.subTest(surface=surface, payload=payload_name):
                    self.check(surface, payload_name, value)
                cells += 1
        self.assertEqual(cells, len(PRINTED) * len(PAYLOADS))
        self.assertGreaterEqual(cells, 17 * 33)

    def test_the_payload_is_still_readable_inside_its_code_span(self):
        # inert does not mean dropped: a reviewer must see the tag / version / sha that changed (control characters shown as spaces)
        for surface in ("tag", "tag_rename", "version", "judges_sha", "repeat_tag", "run_id", "gate_critical_tag", "case_id_failing", "calibration_judge"):
            for name in ("mention", "url", "issue_ref", "backtick", "pipe", "md_link", "script"):
                with self.subTest(surface=surface, payload=name):
                    _, md = render(PRINTED[surface], PAYLOADS[name])
                    in_a_table = _code(PAYLOADS[name], table=True)  # a table cell cuts at the pipe too
                    self.assertTrue(_code(PAYLOADS[name]) in md or in_a_table in md, md)
                    self.assertEqual(shown(_code(PAYLOADS[name])), PAYLOADS[name])
                    self.assertEqual(shown(in_a_table), PAYLOADS[name])

    def test_an_empty_string_is_absent_not_hostile(self):
        # "" means "not said" for a version, a sha, a judge or a decision (the benign twin would say something else): no crash, nothing to notify
        for surface, fn in PRINTED.items():
            with self.subTest(surface=surface):
                _, md = render(fn, "")
                self.assertEqual(notifications(md), [])
                self.assertEqual(CONTROL.findall(md), [])

    def test_the_json_keeps_the_raw_text(self):
        c, _ = render(PRINTED["version"], "@octocat #1 `x` https://evil.example")
        self.assertEqual(c["kit_version"]["candidate"], "@octocat #1 `x` https://evil.example")
        self.assertEqual(json.loads(json.dumps(c))["kit_version"], c["kit_version"])
        self.assertTrue(any("@octocat" in w for w in c["warnings"]))  # the plain text of the message (JSON, logs) is unchanged


class SurfacesCompareNeverPrints(unittest.TestCase):
    def test_the_markdown_is_byte_for_byte_the_benign_one(self):
        for surface, fn in NEVER_PRINTED.items():
            _, benign = render(fn, BENIGN)
            for payload_name, value in PAYLOADS.items():
                with self.subTest(surface=surface, payload=payload_name):
                    c, md = render(fn, value)
                    self.assertEqual(md, benign)
                    self.assertEqual(notifications(md), [])


class KappaDecisionAndEveryCalibrationString(unittest.TestCase):
    def test_the_decision_is_a_code_span_in_the_calibration_line(self):
        for decision in ("usable", "trustworthy", "@octocat", "#12", "https://evil.example", "a`b|c", "<script>"):
            with self.subTest(decision=decision):
                _, md = render(s_calibration_decision_released, decision)
                (line,) = [l for l in md.splitlines() if l.startswith("Judge calibration")]
                self.assertIn(f"decision {_code(decision)}", line)
                self.assertEqual(notifications(line), [])

    def test_the_judge_name_is_a_code_span_in_the_line(self):
        _, md = render(s_calibration_judge, "@octocat")
        (line,) = [l for l in md.splitlines() if l.startswith("Judge calibration")]
        self.assertTrue(line.startswith("Judge calibration `@octocat`: "), line)
        self.assertEqual(notifications(md), [])

    def test_a_calibration_with_non_text_fields_is_ignored_safely(self):
        for bad in ({"judge": 12, "kappa_decision": ["x"]}, {"judge": None, "kappa_decision": {"a": 1}}, {"kappa_decision": 3.5}):
            with self.subTest(bad=bad):
                calibration = {"kappa_weighted": 0.9, "kappa_weighted_ci": [0.8, 1.0], **bad}
                sc = Scenario("x")
                sc.calibration = calibration
                c, md = sc.render()
                self.assertEqual(notifications(md), [])


class MessagesKeepTheirPlainText(unittest.TestCase):
    """The markdown twin is an attribute of the message: JSON, logs and the CLI keep reading an ordinary str."""

    def scenario(self):
        sc = Scenario("@octocat")
        sc.cand_sha = "sha-`x`"
        sc.cand_summary["evalsprint_version"] = "0.2.1"
        sc.cand_summary["repeat"] = {"n": 1, "per_tag": {"@octocat": 5}}
        sc.cand_kw = {"tags": lambda i: tags_of(i)}
        return sc.render()

    def test_reasons_and_warnings_are_strings_with_a_markdown_twin(self):
        c, md = self.scenario()
        self.assertTrue(c["reasons"] and c["warnings"])
        for message in c["reasons"] + c["warnings"]:
            self.assertIsInstance(message, str)
            self.assertTrue(hasattr(message, "md"))
            if "@octocat" in message or "sha-" in message:
                self.assertNotEqual(message.md, str(message))  # an artifact value is in it: the twin puts it in a code span
            self.assertNotIn("\n", message.md)
            self.assertEqual(notifications(f"- {message.md}"), [])
        self.assertEqual(json.loads(json.dumps(c))["reasons"], [str(r) for r in c["reasons"]])
        self.assertIn("@octocat", "\n".join(c["reasons"]))

    def test_a_message_without_a_twin_is_made_inert_as_a_whole(self):
        # a plain str (from a caller that edited the dict, an older writer of comparison.json) goes through the same gate
        c, _ = self.scenario()
        c["warnings"] = [*c["warnings"], "see @octocat, #12, https://evil.example and www.evil.example, ops@evil.example <b>x</b> [a](b)"]
        md = to_markdown(c)
        self.assertEqual(notifications(md), [])
        self.assertNotIn("<b>", outside_code(md))

    def test_the_benign_message_is_the_same_text_as_before(self):
        sc = Scenario("x")
        sc.cand_sha = "bbbbbbbbbbbb"
        c, _ = sc.render()
        self.assertEqual(c["reasons"][0].split(":", 1)[0], "judge configuration or rubric changed")


class JudgesShaReasonNamesWhatV02Hashes(unittest.TestCase):
    NOTE = "evalsprint 0.2 also hashes custom-judge code and data_files"

    def reason(self, baseline_version, candidate_version="0.2.0"):
        sc = Scenario("x")
        sc.cand_sha = "bbbbbbbbbbbb"
        sc.base_summary["evalsprint_version"] = baseline_version
        sc.cand_summary["evalsprint_version"] = candidate_version
        if baseline_version is None:
            sc.base_summary.pop("evalsprint_version")
        c, md = sc.render()
        (reason,) = [r for r in c["reasons"] if "judge configuration or rubric changed" in r]
        return reason, md

    def test_a_v0_1_or_unknown_baseline_is_told_to_refresh_in_a_reviewed_pr(self):
        for version in ("0.1.0", "0.1.5", "v0.1", None, "banana"):
            with self.subTest(baseline=version):
                reason, md = self.reason(version)
                self.assertIn(self.NOTE, reason)
                self.assertIn("with custom judges must be refreshed in a reviewed PR", reason)
                self.assertIn(self.NOTE, md)

    def test_a_v0_2_baseline_gets_no_note(self):
        for version in ("0.2.0", "0.2", "v0.2.3", "0.3.0", "1.0.0"):
            with self.subTest(baseline=version):
                reason, md = self.reason(version, "0.2.0")
                self.assertNotIn("custom-judge code", reason)
                self.assertNotIn("custom-judge code", md)

    def test_the_note_is_there_when_the_change_is_only_a_warning_too(self):
        sc = Scenario("x")
        sc.cand_sha = "bbbbbbbbbbbb"
        sc.base_summary["evalsprint_version"] = "0.1.4"
        sc.gate = {"allow_judge_changes": True}
        c, _ = sc.render()
        self.assertEqual(c["reasons"], [])
        self.assertTrue(any(self.NOTE in w and "allowed by gate.allow_judge_changes" in w for w in c["warnings"]))


if __name__ == "__main__":
    unittest.main()
