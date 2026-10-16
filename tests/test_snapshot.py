"""Tests for scripts/project_snapshot.py (the repository -> one markdown file script).

Everything runs on a throwaway tree built in a temp directory; nothing touches the real repository.
The failure this guards against: a snapshot of a repo with a multi-agent scratch tree came out at
23 MB. The default rules must keep scratch out, keep the decision documents, and say what they dropped.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests import ROOT

_SPEC = importlib.util.spec_from_file_location("project_snapshot", ROOT / "scripts" / "project_snapshot.py")
ps = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["project_snapshot"] = ps  # dataclasses look the module up by name while it executes
_SPEC.loader.exec_module(ps)  # type: ignore[union-attr]

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


def write(root: Path, rel: str, text: str | bytes = "x = 1\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8", newline="\n")  # LF on every platform: the default would write CRLF on Windows
    return path


def make_repo(root: Path) -> None:
    """A small repository that has one of everything the rules care about."""
    write(root, "README.md", "# demo\n")
    write(root, "pyproject.toml", "[project]\nname = 'demo'\n")
    write(root, "evalsprint/core.py", "def f():\n    return 1\n")
    write(root, "tests/test_core.py", "def test_f():\n    pass\n")
    write(root, "docs/GUIDE.md", "guide\n")
    write(root, "examples/bot/cases.jsonl", '{"id": "a"}\n')
    # orchestration: decision documents (kept, at any depth) and scratch (left out)
    write(root, "orchestration/run1/plan.md", "# plan\n")
    write(root, "orchestration/run1/_contrato.md", "contract\n")
    write(root, "orchestration/run1/cycle1/critic/critique.md", "# critique\n")
    write(root, "orchestration/run1/cycle1/critic/report.md", "# report\n")
    write(root, "orchestration/run1/cycle2/builder/builders.md", "builders\n")
    write(root, "orchestration/run1/cycle2/builder/contracts.md", "contracts\n")
    write(root, "orchestration/run1/cycle2/builder/scratch.py", "print('scratch')\n")
    write(root, "orchestration/run1/cycle2/builder/results.json", '{"n": 1}\n')
    write(root, "orchestration/run1/cycle2/builder/notes.md", "not a decision document\n")
    write(root, "orchestration/run1/cycle1/critic/work/big.jsonl", "x\n" * 2000)
    write(root, "orchestration/run1/cycle1/critic/work/plan.md", "a work dir wins over the keep list\n")
    # scratch that lives outside orchestration
    write(root, "examples/bot/work/tmp.py", "tmp = 1\n")
    write(root, "work/notes.md", "top-level work dir\n")
    # folders that were always skipped
    write(root, "demo_out/run/results.jsonl", '{"r": 1}\n')
    write(root, "runs/2026/summary.json", "{}\n")
    write(root, ".venv/lib/x.py", "venv = 1\n")
    write(root, "evalsprint.egg-info/PKG-INFO", "meta\n")
    write(root, "__pycache__/x.pyc", b"\x00\x01")
    # images and logs
    write(root, "docs/shot.png", b"\x89PNG\r\n\x1a\n\x00\x00")
    write(root, "docs/diagram.svg", "<svg/>\n")
    write(root, "docs/small.log", "one line\n")
    write(root, "docs/huge.log", "log line\n" * 10_000)  # ~90 KB
    write(root, "orchestration/run1/plan.log", "ignored anyway\n")


def snap(root: Path, **kw) -> "ps.Snapshot":
    return ps.snapshot(root, None, ps.Options(**kw), now=NOW)


def included(s: "ps.Snapshot") -> set[str]:
    return {rel.as_posix() for rel, _ in s.included}


class SnapshotTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "repo with spaces"
        self.root.mkdir()
        make_repo(self.root)


class TestDefaultRules(SnapshotTestCase):
    def test_kept_files(self):
        got = included(snap(self.root))
        for rel in (
            "README.md",
            "pyproject.toml",
            "evalsprint/core.py",
            "tests/test_core.py",
            "docs/GUIDE.md",
            "docs/small.log",
            "examples/bot/cases.jsonl",
        ):
            self.assertIn(rel, got)

    def test_orchestration_keeps_only_decision_documents_at_any_depth(self):
        got = {p for p in included(snap(self.root)) if p.startswith("orchestration/")}
        self.assertEqual(
            got,
            {
                "orchestration/run1/plan.md",
                "orchestration/run1/_contrato.md",
                "orchestration/run1/cycle1/critic/critique.md",
                "orchestration/run1/cycle1/critic/report.md",
                "orchestration/run1/cycle2/builder/builders.md",
                "orchestration/run1/cycle2/builder/contracts.md",
            },
        )

    def test_work_directories_are_left_out_even_when_the_name_is_on_the_keep_list(self):
        got = included(snap(self.root))
        self.assertNotIn("orchestration/run1/cycle1/critic/work/plan.md", got)
        self.assertFalse([p for p in got if "/work/" in p or p.startswith("work/")], got)

    def test_work_rule_also_holds_with_include_orchestration(self):
        got = included(snap(self.root, include_orchestration=True))
        self.assertFalse([p for p in got if "/work/" in p or p.startswith("work/")], got)
        self.assertIn("orchestration/run1/cycle2/builder/scratch.py", got)
        self.assertIn("orchestration/run1/cycle2/builder/notes.md", got)

    def test_demo_out_runs_venv_egg_info_and_caches_are_left_out(self):
        got = included(snap(self.root))
        for bad in ("demo_out/", "runs/", ".venv/", "egg-info", "__pycache__"):
            self.assertFalse([p for p in got if bad in p], (bad, got))

    def test_images_are_left_out(self):
        s = snap(self.root)
        got = included(s)
        self.assertNotIn("docs/shot.png", got)
        self.assertNotIn("docs/diagram.svg", got)
        self.assertEqual(s.excluded[ps.REASON_IMAGE], 2)

    def test_logs_above_the_limit_are_left_out_and_small_ones_kept(self):
        s = snap(self.root)
        got = included(s)
        self.assertNotIn("docs/huge.log", got)
        self.assertIn("docs/small.log", got)
        self.assertEqual(s.excluded[ps.REASON_LOG], 1)

    def test_log_limit_is_configurable(self):
        got = included(snap(self.root, max_log_kb=500))
        self.assertIn("docs/huge.log", got)
        got = included(snap(self.root, max_log_kb=0))
        self.assertNotIn("docs/small.log", got)

    def test_files_over_max_kb_are_skipped_and_listed(self):
        write(self.root, "data/big.txt", "y" * 3000)
        s = snap(self.root, max_kb=1)
        self.assertNotIn("data/big.txt", included(s))
        self.assertIn("data/big.txt", {rel.as_posix() for rel, _ in s.skipped})
        self.assertIn("`data/big.txt`", s.markdown)

    def test_binary_and_non_utf8_files_are_skipped_not_crashed_on(self):
        write(self.root, "data/blob.bin", b"\x00\x01\x02" * 100)
        write(self.root, "data/latin1.txt", "caf\xe9\n".encode("latin-1"))
        s = snap(self.root)
        skipped = {rel.as_posix() for rel, _ in s.skipped}
        self.assertIn("data/blob.bin", skipped)
        self.assertIn("data/latin1.txt", skipped)

    def test_symlinks_are_not_followed(self):
        target = self.root.parent / "outside.txt"
        target.write_text("secret\n", encoding="utf-8")
        try:
            os.symlink(target, self.root / "link.txt")
            os.symlink(self.root.parent, self.root / "linkdir")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available")
        s = snap(self.root)
        self.assertNotIn("link.txt", included(s))
        self.assertNotIn("secret", s.markdown)
        self.assertFalse([p for p in included(s) if p.startswith("linkdir")])


class TestRuleExclusion(unittest.TestCase):
    """The rule function on its own: no file system needed."""

    def test_reasons(self):
        o = ps.Options()
        r = ps.rule_exclusion
        self.assertEqual(r(Path("orchestration/x/results.json"), 10, o), ps.REASON_ORCHESTRATION)
        self.assertIsNone(r(Path("orchestration/x/plan.md"), 10, o))
        self.assertEqual(r(Path("a/work/b.py"), 10, o), ps.REASON_WORK)
        self.assertEqual(r(Path("a.png"), 10, o), ps.REASON_IMAGE)
        self.assertEqual(r(Path("a.PNG"), 10, o), ps.REASON_IMAGE)
        self.assertEqual(r(Path("a.log"), 60 * 1024, o), ps.REASON_LOG)
        self.assertIsNone(r(Path("a.log"), 10 * 1024, o))
        self.assertIsNone(r(Path("tests/test_a.py"), 10, o))
        self.assertEqual(r(Path("tests/test_a.py"), 10, ps.Options(no_tests=True)), ps.REASON_TESTS)

    def test_a_file_called_work_is_not_a_work_directory(self):
        self.assertIsNone(ps.rule_exclusion(Path("docs/work"), 10, ps.Options()))
        self.assertIsNone(ps.rule_exclusion(Path("docs/network.md"), 10, ps.Options()))

    def test_no_tests_only_touches_the_top_level_tests_directory(self):
        o = ps.Options(no_tests=True)
        self.assertIsNone(ps.rule_exclusion(Path("evalsprint/tests_helper.py"), 10, o))
        self.assertIsNone(ps.rule_exclusion(Path("examples/tests/x.py"), 10, o))


class TestNoTestsAndExclude(SnapshotTestCase):
    def test_no_tests_drops_tests_and_counts_them(self):
        s = snap(self.root, no_tests=True)
        got = included(s)
        self.assertFalse([p for p in got if p.startswith("tests/")])
        self.assertIn("evalsprint/core.py", got)
        self.assertEqual(s.excluded[ps.REASON_TESTS], 1)

    def test_no_tests_makes_the_snapshot_smaller(self):
        write(self.root, "tests/test_big.py", "# filler\n" * 5000)
        self.assertLess(snap(self.root, no_tests=True).size_bytes, snap(self.root).size_bytes - 40_000)

    def test_user_exclude_globs(self):
        s = snap(self.root, exclude=["examples/*", "*.toml"])
        got = included(s)
        self.assertNotIn("examples/bot/cases.jsonl", got)
        self.assertNotIn("pyproject.toml", got)
        self.assertIn("README.md", got)
        # cases.jsonl, work/tmp.py (a user glob is checked before the default rules) and pyproject.toml
        self.assertEqual(s.excluded[ps.REASON_USER], 3)

    def test_the_smallest_useful_snapshot_combination(self):
        s = snap(self.root, no_tests=True, exclude=["orchestration/*"])
        got = included(s)
        self.assertFalse([p for p in got if p.startswith(("tests/", "orchestration/"))])


class TestDocument(SnapshotTestCase):
    def test_header_carries_the_size_of_the_finished_document(self):
        s = snap(self.root)
        m = re.search(r"^- Size: (\d+) KB$", s.markdown, re.M)
        self.assertIsNotNone(m, s.markdown[:400])
        self.assertLessEqual(abs(int(m.group(1)) - s.size_bytes / 1024), 0.6)

    def test_header_size_is_consistent_for_a_bigger_document(self):
        write(self.root, "docs/bulk.md", "word " * 60_000)  # ~293 KB: over the default 200 KB limit, so raise it
        s = snap(self.root, max_kb=400)
        m = re.search(r"^- Size: (\d+) KB$", s.markdown, re.M)
        self.assertLessEqual(abs(int(m.group(1)) - s.size_bytes / 1024), 0.6)
        self.assertGreater(s.size_bytes, 290 * 1024)

    def test_header_counts_and_generated_stamp(self):
        s = snap(self.root)
        self.assertIn("- Generated: 2026-09-29 12:00 UTC", s.markdown)
        self.assertIn(f"- Files included: {len(s.included)} ", s.markdown)
        self.assertIn(f"left out by rule: {sum(s.excluded.values())}", s.markdown)

    def test_left_out_section_lists_every_reason_with_counts(self):
        s = snap(self.root)
        self.assertIn("## Left out by rule", s.markdown)
        for reason, count in s.excluded.items():
            self.assertIn(f"- {count} file(s): {reason}", s.markdown)
        self.assertGreater(s.excluded[ps.REASON_ORCHESTRATION], 0)
        self.assertGreater(s.excluded[ps.REASON_WORK], 0)

    def test_left_out_files_are_counted_not_listed_and_their_content_is_absent(self):
        s = snap(self.root)
        self.assertNotIn("scratch.py", s.markdown)
        self.assertNotIn("print('scratch')", s.markdown)
        self.assertNotIn("not a decision document", s.markdown)

    def test_tree_and_file_sections(self):
        s = snap(self.root)
        tree = s.markdown.split("## File tree", 1)[1].split("## Files", 1)[0]
        self.assertIn("evalsprint/", tree)
        self.assertIn("core.py", tree)
        self.assertNotIn("scratch.py", tree)
        self.assertIn("### `evalsprint/core.py`", s.markdown)
        self.assertIn("```python\ndef f():\n    return 1\n```", s.markdown)

    def test_output_is_deterministic(self):
        self.assertEqual(snap(self.root).markdown, snap(self.root).markdown)

    def test_crlf_files_are_read_as_lf(self):
        write(self.root, "docs/CRLF.md", b"one\r\ntwo\r\n")  # a Windows checkout with core.autocrlf=true
        s = snap(self.root)
        self.assertIn("```markdown\none\ntwo\n```", s.markdown)
        self.assertNotIn("\r", s.markdown)

    def test_nested_fences_survive(self):
        write(self.root, "docs/NESTED.md", "text\n```python\nx = 1\n```\nmore\n")
        s = snap(self.root)
        self.assertIn("````markdown\ntext\n```python\nx = 1\n```\nmore\n````", s.markdown)

    def test_output_file_is_never_part_of_the_snapshot(self):
        out = self.root / "PROJECT_SNAPSHOT.md"
        out.write_text("stale snapshot\n" * 100, encoding="utf-8")
        s = ps.snapshot(self.root, out.resolve(), ps.Options(), now=NOW)
        self.assertNotIn("PROJECT_SNAPSHOT.md", included(s))
        self.assertNotIn("stale snapshot", s.markdown)

    def test_git_line_when_git_is_missing(self):
        s = snap(self.root)  # the temp dir is not a git repository
        self.assertRegex(s.markdown, r"- Git commit: (not available|`[0-9a-f]+`)")

    def test_build_snapshot_keeps_the_original_signature(self):
        md, n_in, n_skipped = ps.build_snapshot(self.root, None, 200, NOW, ["examples/*"])
        self.assertIsInstance(md, str)
        self.assertGreater(n_in, 0)
        self.assertEqual(n_skipped, 0)
        self.assertNotIn("examples/bot/cases.jsonl", md.split("## Files", 1)[1])


class TestCli(SnapshotTestCase):
    def run_cli(self, *args: str) -> tuple[int, str, str, Path]:
        out = self.root.parent / "out" / "snap.md"
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = ps.main(["--root", str(self.root), "--out", str(out), *args])
        return code, stdout.getvalue(), stderr.getvalue(), out

    def test_prints_size_and_reasons_and_writes_the_file(self):
        code, stdout, _, out = self.run_cli()
        self.assertEqual(code, 0)
        text = out.read_text(encoding="utf-8")
        m = re.search(r"\((\d+) KB\)", stdout)
        self.assertIsNotNone(m, stdout)
        self.assertLessEqual(abs(int(m.group(1)) - len(text.encode("utf-8")) / 1024), 0.6)
        self.assertIn("left out by rule", stdout)
        self.assertIn("orchestration/ scratch", stdout)
        self.assertIn(f"- Size: {m.group(1)} KB", text)

    def test_the_file_has_lf_endings_and_the_size_of_the_header(self):
        write(self.root, "docs/CRLF.md", b"one\r\ntwo\r\n")
        code, _, _, out = self.run_cli()
        self.assertEqual(code, 0)
        raw = out.read_bytes()
        self.assertNotIn(b"\r", raw)  # neither the CRLF of a source file nor the "\n" -> "\r\n" of a text-mode write on Windows
        size = re.search(rb"^- Size: (\d+) KB$", raw, re.M)
        self.assertIsNotNone(size)
        self.assertLessEqual(abs(int(size.group(1)) - len(raw) / 1024), 0.6)

    def test_no_tests_flag(self):
        _, stdout_all, _, out_all = self.run_cli()
        size_all = out_all.stat().st_size
        code, stdout, _, out = self.run_cli("--no-tests")
        self.assertEqual(code, 0)
        self.assertNotIn("### `tests/test_core.py`", out.read_text(encoding="utf-8"))
        self.assertIn("tests/ (--no-tests)", stdout)
        self.assertLess(out.stat().st_size, size_all)

    def test_include_orchestration_flag(self):
        _, _, _, out = self.run_cli("--include-orchestration")
        self.assertIn("### `orchestration/run1/cycle2/builder/scratch.py`", out.read_text(encoding="utf-8"))

    def test_exclude_flag_is_repeatable(self):
        _, _, _, out = self.run_cli("--exclude", "examples/*", "--exclude", "docs/*")
        text = out.read_text(encoding="utf-8")
        self.assertNotIn("### `examples/bot/cases.jsonl`", text)
        self.assertNotIn("### `docs/GUIDE.md`", text)

    def test_help_says_which_flag_the_claude_project_copy_uses(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as raised:
            ps.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        text = " ".join(out.getvalue().split())
        self.assertRegex(text, r"--no-tests .*copy that goes into the Claude Project")
        self.assertIn("the tests stay in the git repository", text)

    def test_bad_root_is_an_error(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = ps.main(["--root", str(self.root / "missing"), "--out", str(self.root.parent / "x.md")])
        self.assertEqual(code, 2)
        self.assertIn("not a directory", stderr.getvalue())


class TestTheRealRepository(unittest.TestCase):
    """The shipped repository itself: the rules must keep the snapshot readable, not 23 MB."""

    def test_default_snapshot_of_this_repo_is_bounded_and_clean(self):
        s = ps.snapshot(ROOT, None, ps.Options(), now=NOW)
        got = included(s)
        self.assertFalse([p for p in got if p.startswith("demo_out/")])
        self.assertFalse([p for p in got if "/work/" in p])
        allowed = ps.ORCHESTRATION_KEEP
        stray = [p for p in got if p.startswith("orchestration/") and Path(p).name not in allowed]
        self.assertEqual(stray, [])
        # A generous ceiling on purpose: the point is "megabytes of scratch never come back", not a byte budget.
        self.assertLess(s.size_bytes, 3.5 * 1024 * 1024)
        self.assertIn("scripts/project_snapshot.py", got)

    def test_no_tests_is_smaller_on_this_repo(self):
        full = ps.snapshot(ROOT, None, ps.Options(), now=NOW).size_bytes
        lean = ps.snapshot(ROOT, None, ps.Options(no_tests=True), now=NOW).size_bytes
        self.assertLess(lean, full)

    def test_readme_and_script_docs_say_the_claude_project_copy_uses_no_tests(self):
        readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
        self.assertIn("The copy in the Claude Project is made with `--no-tests`", readme)
        self.assertIn("--no-tests", ps.__doc__)
        self.assertIn("Claude Project", ps.__doc__)

    def test_the_sizes_the_readme_and_the_script_state_are_the_measured_ones_within_30_percent(self):
        """\"About 2.6 MB by default, 1.4 MB with --no-tests, 1.3 MB with --exclude orchestration\": a doc number a test recomputes."""
        measured = {
            "default": ps.snapshot(ROOT, None, ps.Options(), now=NOW).size_bytes / 1e6,
            "no_tests": ps.snapshot(ROOT, None, ps.Options(no_tests=True), now=NOW).size_bytes / 1e6,
            "smallest": ps.snapshot(ROOT, None, ps.Options(no_tests=True, exclude=["orchestration/*"]), now=NOW).size_bytes / 1e6,
        }
        if (ROOT / "orchestration").exists():
            self.assertLess(measured["smallest"], measured["no_tests"])
        else:  # the public repository has no orchestration/: excluding it changes nothing
            self.assertEqual(measured["smallest"], measured["no_tests"])
        self.assertLess(measured["no_tests"], measured["default"])
        readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
        doc = " ".join((ps.__doc__ or "").split())
        patterns = {
            "default": r"about (\d+(?:\.\d+)?) MB by default",
            "no_tests": r"about (\d+(?:\.\d+)?) MB with `{1,2}--no-tests`{1,2}[;,]",
            "smallest": r"about (\d+(?:\.\d+)?) MB with `{1,2}--no-tests --exclude 'orchestration/\*'`{1,2}",
        }
        readme_patterns = dict(patterns, smallest=r"\(about (\d+(?:\.\d+)?) MB\) is the smallest useful snapshot")
        for where, text, pats in (("README.md", readme, readme_patterns), ("scripts/project_snapshot.py", doc, patterns)):
            for key, pattern in pats.items():
                with self.subTest(where=where, size=key):
                    found = re.search(pattern, text)
                    self.assertIsNotNone(found, f"{where} no longer states the {key} size ({pattern})")
                    stated = float(found.group(1))
                    self.assertLessEqual(abs(stated - measured[key]) / measured[key], 0.30, f"{where} says {stated} MB, measured {measured[key]:.2f} MB")

    def test_gitignore_lists_the_scratch_paths(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        lines = {line.strip() for line in text.splitlines()}
        for entry in ("demo_out/", "orchestration/**/work/", "results.partial.jsonl", ".venv/"):
            self.assertIn(entry, lines)


if __name__ == "__main__":
    unittest.main()
