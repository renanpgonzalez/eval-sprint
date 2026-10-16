"""`evalsprint init [DIR] [--ci] [--force]` (contract C13): a client repository gets a working `evals/` in one command.

The scaffold comes ONLY from the packaged templates (`evalsprint/templates/`, read through importlib.resources). These tests run
against the real package data and, where they need to control the templates, against a temp copy patched in with
`evalsprint.cli._templates_root`.
"""

import contextlib
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evalsprint import __version__, cli
from evalsprint.cli import main
from evalsprint.dataset import load_cases
from tests import TEMPLATES

CORE = ["evals/cases.jsonl", "evals/eval.yaml", "evals/requirements.txt", "evals/rubric.yaml", "evals/target_adapter.py"]
CI = [".github/CODEOWNERS.evals.example", ".github/workflows/evals-baseline.yml", ".github/workflows/evals.yml"]
IGNORE = [".gitignore"]  # C18: `init` also creates or extends it (runs/ and .label_secret)


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def files_under(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


class InitCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.repo = self.tmp / "client-repo"
        self.repo.mkdir()

    def temp_templates(self, layout: str = "repo") -> Path:
        """A copy of the shipped templates in a temp dir, so a test can edit or remove a file."""
        root = self.tmp / "templates"
        shutil.copytree(TEMPLATES, root, ignore=shutil.ignore_patterns("__pycache__"))
        return root

    def patched(self, root: Path):
        return mock.patch("evalsprint.cli._templates_root", lambda: root)


class ScaffoldTests(InitCase):
    def test_an_empty_directory_gets_the_five_files_and_a_gitignore(self):
        code, out, err = run_cli("init", self.repo)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(files_under(self.repo), sorted(CORE + IGNORE))
        for name in CORE + IGNORE:
            self.assertIn(f"wrote  {name}", out)

    def test_the_scaffold_passes_describe_out_of_the_box(self):
        run_cli("init", self.repo)
        code, out, err = run_cli("describe", "-c", self.repo / "evals" / "eval.yaml")
        self.assertEqual((code, err), (0, ""))  # offline: describe swaps the LLM judge's provider for the mock one
        self.assertIn("Config and test set are valid.", out)
        self.assertIn("target_adapter.py:offline_stub", out)  # the adapter runs in its documented offline mode

    def test_cases_jsonl_is_the_example_set_and_loads(self):
        run_cli("init", self.repo)
        cases = load_cases(self.repo / "evals" / "cases.jsonl")
        self.assertGreaterEqual(len(cases), 8)
        self.assertEqual((self.repo / "evals" / "cases.jsonl").read_text(encoding="utf-8"), (TEMPLATES / "cases.example.jsonl").read_text(encoding="utf-8"))

    def test_files_are_the_templates_with_lf_line_endings(self):
        root = self.temp_templates()
        for path in root.rglob("*.yaml"):  # a Windows checkout of the kit: CRLF templates
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        with self.patched(root):
            self.assertEqual(run_cli("init", self.repo)[0], 0)
        raw = (self.repo / "evals" / "eval.yaml").read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertEqual(raw.decode("utf-8"), (TEMPLATES / "eval.yaml").read_text(encoding="utf-8"))

    def test_a_missing_directory_is_created_and_the_default_is_the_current_directory(self):
        code, _, err = run_cli("init", self.tmp / "new" / "deeper")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(files_under(self.tmp / "new" / "deeper"), sorted(CORE + IGNORE))
        previous = os.getcwd()
        os.chdir(self.repo)
        self.addCleanup(os.chdir, previous)
        code, _, err = run_cli("init")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(files_under(self.repo), sorted(CORE + IGNORE))

    def test_a_file_where_the_directory_should_be_is_a_usage_error(self):
        blocker = self.tmp / "afile"
        blocker.write_text("x", encoding="utf-8")
        code, _, err = run_cli("init", blocker)
        self.assertEqual(code, 2)
        self.assertIn("is not a directory", err)

    def test_no_temp_files_are_left_behind(self):
        run_cli("init", self.repo, "--ci")
        self.assertEqual([p for p in self.repo.rglob("*") if ".tmp" in p.name], [])

    def test_the_kit_is_pinned_to_the_version_that_wrote_the_scaffold(self):
        root = self.temp_templates()
        req = root / "evals" / "requirements.txt"
        req.write_text(
            "evalsprint @ git+https://github.com/<you>/eval-sprint@v0.1.0\n"
            "#      evalsprint @ git+ssh://git@github.com/<you>/eval-sprint.git@v0.1.0\n"
            "#      evalsprint==0.1.0\n"
            "openai==1.99.9\n",
            encoding="utf-8",
        )
        with self.patched(root):
            run_cli("init", self.repo)
        text = (self.repo / "evals" / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn(f"eval-sprint@v{__version__}\n", text)
        self.assertIn(f"eval-sprint.git@v{__version__}\n", text)
        self.assertIn(f"evalsprint=={__version__}\n", text)
        self.assertIn("openai==1.99.9", text)  # nothing else is touched
        self.assertNotIn("0.1.0", text)

    def test_next_steps_are_printed(self):
        code, out, _ = run_cli("init", self.repo)
        self.assertIn("Next steps:", out)
        for expected in ("pip install -r evals/requirements.txt", "evalsprint describe -c evals/eval.yaml", "evalsprint run -c evals/eval.yaml", "API key", "evals/target_adapter.py", ".gitignore", "evalsprint baseline"):
            self.assertIn(expected, out)
        self.assertIn("evalsprint init --ci", out)  # without --ci: says how to add the gate

    def test_next_steps_for_a_directory_tell_you_to_cd_into_it(self):
        _, out, _ = run_cli("init", self.repo)
        self.assertIn(f"cd {self.repo} && pip install", out)


class CiScaffoldTests(InitCase):
    def test_ci_adds_the_workflows_and_the_codeowners_example(self):
        code, out, err = run_cli("init", self.repo, "--ci")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(files_under(self.repo), sorted(CORE + CI + IGNORE))
        self.assertEqual((self.repo / ".github" / "workflows" / "evals.yml").read_text(encoding="utf-8"), (TEMPLATES / "github" / "evals.yml").read_text(encoding="utf-8"))
        self.assertEqual((self.repo / ".github" / "workflows" / "evals-baseline.yml").read_text(encoding="utf-8"), (TEMPLATES / "github" / "evals-baseline.yml").read_text(encoding="utf-8"))
        self.assertEqual((self.repo / ".github" / "CODEOWNERS.evals.example").read_text(encoding="utf-8"), (TEMPLATES / "github" / "CODEOWNERS.example").read_text(encoding="utf-8"))
        self.assertIn("EVAL_PATHS", out)
        self.assertIn("CODEOWNERS", out)
        self.assertNotIn("evalsprint init --ci", out)

    def test_ci_never_writes_a_live_codeowners_file(self):
        run_cli("init", self.repo, "--ci")
        self.assertFalse((self.repo / ".github" / "CODEOWNERS").exists())  # the example has placeholder teams: activating it is a human decision

    def test_without_ci_no_github_folder_is_created(self):
        run_cli("init", self.repo)
        self.assertFalse((self.repo / ".github").exists())

    def test_ci_can_be_added_later_without_touching_the_first_scaffold(self):
        run_cli("init", self.repo)
        edited = self.repo / "evals" / "eval.yaml"
        edited.write_text(edited.read_text(encoding="utf-8").replace("acme-support-bot", "my-real-suite"), encoding="utf-8")
        before = edited.read_bytes()
        code, out, _ = run_cli("init", self.repo, "--ci")
        self.assertEqual(code, 0)
        self.assertEqual(edited.read_bytes(), before)  # the client's edit is safe
        self.assertEqual(files_under(self.repo), sorted(CORE + CI + IGNORE))
        for name in CI:
            self.assertIn(f"wrote  {name}", out)
        for name in CORE:
            self.assertIn(f"kept   {name}", out)


class NeverOverwriteTests(InitCase):
    def test_existing_files_are_kept_and_reported(self):
        (self.repo / "evals").mkdir()
        mine = self.repo / "evals" / "eval.yaml"
        mine.write_text("suite: mine\n", encoding="utf-8")
        code, out, err = run_cli("init", self.repo)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(mine.read_text(encoding="utf-8"), "suite: mine\n")
        self.assertIn("kept   evals/eval.yaml", out)
        self.assertIn("--force", out)
        for name in CORE:
            self.assertTrue((self.repo / name).is_file(), name)  # the missing ones are still written

    def test_running_it_twice_is_a_harmless_no_op(self):
        run_cli("init", self.repo, "--ci")
        snapshot = {name: (self.repo / name).read_bytes() for name in CORE + CI}
        code, out, _ = run_cli("init", self.repo, "--ci")
        self.assertEqual(code, 0)
        self.assertIn("Nothing to do", out)
        self.assertNotIn("wrote", out)
        self.assertEqual({name: (self.repo / name).read_bytes() for name in snapshot}, snapshot)

    def test_force_overwrites_every_file(self):
        run_cli("init", self.repo, "--ci")
        for name in CORE + CI:
            (self.repo / name).write_text("hand edited\n", encoding="utf-8")
        code, out, _ = run_cli("init", self.repo, "--ci", "--force")
        self.assertEqual(code, 0)
        for name in CORE + CI:
            self.assertIn(f"wrote  {name}", out)
            self.assertNotEqual((self.repo / name).read_text(encoding="utf-8"), "hand edited\n", name)
        self.assertEqual(run_cli("describe", "-c", self.repo / "evals" / "eval.yaml")[0], 0)

    def test_other_files_in_evals_are_never_touched(self):
        (self.repo / "evals" / "baseline").mkdir(parents=True)
        (self.repo / "evals" / "baseline" / "summary.json").write_text("{}", encoding="utf-8")
        (self.repo / "evals" / "notes.md").write_text("mine", encoding="utf-8")
        run_cli("init", self.repo, "--force")
        self.assertEqual((self.repo / "evals" / "baseline" / "summary.json").read_text(encoding="utf-8"), "{}")
        self.assertEqual((self.repo / "evals" / "notes.md").read_text(encoding="utf-8"), "mine")

    def test_a_directory_in_the_way_is_an_error_not_a_crash(self):
        (self.repo / "evals" / "eval.yaml").mkdir(parents=True)
        code, _, err = run_cli("init", self.repo, "--force")
        self.assertEqual(code, 2)
        self.assertIn("is a directory", err)


class TemplateSourceTests(InitCase):
    def test_a_missing_template_is_a_clear_error_and_writes_nothing(self):
        root = self.temp_templates()
        (root / "target_adapter.py").unlink()
        with self.patched(root):
            code, _, err = run_cli("init", self.repo)
        self.assertEqual(code, 2)
        self.assertIn("evals/target_adapter.py", err)
        self.assertIn("missing from this evalsprint install", err)
        self.assertEqual(files_under(self.repo), [])  # all-or-nothing: no half scaffold

    def test_no_templates_at_all(self):
        with mock.patch("evalsprint.cli._templates_root", lambda: None):
            code, _, err = run_cli("init", self.repo)
        self.assertEqual(code, 2)
        self.assertIn("templates are missing", err)

    def test_templates_are_read_through_importlib_resources_from_the_package_first(self):
        packaged = self.tmp / "packaged"
        shutil.copytree(TEMPLATES, packaged, ignore=shutil.ignore_patterns("__pycache__"))
        (packaged / "rubric.yaml").write_text((TEMPLATES / "rubric.yaml").read_text(encoding="utf-8") + "# from the package\n", encoding="utf-8")

        class Anchor:  # what importlib.resources.files("evalsprint") returns for a package directory
            def joinpath(self, *parts):
                return packaged

        asked = []
        with mock.patch("importlib.resources.files", lambda package: asked.append(package) or Anchor()):
            self.assertEqual(cli._templates_root(), packaged)
            self.assertEqual(run_cli("init", self.repo)[0], 0)
        self.assertEqual(set(asked), {"evalsprint"})
        self.assertTrue((self.repo / "evals" / "rubric.yaml").read_text(encoding="utf-8").endswith("# from the package\n"))

    def test_a_package_that_ships_without_templates_is_a_one_line_error_and_writes_nothing(self):
        missing = self.tmp / "no-such-dir"

        class Anchor:  # importlib.resources.files("evalsprint") for an install whose package data lacks templates/
            def joinpath(self, *parts):
                return missing

        with mock.patch("importlib.resources.files", lambda package: Anchor()):
            self.assertIsNone(cli._templates_root())  # no fallback to a source checkout: the package is the only source
            code, out, err = run_cli("init", self.repo, "--ci")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertIn("templates are missing from this evalsprint install", err)
        self.assertIn("reinstall the kit", err)
        self.assertEqual(files_under(self.repo), [])

    def test_a_package_that_cannot_be_resolved_is_the_same_error(self):
        with mock.patch("importlib.resources.files", side_effect=ModuleNotFoundError("evalsprint")):
            code, _, err = run_cli("init", self.repo)
        self.assertEqual((code, len(err.strip().splitlines())), (2, 1))
        self.assertIn("templates are missing", err)

    def test_the_real_install_can_scaffold(self):
        root = cli._templates_root()
        self.assertIsNotNone(root, "evalsprint/templates/ is not packaged")
        for dest, source in cli._SCAFFOLD + cli._SCAFFOLD_CI:
            self.assertIsNotNone(cli._template_text(root, source), f"{dest} <- evalsprint/templates/{source}")


class GitignoreTests(InitCase):
    """C18: `init` creates or extends `.gitignore` with `runs/` and `.label_secret`, never duplicating a line."""

    def lines(self):
        return (self.repo / ".gitignore").read_text(encoding="utf-8").splitlines()

    def test_a_new_repository_gets_both_entries(self):
        code, out, _ = run_cli("init", self.repo)
        self.assertEqual(code, 0)
        self.assertIn("runs/", self.lines())
        self.assertIn(".label_secret", self.lines())
        self.assertIn("wrote  .gitignore  (runs/, .label_secret)", out)

    def test_an_existing_file_is_only_appended_to(self):
        original = b"node_modules/\n*.pyc\n"
        (self.repo / ".gitignore").write_bytes(original)
        code, out, _ = run_cli("init", self.repo)
        self.assertEqual(code, 0)
        raw = (self.repo / ".gitignore").read_bytes()
        self.assertTrue(raw.startswith(original))  # the client's lines are untouched
        self.assertEqual(self.lines().count("runs/"), 1)
        self.assertEqual(self.lines().count(".label_secret"), 1)
        self.assertIn("added  runs/, .label_secret to .gitignore", out)

    def test_a_file_without_a_final_newline_and_with_crlf_keeps_its_style(self):
        (self.repo / ".gitignore").write_bytes(b"dist/\r\nbuild/")
        run_cli("init", self.repo)
        raw = (self.repo / ".gitignore").read_bytes()
        self.assertTrue(raw.startswith(b"dist/\r\nbuild/\r\n#"))
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))  # every line ending is CRLF, none bare
        self.assertIn(b"runs/\r\n", raw)

    def test_running_init_twice_changes_nothing(self):
        run_cli("init", self.repo)
        first = (self.repo / ".gitignore").read_bytes()
        code, out, _ = run_cli("init", self.repo)
        self.assertEqual(code, 0)
        self.assertEqual((self.repo / ".gitignore").read_bytes(), first)
        self.assertIn("kept   .gitignore", out)
        self.assertIn("Nothing to do", out)

    def test_other_spellings_of_the_same_ignore_count(self):
        (self.repo / ".gitignore").write_text("# mine\nruns\n**/.label_secret\n", encoding="utf-8")
        code, out, _ = run_cli("init", self.repo)
        self.assertEqual(code, 0)
        self.assertEqual((self.repo / ".gitignore").read_text(encoding="utf-8"), "# mine\nruns\n**/.label_secret\n")
        self.assertIn("kept   .gitignore", out)

    def test_only_the_missing_entry_is_added(self):
        (self.repo / ".gitignore").write_text("runs/\n", encoding="utf-8")
        run_cli("init", self.repo)
        self.assertEqual(self.lines().count("runs/"), 1)
        self.assertEqual(self.lines().count(".label_secret"), 1)

    def test_the_committed_baseline_is_not_ignored(self):
        run_cli("init", self.repo)
        self.assertNotIn("evals/", self.lines())
        self.assertNotIn("baseline/", self.lines())
        self.assertFalse(any("baseline" in line and not line.startswith("#") for line in self.lines()))

    def test_a_directory_named_gitignore_is_an_error_and_writes_nothing(self):
        (self.repo / ".gitignore").mkdir()
        code, _, err = run_cli("init", self.repo)
        self.assertEqual(code, 2)
        self.assertIn(".gitignore is a directory", err)
        self.assertFalse((self.repo / "evals").exists())

    def test_no_temp_file_is_left_next_to_it(self):
        (self.repo / ".gitignore").write_text("x\n", encoding="utf-8")
        run_cli("init", self.repo)
        self.assertEqual([p.name for p in self.repo.iterdir() if ".tmp" in p.name], [])

    def test_the_next_steps_start_with_the_offline_smoke_run(self):
        _, out, _ = run_cli("init", self.repo)
        self.assertIn("JUDGE_PROVIDER=mock evalsprint run -c evals/eval.yaml", out)
        self.assertLess(out.index("JUDGE_PROVIDER=mock"), out.index("ANTHROPIC_API_KEY"))  # before the real (paid) one


class HelpTests(InitCase):
    def test_help_lists_init_and_its_options(self):
        code, out, _ = run_cli("--help")
        self.assertEqual(code, 0)
        self.assertIn("evalsprint init", out)  # the typical flow starts with it
        code, out, _ = run_cli("init", "--help")
        self.assertEqual(code, 0)
        for expected in ("DIR", "--ci", "--force"):
            self.assertIn(expected, out)

    def test_unknown_option_is_a_usage_error_and_writes_nothing(self):
        code, _, err = run_cli("init", self.repo, "--nope")
        self.assertEqual(code, 2)
        self.assertEqual(files_under(self.repo), [])


class ScaffoldRunsTests(InitCase):
    def test_the_scaffolded_adapter_answers_offline(self):
        run_cli("init", self.repo)
        import importlib.util

        spec = importlib.util.spec_from_file_location("scaffold_adapter", self.repo / "evals" / "target_adapter.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        answer = module.offline_stub("How do I get a refund?", {})
        text = answer["text"] if isinstance(answer, dict) else answer
        self.assertTrue(str(text).strip())


if __name__ == "__main__":
    unittest.main()
