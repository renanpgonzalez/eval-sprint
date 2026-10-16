"""C13 packaging: the wheel carries the templates, and `evalsprint init` works from an installed copy, not from a checkout.

A client installs the kit with `pip install git+https://...` (pip builds a wheel from the repository) and then runs
`evalsprint init`. Everything `init` writes is read from `evalsprint/templates/` inside the installed package, so a template
missing from the wheel would only show up in a client's repository. These tests build the wheel for real, in a scratch copy of
the project (the repository never gets a `build/` or an `*.egg-info/`), install it into a temporary directory and drive the
scaffold through its whole offline path from there.

The wheel is pure Python (`py3-none-any`), so any interpreter that can build it yields the same file; the install and every
command run on the interpreter that runs the tests. The build needs no network (`--no-build-isolation --no-index`) and therefore
a setuptools and a wheel package installed in some interpreter (the test is skipped when none of them has both). Build
toolchains differ by machine: on Debian and Ubuntu the system setuptools needs `SETUPTOOLS_USE_DISTUTILS=stdlib` to build any
project, and on Python 3.13 that same system setuptools (68.1.2) fails with `install_layout` or `distutils` errors in both modes.
So the build tries each candidate interpreter in both modes and uses the first that works; a machine where setuptools and wheel
import but nothing builds is a failure, not a skip.
"""

import configparser
import email
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from tests import ROOT, TEMPLATES

# What `evalsprint init` needs to find in the install, by candidate name (mirrors cli._SCAFFOLD / cli._SCAFFOLD_CI).
CORE = ["evals/cases.jsonl", "evals/eval.yaml", "evals/requirements.txt", "evals/rubric.yaml", "evals/target_adapter.py"]
CI = [".github/CODEOWNERS.evals.example", ".github/workflows/evals-baseline.yml", ".github/workflows/evals.yml"]


def _available(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)


def _version_in_pyproject() -> str:
    return re.search(r'(?m)^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8")).group(1)


def _template_files() -> dict[str, bytes]:
    """{path inside the wheel: bytes} for every file of the templates tree in the source checkout."""
    out = {}
    for path in sorted(TEMPLATES.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
            out["evalsprint/templates/" + path.relative_to(TEMPLATES).as_posix()] = path.read_bytes()
    return out


def _candidate_pythons() -> list[str]:
    """The interpreter running the tests first, then the other CPython 3 interpreters on the PATH (deduplicated by real path)."""
    seen, out = set(), []
    for exe in [sys.executable, *(shutil.which(name) for name in ("python3", "python3.10", "python3.11", "python3.12", "python3.13", "python"))]:
        if exe and os.path.realpath(exe) not in seen:
            seen.add(os.path.realpath(exe))
            out.append(exe)
    return out


def _can_build(python: str) -> bool:
    """pip, setuptools and wheel all usable by this interpreter. (Separate processes: on Debian `import pip` before
    `import setuptools` in ONE process trips setuptools' distutils hack, which says nothing about whether a build works.)"""
    checks = (["-c", "import setuptools, wheel"], ["-m", "pip", "--version"])
    return all(subprocess.run([python, *check], capture_output=True).returncode == 0 for check in checks)


def _stray_build_artifacts() -> list[str]:
    return sorted([p.name for p in ROOT.glob("*.egg-info")] + (["build"] if (ROOT / "build").exists() else []) + (["dist"] if (ROOT / "dist").exists() else []))


@unittest.skipUnless(_available("pip"), "pip is needed to install the wheel")
class InstalledKitTests(unittest.TestCase):
    """One wheel per test class: built once, installed once, then read and driven by the tests below."""

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.tmp = Path(tmp.name)
        cls.stray_before = _stray_build_artifacts()
        src = cls.tmp / "src"
        src.mkdir()
        for name in ("pyproject.toml", "README.md"):
            shutil.copy2(ROOT / name, src / name)
        shutil.copytree(ROOT / "evalsprint", src / "evalsprint", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        cls.env = {
            **{k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "OPENAI_", "EVALSPRINT_", "JUDGE_")) and k not in ("PYTHONPATH", "PYTHONSTARTUP")},
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        cls.wheels = cls.tmp / "wheels"
        builders = [python for python in _candidate_pythons() if _can_build(python)]
        if not builders:
            raise unittest.SkipTest("no interpreter on this machine has pip, setuptools and wheel: nothing can build the wheel offline")
        attempts = []
        for python in builders:
            for mode in ("default", "stdlib"):
                env = {**cls.env, **({"SETUPTOOLS_USE_DISTUTILS": "stdlib"} if mode == "stdlib" else {})}
                shutil.rmtree(cls.wheels, ignore_errors=True)
                build = subprocess.run(
                    [python, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "--no-index", "-w", str(cls.wheels), str(src)],
                    cwd=cls.tmp, env=env, capture_output=True, text=True, encoding="utf-8",
                )
                if build.returncode == 0 and list(cls.wheels.glob("*.whl")):
                    cls.builder = f"{python} (SETUPTOOLS_USE_DISTUTILS={'stdlib' if mode == 'stdlib' else 'unset'})"
                    break
                attempts.append(f"{python} [{mode}] exit {build.returncode}: {(build.stdout + build.stderr).strip().splitlines()[-1:]}")
            else:
                continue
            break
        else:
            raise AssertionError("setuptools and wheel import, but the wheel did not build with any interpreter or mode:\n" + "\n".join(attempts))
        print(f"\n[test_packaging] wheel built by {cls.builder}; installed and driven by {sys.executable} ({sys.version.split()[0]})", file=sys.stderr)
        (cls.wheel,) = sorted(cls.wheels.glob("*.whl"))
        cls.site = cls.tmp / "site"
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--no-compile", "--target", str(cls.site), str(cls.wheel)],
            cwd=cls.tmp, env=cls.env, capture_output=True, text=True, encoding="utf-8",
        )
        if install.returncode != 0:
            raise AssertionError(f"the wheel did not install (exit {install.returncode}):\n{(install.stdout + install.stderr)[-3000:]}")

    def cli(self, cwd: Path, *args: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
        """`python -m evalsprint ...` from the INSTALLED copy: PYTHONPATH is the install dir, the cwd is not the repository."""
        env = {**self.env, "PYTHONPATH": str(self.site), **(extra_env or {})}
        return subprocess.run([sys.executable, "-m", "evalsprint", *args], cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8")

    def workdir(self, name: str) -> Path:
        path = self.tmp / name
        path.mkdir()
        return path

    # ------------------------------------------------------------------ the wheel itself

    def test_the_wheel_is_named_and_versioned_after_pyproject(self):
        version = _version_in_pyproject()
        self.assertEqual(self.wheel.name, f"evalsprint-{version}-py3-none-any.whl")
        import evalsprint

        self.assertEqual(version, evalsprint.__version__, "pyproject.toml and evalsprint.__version__ disagree")
        self.assertEqual(version, "0.2.0")

    def test_the_wheel_contains_every_template_with_the_bytes_of_the_source_tree(self):
        expected = _template_files()
        self.assertGreaterEqual(len(expected), 9)  # the five scaffold files, the three CI files and the CI README
        with zipfile.ZipFile(self.wheel) as wheel:
            names = set(wheel.namelist())
            for name, content in expected.items():
                with self.subTest(file=name):
                    self.assertIn(name, names, "missing from the wheel: add it to [tool.setuptools.package-data]")
                    self.assertEqual(wheel.read(name), content)
            packaged = {n for n in names if n.startswith("evalsprint/templates/")}
            self.assertEqual(packaged, set(expected), "the wheel holds template files the source tree does not (or vice versa)")
            self.assertIn("evalsprint/pricing.yaml", names)
            for name in names:
                with self.subTest(unwanted=name):
                    self.assertFalse(name.startswith(("tests/", "orchestration/", "docs/", "scripts/", "examples/", "evalsprint/templates/__pycache__")), name)
                    self.assertFalse(name.endswith((".pyc", ".pyo")), name)
            # `templates/` is data: a wheel that turned it into a package would import the client adapter's sibling modules wrongly
            self.assertNotIn("evalsprint/templates/__init__.py", names)

    def test_the_metadata_says_what_the_docs_say(self):
        with zipfile.ZipFile(self.wheel) as wheel:
            (meta_name,) = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
            meta = email.message_from_string(wheel.read(meta_name).decode("utf-8"))
            (entry_name,) = [n for n in wheel.namelist() if n.endswith(".dist-info/entry_points.txt")]
            entry = configparser.ConfigParser()
            entry.read_string(wheel.read(entry_name).decode("utf-8"))
        self.assertEqual((meta["Name"], meta["Version"]), ("evalsprint", "0.2.0"))
        self.assertEqual(meta["Requires-Python"], ">=3.10")
        requirements = meta.get_all("Requires-Dist") or []
        self.assertTrue(any(r.lower().startswith("pyyaml") and ">=6.0" in r and "extra" not in r for r in requirements), requirements)
        self.assertEqual([r for r in requirements if "extra" not in r and not r.lower().startswith("pyyaml")], [], "one runtime dependency (PyYAML), as the docs say")
        self.assertIn("Private :: Do Not Upload", meta.get_all("Classifier"))  # installed from git into client repositories, never from an index
        self.assertEqual(dict(entry["console_scripts"]), {"evalsprint": "evalsprint.cli:main"})

    def test_building_the_wheel_left_nothing_behind_in_the_repository(self):
        self.assertEqual(_stray_build_artifacts(), self.stray_before)

    # ------------------------------------------------------------------ the installed copy

    def test_the_installed_copy_is_the_one_the_tests_drive(self):
        code = "import evalsprint, pathlib; print(pathlib.Path(evalsprint.__file__).resolve()); print(evalsprint.__version__)"
        work = self.workdir("which")
        done = subprocess.run([sys.executable, "-c", code], cwd=work, env={**self.env, "PYTHONPATH": str(self.site)}, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(done.returncode, 0, done.stderr)
        where, version = done.stdout.split()
        self.assertTrue(Path(where).is_relative_to(self.site.resolve()), where)
        self.assertFalse(Path(where).is_relative_to(ROOT.resolve()), "this would be a test of the checkout, not of the install")
        self.assertEqual(version, "0.2.0")

    def test_init_ci_scaffolds_everything_from_the_install(self):
        work = self.workdir("init")
        done = self.cli(work, "init", "--ci")
        self.assertEqual((done.returncode, done.stderr), (0, ""), done.stdout)
        # `init` also writes a `.gitignore` (C18); what is in it is checked against the contract in tests/test_templates.py
        written = sorted(p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file() and p.name != ".gitignore")
        self.assertEqual(written, sorted(CORE + CI))
        pairs = {
            "evals/eval.yaml": "eval.yaml",
            "evals/rubric.yaml": "rubric.yaml",
            "evals/cases.jsonl": "cases.example.jsonl",
            "evals/target_adapter.py": "target_adapter.py",
            ".github/workflows/evals.yml": "github/evals.yml",
            ".github/workflows/evals-baseline.yml": "github/evals-baseline.yml",
            ".github/CODEOWNERS.evals.example": "github/CODEOWNERS.example",
        }
        for dest, source in pairs.items():
            with self.subTest(file=dest):
                self.assertEqual((work / dest).read_bytes(), (TEMPLATES / source).read_bytes().replace(b"\r\n", b"\n"))
        # the kit is pinned to the release that wrote the scaffold (the template names the release it was cut for)
        pinned = [l for l in (work / "evals" / "requirements.txt").read_text(encoding="utf-8").splitlines() if l.startswith("evalsprint @")]
        self.assertEqual(len(pinned), 1)
        self.assertTrue(pinned[0].endswith("@v0.2.0"), pinned)
        # a second run keeps every file (nothing of the client's is overwritten) and says so
        again = self.cli(work, "init", "--ci")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("Nothing to do", again.stdout)

    def test_the_scaffold_describes_offline_and_shows_a_real_judge_by_default(self):
        work = self.workdir("describe")
        self.assertEqual(self.cli(work, "init").returncode, 0)
        done = self.cli(work, "describe", "-c", "evals/eval.yaml")
        self.assertEqual((done.returncode, done.stderr), (0, ""), done.stdout)
        self.assertIn("quality", done.stdout)
        self.assertIn("llm  anthropic/", done.stdout)  # the default judge is a real provider: CI can never be graded by the placeholder
        self.assertNotIn("mock", done.stdout)
        self.assertIn("repeat n=3", done.stdout)  # the template repeats every case below ~150 cases
        self.assertIn("Config and test set are valid.", done.stdout)

    def test_without_the_offline_switch_the_judge_needs_its_api_key(self):
        """The placeholder judge exists only behind `JUDGE_PROVIDER=mock`: a missing key is an error, not a silent pass."""
        work = self.workdir("nokey")
        self.assertEqual(self.cli(work, "init").returncode, 0)
        done = self.cli(work, "run", "-c", "evals/eval.yaml", "--no-report", "--quiet", "--out", str(work / "out"))
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn("ANTHROPIC_API_KEY", done.stderr)
        self.assertFalse((work / "out" / "summary.json").exists())

    def test_the_scaffold_runs_and_compares_offline_with_the_mock_provider(self):
        """init, then two runs on the placeholder target and judge, then the gate: the whole pipeline with no key and no network."""
        work = self.workdir("offline")
        self.assertEqual(self.cli(work, "init").returncode, 0)
        offline = {"JUDGE_PROVIDER": "mock"}
        for name in ("a", "b"):
            done = self.cli(work, "run", "-c", "evals/eval.yaml", "--label", "smoke", "--no-report", "--quiet", "--out", str(work / "evals" / "runs" / name), extra_env=offline)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        summary = json.loads((work / "evals" / "runs" / "a" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["repeat"]["n"], 3)  # the scaffold repeats every case 3 times, with no per-tag count (the exact shape is C17's: tests/test_templates.py)
        self.assertFalse(summary["repeat"].get("per_tag") or summary["repeat"].get("tags"))
        self.assertEqual((summary["n_errors"], summary["n_cases"]), (0, 10))
        (llm,) = [j for j in summary["judges_config"] if j["type"] == "llm"]
        self.assertEqual((llm["provider"], llm["provider_options"]), ("mock", {"responder": "target_adapter.py:offline_judge"}))
        self.assertEqual(summary["evalsprint_version"], "0.2.0")  # the version of the install that wrote the run
        compare = self.cli(work, "compare", str(work / "evals" / "runs" / "a"), str(work / "evals" / "runs" / "b"), "--json", str(work / "c.json"))
        self.assertEqual(compare.returncode, 0, compare.stdout + compare.stderr)
        verdict = json.loads((work / "c.json").read_text(encoding="utf-8"))
        self.assertEqual(verdict["verdict"], "pass")
        self.assertEqual(verdict["gate_mode"], "auto")
        self.assertFalse(verdict["repeat_changed"])
        self.assertEqual(verdict["repeat_protocol"], {"baseline": {"n": 3, "per_tag": {}}, "candidate": {"n": 3, "per_tag": {}}})  # normalised (C17)
        self.assertEqual(verdict["kit_version"], {"baseline": "0.2.0", "candidate": "0.2.0", "changed": False})
        # ...and the history page reads the two runs
        history = self.cli(work, "history", str(work / "evals" / "runs"), "--output", str(work / "history.html"))
        self.assertEqual(history.returncode, 0, history.stdout + history.stderr)
        self.assertTrue((work / "history.html").is_file())

    def test_the_vendored_wheel_recipe_of_the_docs_installs_from_the_repository_root(self):
        """Workflow README section 2 and `evals/requirements.txt`: a client commits the wheel under `evals/vendor/`, names it on a line of
        its own, and the workflows (which run pip from the repository root) install it with no token and no network.

        The path is relative to the directory pip runs in, NOT to the requirements file: from anywhere else pip stops with
        `No such file or directory`. Both halves are what the docs say, so both are asserted.
        """
        name = f"evalsprint-{_version_in_pyproject()}-py3-none-any.whl"
        line = f"./evals/vendor/{name}"
        for doc in (TEMPLATES / "evals" / "requirements.txt", TEMPLATES / "github" / "README.md"):
            with self.subTest(doc=doc.name):
                self.assertIn(line, doc.read_text(encoding="utf-8"))
        repo = self.workdir("vendored")
        (repo / "evals" / "vendor").mkdir(parents=True)
        shutil.copy2(self.wheel, repo / "evals" / "vendor" / name)
        (repo / "evals" / "requirements.txt").write_text(line + "\n", encoding="utf-8")
        target = self.tmp / "vendored_site"

        def pip(cwd: Path):
            return subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile", "--target", str(target), "-r", str(repo / "evals" / "requirements.txt")],
                cwd=cwd, env=self.env, capture_output=True, text=True, encoding="utf-8",
            )

        elsewhere = pip(self.workdir("elsewhere"))
        self.assertNotEqual(elsewhere.returncode, 0, "the path is relative to the working directory, so another directory must not find it")
        self.assertIn("No such file or directory", elsewhere.stdout + elsewhere.stderr)
        root = pip(repo)
        self.assertEqual(root.returncode, 0, root.stdout + root.stderr)
        self.assertTrue((target / "evalsprint" / "cli.py").is_file())

    def test_the_label_secret_and_the_runs_are_ignored_by_the_repository_gitignore(self):
        """Running the pipeline writes `runs/` and (for blind labeling) `.label_secret`: neither may ever be committed."""
        lines = {l.strip() for l in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
        for entry in (".label_secret", "runs/", "__pycache__/", "*.egg-info/", "build/", "dist/"):
            with self.subTest(entry=entry):
                self.assertIn(entry, lines)


if __name__ == "__main__":
    unittest.main()
