"""Resume refinements (contract C20): the fingerprint covers the target's source file, `--retry-errors` also re-runs crashed ATTEMPTS
of finished repeated cases, SIGTERM is handled like Ctrl-C, a resume takes a lock, and a failed final save says how to continue.

Builds on the fixture of tests/test_resume.py (a target that logs every call and can be told to stop or crash at chosen inputs).
"""

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from evalsprint import runner
from evalsprint.cli import main
from evalsprint.runner import LOCK_NAME, PARTIAL_NAME, RUN_META_NAME, ResumeLocked
from tests import test_resume

REPO_ROOT = Path(__file__).resolve().parents[1]


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def wait_for(predicate, timeout=30.0, what="the condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


class TargetSourceFingerprintTests(test_resume.ResumeFixture):
    """A changed adapter means every old record came from different code: nothing is reused, with a warning."""

    def resume(self, run_dir):
        self.reset_calls()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            run = self.run_it(resume_from=run_dir)
        return run, err.getvalue()

    def test_the_summary_and_the_fingerprint_carry_the_sha_of_the_adapter_source(self):
        run = self.run_it()
        self.assertRegex(run.summary["target"]["source_sha"], r"^[0-9a-f]{12}$")
        meta = json.loads((Path(run.run_dir) / RUN_META_NAME).read_text(encoding="utf-8"))
        self.assertEqual(meta["fingerprint"]["target"]["source_sha"], run.summary["target"]["source_sha"])

    def test_an_untouched_adapter_is_reused(self):
        finished = self.run_it()
        run, err = self.resume(finished.run_dir)
        self.assertEqual((self.calls(), run.summary["resumed_from"]["reused_records"], err), ([], 6, ""))
        self.assertNotIn("fingerprint_mismatch", run.summary["resumed_from"])

    def test_an_edited_adapter_reuses_nothing_and_warns(self):
        finished = self.run_it()
        with open(self.dir / "target.py", "a", encoding="utf-8") as fh:
            fh.write("\n# a pull request changed what the feature answers\n")
        run, err = self.resume(finished.run_dir)
        self.assertEqual(len(self.calls()), 6)  # every case ran again
        self.assertEqual(run.summary["resumed_from"]["reused_records"], 0)
        self.assertEqual(run.summary["resumed_from"]["fingerprint_mismatch"], ["target_source"])
        self.assertIn("target_source differ", err)
        self.assertIn("the source file of the target callable was edited", err)
        self.assertNotEqual(run.summary["target"]["source_sha"], finished.summary["target"]["source_sha"])

    def test_an_interrupted_run_is_not_resumed_with_an_edited_adapter_either(self):
        run_dir = self.interrupted("q3")
        (self.dir / "target.py").write_text((self.dir / "target.py").read_text(encoding="utf-8").replace('"nope"', '"still nope"'), encoding="utf-8")
        run, err = self.resume(run_dir)
        self.assertEqual(run.summary["resumed_from"]["fingerprint_mismatch"], ["target_source"])
        self.assertEqual(len(self.calls()), 6)  # the 3 cases the old adapter answered were paid for again: they came from other code
        self.assertEqual(run.summary["resumed_from"]["reused_records"], 0)

    def test_line_endings_and_a_bom_are_not_an_edit(self):
        finished = self.run_it()
        raw = (self.dir / "target.py").read_bytes()
        (self.dir / "target.py").write_bytes(b"\xef\xbb\xbf" + raw.replace(b"\n", b"\r\n"))  # a Windows checkout of the same file
        run, _ = self.resume(finished.run_dir)
        self.assertEqual((self.calls(), run.summary["resumed_from"]["reused_records"]), ([], 6))

    def test_the_mismatch_names_every_part_that_differs(self):
        finished = self.run_it()
        with open(self.dir / "target.py", "a", encoding="utf-8") as fh:
            fh.write("\n# edited\n")
        self.reset_calls()
        with contextlib.redirect_stderr(io.StringIO()):
            run = self.run_it(["target.params.prompt=v2", "judges.0.values=[answer, to]"], resume_from=finished.run_dir)
        self.assertEqual(run.summary["resumed_from"]["fingerprint_mismatch"], ["judges_sha", "target", "target_source"])

    def test_a_finished_run_without_run_meta_still_compares_the_source(self):
        finished = self.run_it()
        (Path(finished.run_dir) / RUN_META_NAME).unlink()  # the fingerprint is rebuilt from summary.json
        run, err = self.resume(finished.run_dir)
        self.assertEqual((self.calls(), run.summary["resumed_from"]["reused_records"]), ([], 6))
        with open(self.dir / "target.py", "a", encoding="utf-8") as fh:
            fh.write("\n# edited\n")
        run, err = self.resume(finished.run_dir)
        self.assertEqual(run.summary["resumed_from"]["fingerprint_mismatch"], ["target_source"])

    def test_the_cli_prints_the_warning_and_the_run_still_succeeds(self):
        run_dir = self.interrupted("q3")
        with open(self.dir / "target.py", "a", encoding="utf-8") as fh:
            fh.write("\n# edited\n")
        code, out, err = cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report", "--resume", run_dir)
        self.assertEqual(code, 0, err)
        self.assertIn("the source file of the target callable was edited", err)
        self.assertIn("every case runs again", err)
        self.assertEqual(len(self.calls()), 6)


class RetryCrashedAttemptsTests(test_resume.ResumeFixture):
    """`--retry-errors` on a FINISHED repeated run: the crashed attempts hidden behind a standing majority run again, the rest is kept."""

    def crashed_run(self, crash="q1|1", n=3, **kw):
        os.environ["RESUME_TEST_CRASH"] = crash
        try:
            run = self.run_it(repeat=n, **kw)
        finally:
            os.environ.pop("RESUME_TEST_CRASH")
        self.reset_calls()
        return run

    def test_only_the_crashed_attempt_runs_again_and_the_record_equals_a_clean_run(self):
        clean = self.run_it(repeat=3)
        self.reset_calls()
        crashed = self.crashed_run()
        record = crashed.by_case()["c1"]
        self.assertEqual((record["outcome"], record["n_attempt_errors"]), ("pass", 1))
        self.assertEqual(crashed.summary["n_attempt_errors"], 1)

        fixed = self.run_it(repeat=3, resume_from=crashed.run_dir, retry_errors=True)
        self.assertEqual(self.calls(), ["q1|1"])  # one attempt, not 18
        self.assertEqual(fixed.by_case()["c1"]["n_attempt_errors"], 0)
        self.assertEqual(fixed.summary["n_attempt_errors"], 0)
        self.assertEqual(fixed.summary["attempt_error_rate"], 0.0)
        self.assertEqual(self.comparable(fixed), self.comparable(clean))
        self.assertEqual(fixed.summary["resumed_from"]["retried_errors"], 1)
        self.assertEqual(fixed.summary["resumed_from"]["reused_records"], 6)

    def test_without_retry_errors_the_crashed_attempt_is_kept(self):
        crashed = self.crashed_run()
        again = self.run_it(repeat=3, resume_from=crashed.run_dir)
        self.assertEqual(self.calls(), [])
        self.assertEqual(again.summary["n_attempt_errors"], 1)

    def test_a_crash_that_is_still_there_stays_an_attempt_error(self):
        crashed = self.crashed_run()
        os.environ["RESUME_TEST_CRASH"] = "q1|1"
        try:
            again = self.run_it(repeat=3, resume_from=crashed.run_dir, retry_errors=True)
        finally:
            os.environ.pop("RESUME_TEST_CRASH")
        self.assertEqual(self.calls(), ["q1|1", "q1|1"])  # the attempt and its one in-run retry
        self.assertEqual((again.by_case()["c1"]["outcome"], again.by_case()["c1"]["n_attempt_errors"]), ("pass", 1))

    def test_a_majority_of_crashed_attempts_reruns_the_whole_case(self):
        crashed = self.crashed_run(crash="q1|0,q1|1")  # 2 of 3 crashed: the case is an error
        self.assertEqual(crashed.by_case()["c1"]["outcome"], "error")
        fixed = self.run_it(repeat=3, resume_from=crashed.run_dir, retry_errors=True)
        self.assertEqual(self.calls(), ["q1|0", "q1|1", "q1|2"])
        self.assertEqual((fixed.by_case()["c1"]["outcome"], fixed.by_case()["c1"]["n_attempt_errors"]), ("pass", 0))
        self.assertEqual(fixed.summary["resumed_from"]["retried_errors"], 1)

    def test_an_error_case_is_kept_without_retry_errors(self):
        crashed = self.crashed_run(crash="q1|0,q1|1")
        again = self.run_it(repeat=3, resume_from=crashed.run_dir)
        self.assertEqual((self.calls(), again.by_case()["c1"]["outcome"]), ([], "error"))

    def test_a_case_whose_verdict_flips_when_the_crashed_attempts_come_back(self):
        # 5 attempts, 2 crashed: the 3 that ran FAILED ("bad" input) -> fail; the retried ones fail too -> still fail, recomputed over all 5
        crashed = self.crashed_run(crash="bad-2|0,bad-2|1", n=5)
        record = crashed.by_case()["c2"]
        self.assertEqual((record["outcome"], record["n_attempt_errors"]), ("fail", 2))
        fixed = self.run_it(repeat=5, resume_from=crashed.run_dir, retry_errors=True)
        self.assertEqual(self.calls(), ["bad-2|0", "bad-2|1"])
        self.assertEqual((fixed.by_case()["c2"]["outcome"], fixed.by_case()["c2"]["n_attempt_errors"]), ("fail", 0))
        self.assertEqual(fixed.by_case()["c2"]["passed"], False)

    def test_the_cost_of_a_crashed_attempt_that_ran_again_is_not_counted_twice(self):
        clean = self.run_it(repeat=3)
        crashed = self.crashed_run()
        fixed = self.run_it(repeat=3, resume_from=crashed.run_dir, retry_errors=True)
        self.assertAlmostEqual(fixed.summary["cost"]["system_total_usd"], clean.summary["cost"]["system_total_usd"])
        self.assertEqual(fixed.summary["resumed_from"]["discarded_cost_usd"], 0.0)  # a raise cost nothing

    def test_an_interrupted_retry_resumes_from_where_it_stopped(self):
        clean = self.run_it(repeat=5)
        self.reset_calls()
        crashed = self.crashed_run(crash="q1|0,q1|1", n=5)  # 2 of 5: the case still passes by majority
        self.assertEqual(crashed.by_case()["c1"]["n_attempt_errors"], 2)

        os.environ["RESUME_TEST_STOP_AT"] = "q1|1"  # Ctrl-C while the second crashed attempt is running again
        try:
            with self.assertRaises(KeyboardInterrupt) as ctx:
                self.run_it(repeat=5, resume_from=crashed.run_dir, retry_errors=True)
        finally:
            os.environ.pop("RESUME_TEST_STOP_AT")
        interrupted = Path(ctx.exception.run_dir)
        self.assertEqual(self.calls(), ["q1|0", "q1|1"])
        loose = [line for line in self.partial_lines(interrupted) if line.get("attempt") is not None]
        self.assertEqual([(line["case_id"], line["attempt"]) for line in loose], [("c1", 0)])  # the retried attempt 0 is saved
        self.reset_calls()

        done = self.run_it(repeat=5, resume_from=interrupted, retry_errors=True)
        self.assertEqual(self.calls(), ["q1|1"])  # only what is still missing
        self.assertEqual(done.by_case()["c1"]["n_attempt_errors"], 0)
        self.assertEqual(self.comparable(done), self.comparable(clean))

    def test_the_cli_reports_the_number_of_retried_attempts(self):
        crashed = self.crashed_run()
        code, out, err = cli("run", "-c", self.dir / "eval.yaml", "--set", "repeat={n: 3}", "--quiet", "--no-report", "--resume", crashed.run_dir, "--retry-errors")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.calls(), ["q1|1"])
        self.assertIn("Resumed:", out)
        self.assertNotIn("Attempt errors", out)


class ResumeLockTests(test_resume.ResumeFixture):
    def resume_args(self, run_dir):
        return [sys.executable, "-m", "evalsprint", "run", "-c", str(self.dir / "eval.yaml"), "--resume", str(run_dir), "--quiet", "--no-report"]

    def start_slow_resume(self, run_dir):
        os.environ["RESUME_TEST_SLEEP"] = "1.0"
        try:
            proc = subprocess.Popen(self.resume_args(run_dir), cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        finally:
            os.environ.pop("RESUME_TEST_SLEEP")
        self.addCleanup(lambda: proc.poll() is None and (proc.kill(), proc.communicate()))
        wait_for(lambda: (Path(run_dir) / LOCK_NAME).exists() and self.calls_file.exists(), what="the first resume to start working")
        return proc

    def test_a_second_concurrent_resume_exits_2_and_pays_nothing(self):
        run_dir = self.interrupted("q3")
        first = self.start_slow_resume(run_dir)
        second = subprocess.run(self.resume_args(run_dir), cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(second.returncode, 2, second.stderr)
        self.assertIn("another `evalsprint run --resume` is already working on", second.stderr)
        self.assertIn(str(run_dir), second.stderr)
        self.assertIn("twice", second.stderr)  # why it matters: every missing case would be paid twice
        if hasattr(os, "pread"):  # the holder's pid is read from the locked file: POSIX only (Windows locks that byte range against readers)
            self.assertIn(f"held by process {first.pid}", second.stderr)
        _, err = first.communicate(timeout=120)
        self.assertEqual(first.returncode, 0, err)
        inputs = [line.split("|")[0] for line in self.calls()]
        self.assertEqual(sorted(inputs), sorted(set(inputs)))  # no case was called twice across the two processes

    def test_the_lock_file_is_removed_when_the_resume_ends(self):
        run_dir = self.interrupted("q3")
        code, _, err = cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report", "--resume", run_dir)
        self.assertEqual(code, 0, err)
        self.assertFalse((run_dir / LOCK_NAME).exists())

    def test_the_lock_is_released_when_the_resume_fails(self):
        run_dir = self.interrupted("q3")
        os.environ["RESUME_TEST_STOP_AT"] = "q4"
        try:
            code, _, _ = cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report", "--resume", run_dir)
        finally:
            os.environ.pop("RESUME_TEST_STOP_AT")
        self.assertEqual(code, 130)
        self.assertFalse((run_dir / LOCK_NAME).exists())
        code, _, err = cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report", "--resume", run_dir)
        self.assertEqual(code, 0, err)  # and the same directory can be resumed again

    def test_a_killed_resume_does_not_leave_a_lock_behind(self):
        run_dir = self.interrupted("q3")
        first = self.start_slow_resume(run_dir)
        first.kill()
        first.communicate(timeout=60)
        self.assertTrue((run_dir / LOCK_NAME).exists())  # the file stays, but the operating system dropped the lock with the process
        code, _, err = cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report", "--resume", run_dir)
        self.assertEqual(code, 0, err)
        self.assertFalse((run_dir / LOCK_NAME).exists())

    def test_the_same_process_cannot_hold_two_locks_on_one_directory(self):
        with runner._resume_lock(self.dir):
            with self.assertRaises(ResumeLocked):
                with runner._resume_lock(self.dir):
                    self.fail("the second lock must not be granted")
            self.assertTrue((self.dir / LOCK_NAME).exists())  # the failed attempt did not remove the holder's file
        self.assertFalse((self.dir / LOCK_NAME).exists())

    def test_different_directories_do_not_block_each_other(self):
        other = self.dir / "other"
        other.mkdir()
        with runner._resume_lock(self.dir), runner._resume_lock(other):
            pass

    @unittest.skipIf(os.name == "nt", "Windows locks the byte range against every other reader, this one included")
    def test_the_holder_writes_its_pid(self):
        with runner._resume_lock(self.dir):
            self.assertEqual((self.dir / LOCK_NAME).read_text(encoding="ascii").strip(), str(os.getpid()))

    def test_no_resume_no_lock(self):
        with runner._resume_lock(None):
            pass
        with runner._resume_lock(self.dir / "does-not-exist"):  # `_load_resume` explains that one
            pass
        run = self.run_it()
        self.assertEqual([p.name for p in Path(run.run_dir).iterdir() if p.name == LOCK_NAME], [])

    def test_a_directory_that_cannot_hold_the_lock_file_still_resumes_with_a_warning(self):
        run_dir = self.interrupted("q3")
        real_open = os.open

        def refuse(path, *a, **kw):
            if str(path).endswith(LOCK_NAME):
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, *a, **kw)

        err = io.StringIO()
        with mock.patch("evalsprint.runner.os.open", refuse), contextlib.redirect_stderr(err):
            run = self.run_it(resume_from=run_dir)
        self.assertEqual(run.summary["resumed_from"]["reused_records"], 3)
        self.assertIn("could not take the resume lock", err.getvalue())

    def test_the_lock_error_is_a_usage_error_not_a_crash(self):
        run_dir = self.interrupted("q3")
        with runner._resume_lock(run_dir):
            code, _, err = cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report", "--resume", run_dir)
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("error: another `evalsprint run --resume`"), err)
        self.assertEqual(self.calls(), [])  # nothing ran


@unittest.skipUnless(hasattr(signal, "SIGTERM") and os.name == "posix", "SIGTERM delivery to a child needs POSIX")
class SigtermTests(test_resume.ResumeFixture):
    def run_args(self, *extra):
        return [sys.executable, "-m", "evalsprint", "run", "-c", str(self.dir / "eval.yaml"), "--quiet", "--no-report", *extra]

    def start(self, args, sleep="0.5"):
        os.environ["RESUME_TEST_SLEEP"] = sleep
        try:
            # A test runner started as a background job (`cmd &` in a script, nohup) has SIGINT ignored, and a child inherits that: Python
            # then never installs its KeyboardInterrupt handler and Ctrl-C does nothing. Give the child the default disposition back.
            proc = subprocess.Popen(
                args, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
            )
        finally:
            os.environ.pop("RESUME_TEST_SLEEP")
        self.addCleanup(lambda: proc.poll() is None and (proc.kill(), proc.communicate()))
        return proc

    def partial_of(self, proc):
        runs = self.dir / "runs"
        wait_for(lambda: runs.is_dir() and any((d / PARTIAL_NAME).is_file() and (d / PARTIAL_NAME).stat().st_size > 0 for d in runs.iterdir()), what="a finished case in results.partial.jsonl")
        return next(d for d in runs.iterdir() if (d / PARTIAL_NAME).is_file())

    def test_sigterm_drains_keeps_the_partial_and_prints_the_resume_hint(self):
        proc = self.start(self.run_args())
        run_dir = self.partial_of(proc)
        proc.terminate()  # SIGTERM: a CI cancel, `kill`, a container stop
        out, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 143, err)
        self.assertEqual(err.splitlines()[0], "terminated")
        self.assertIn(str(run_dir / PARTIAL_NAME), err)
        self.assertIn("--resume", err.splitlines()[-1])
        self.assertIn(str(run_dir), err.splitlines()[-1])
        self.assertTrue((run_dir / PARTIAL_NAME).is_file())
        self.assertFalse((run_dir / "summary.json").exists())
        saved = self.partial_lines(run_dir)
        self.assertGreaterEqual(len(saved), 1)
        self.assertLess(len(saved), 6)

    def test_what_sigterm_kept_is_what_the_resume_does_not_pay_for_again(self):
        proc = self.start(self.run_args())
        run_dir = self.partial_of(proc)
        proc.terminate()
        proc.communicate(timeout=60)
        done = {line["case_id"] for line in self.partial_lines(run_dir)}
        self.reset_calls()
        code, out, err = cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report", "--resume", run_dir)
        self.assertEqual(code, 0, err)
        called = {line.split("|")[0] for line in self.calls()}
        paid = {f"q{i}" if i != 2 else "bad-2" for i in range(6)}
        self.assertEqual(len(called), 6 - len(done))
        self.assertTrue(called <= paid)
        self.assertIn("Resumed:", out)

    def test_sigint_keeps_its_own_exit_code(self):
        proc = self.start(self.run_args())
        run_dir = self.partial_of(proc)
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 130, err)
        self.assertEqual(err.splitlines()[0], "interrupted")
        self.assertTrue((run_dir / PARTIAL_NAME).is_file())

    def test_the_handler_is_removed_when_the_run_ends(self):
        before = signal.getsignal(signal.SIGTERM)
        self.assertEqual(cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report")[0], 0)
        self.assertIs(signal.getsignal(signal.SIGTERM), before)

    def test_outside_the_main_thread_nothing_is_installed(self):
        import threading

        from evalsprint import cli as cli_module

        seen = []

        def body():
            before = signal.getsignal(signal.SIGTERM)
            with cli_module._sigterm_as_interrupt():
                seen.append(signal.getsignal(signal.SIGTERM) is before)

        thread = threading.Thread(target=body)
        thread.start()
        thread.join()
        self.assertEqual(seen, [True])  # a thread cannot install handlers: the run is simply not protected there


class FailedFinalSaveTests(test_resume.ResumeFixture):
    def test_a_failed_final_save_keeps_the_partial_and_prints_the_resume_hint(self):
        with mock.patch("evalsprint.runner.save_run", side_effect=OSError(28, "No space left on device")):
            code, _, err = cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report")
        self.assertEqual(code, 2)
        self.assertIn("No space left on device", err)
        self.assertIn("--resume", err)
        run_dir = next((self.dir / "runs").iterdir())
        self.assertTrue((run_dir / PARTIAL_NAME).is_file())
        self.assertEqual(len(self.partial_lines(run_dir)), 6)  # all six cases were paid for and are safe
        self.assertIn(str(run_dir), err)

    def test_the_hint_works(self):
        with mock.patch("evalsprint.runner.save_run", side_effect=OSError(28, "No space left on device")):
            cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report")
        run_dir = next((self.dir / "runs").iterdir())
        self.reset_calls()
        code, out, err = cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report", "--resume", run_dir)
        self.assertEqual((code, self.calls()), (0, []))  # nothing is paid again
        self.assertIn("Resumed:", out)
        self.assertIn("6 record(s)", out)

    def test_the_exception_knows_the_run_directory(self):
        with mock.patch("evalsprint.runner.save_run", side_effect=OSError(28, "No space left on device")):
            with self.assertRaises(OSError) as ctx:
                self.run_it()
        self.assertTrue((Path(ctx.exception.run_dir) / PARTIAL_NAME).is_file())


if __name__ == "__main__":
    unittest.main()
