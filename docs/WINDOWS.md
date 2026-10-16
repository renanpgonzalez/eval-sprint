# Running the kit on Windows

The kit is plain Python and runs on Windows 10 and 11. This page is the short list of things that differ from Linux and macOS. The CI workflows run on Linux runners; nothing in them needs Windows.

> **What was and was not verified.** The test suite was run on a real Windows 11 machine with Python 3.12 and passes (section 6; the fixes that run led to are under *Unreleased* in the [changelog](../CHANGELOG.md)). The rest of this page, the install steps, the console and PowerShell commands, labeling and the Excel notes, still comes from reading the code and from simulating a legacy console encoding on Linux (`PYTHONIOENCODING=cp1252`): nobody has followed it step by step on Windows. Treat it as a checklist, and tell us which line was wrong.

## 1. Install

Use the Python launcher, not `python3`. On a stock Windows the name `python3` is often a stub that opens the Microsoft Store.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m evalsprint --version
```

Calling the interpreter inside `.venv` directly avoids two Windows traps: PowerShell's default execution policy blocks `.\.venv\Scripts\Activate.ps1`, and after activation `evalsprint` is only on `PATH` for that shell. If you prefer to activate: `Set-ExecutionPolicy -Scope Process Bypass`, then `.\.venv\Scripts\Activate.ps1`. `uv venv` and `uv pip install -e .` also work.

`python -m evalsprint ...` is the same as `evalsprint ...`, and is the form to use when the command is not found.

Python 3.10 to 3.13 are supported (the tests run on all four on Linux). `py -0p` lists what is installed.

## 2. Encoding: set UTF-8 once

Reports, labels and run files are always written as UTF-8, whatever the console does. What can go wrong is **printing**: a redirected or piped legacy console (`>`, `|`, `Tee-Object`, a CI runner) uses a code page such as cp1252, and characters like arrows, check marks and emoji in a comparison or in your own test data cannot be represented.

The kit writes files first and prints afterwards, and replaces characters the console cannot show (a `?` instead of a crash), so a comparison is never lost. To see the real characters, ask Python for UTF-8:

```powershell
$env:PYTHONUTF8 = "1"                                 # for this shell
[Console]::OutputEncoding = [Text.Encoding]::UTF8     # so PowerShell decodes what Python prints
```

To make it permanent for your user: `setx PYTHONUTF8 1` (applies to new shells). In `cmd.exe`: `chcp 65001` and `set PYTHONUTF8=1`.

PowerShell 5.1's `>` writes UTF-16. When you need a markdown or JSON file, use the kit's own `--md`, `--json` and `-o` options (always UTF-8), not shell redirection.

## 3. Paths

- **Spaces are fine, quote them.** `evalsprint run -c "C:\Users\Renan Gonzalez\evals\eval.yaml"`. A full suite was run from a folder with spaces in its path, with CRLF files and a BOM in `cases.jsonl` (on Linux).
- **In `eval.yaml`, use forward slashes, or single quotes.** In a double-quoted YAML string a backslash starts an escape sequence, so `callable: "C:\Users\me\bot.py:answer"` is a YAML error. Write `callable: 'C:\Users\me\bot.py:answer'` or `callable: "C:/Users/me/bot.py:answer"`. Relative paths (the usual case) are resolved against the folder that contains `eval.yaml`, and may use `/` on Windows.
- **`file.py:function` with a drive letter** is understood (the function name is taken after the last colon). Keep the extension lowercase (`.py`, not `.PY`) and always name the function.
- **Long paths.** Windows limits paths to 260 characters unless long paths are enabled. If `git` or `pip` reports "filename too long": `git config --global core.longpaths true`, and keep the repository near the drive root.
- **Line endings.** CRLF files and a UTF-8 BOM in `cases.jsonl` are read correctly. Add `* text=auto` to `.gitattributes` in the client repository so the committed baseline does not flip between LF and CRLF and show up as changed in every pull request.

## 4. Labeling on Windows

The first `evalsprint label` of a run creates `.label_secret` in the run folder: the key of the tokens that hide the judge's verdict in the exported labels. The kit creates it with mode 0600, and Windows ignores POSIX mode bits (it only has a read-only flag), so there the secret is protected only by the permissions of the folder. Keep run folders out of shared drives and out of git (`runs/` and `.label_secret` belong in `.gitignore`). If the file is lost, `calibrate` still works: it recomputes the strata from the run and warns.

Prefer the HTML labeling page (`evalsprint label RUN -o labeling.html`) and open it in a browser; it saves as you go and downloads `labels.jsonl`. The CSV variant (`--csv`) works, but Excel rewrites cells on save: a `case_id` made only of digits loses its leading zeros, and a 12-character `output_sha` made only of digits (or shaped like `123e456`) can be turned into a number, which makes the label count as "sha mismatch". This was seen by reading the data, not by opening Excel; if you must use Excel, format the `case_id` and `output_sha` columns as text before pasting.

## 5. Starting a suite, interrupted runs and locked files

`evalsprint init --ci` works from PowerShell or cmd exactly as on Linux (quote a path with spaces). For the offline smoke run set the variable for the shell first: `$env:JUDGE_PROVIDER = "mock"` (PowerShell) or `set JUDGE_PROVIDER=mock` (cmd), then `evalsprint run -c evals\eval.yaml --label smoke`. The placeholder judge proves the pipeline runs; it measures nothing.

A run that is interrupted (Ctrl-C, a closed laptop, a provider outage) keeps the finished attempts in `results.partial.jsonl` and prints the command that continues it: `evalsprint run -c evals\eval.yaml --resume runs\<id>`. The finished cases are not paid for again.

Windows refuses to replace a file that another process has open, for example an antivirus scanning a report that was just written or a browser showing `report.html`. Every atomic write in the kit (`results.jsonl`, `summary.json`, `run_meta.json`, the files `init` writes) now retries a `PermissionError` five times with a growing pause (0.2 s, 0.4 s, 0.8 s, ...) before it gives up. That behavior is tested with a simulated lock on Linux, not with a real antivirus: if you still see "Access is denied" on a run file, close what has the folder open and run the command again.

## 6. Running the tests and the demo

```powershell
py -3.12 -m unittest discover -s tests -p "test_*.py"
py -3.12 examples\support_bot\run_demo.py
```

The part of `tests/test_templates.py` that executes the CI workflows skips itself on Windows (it needs `bash`); `tests/test_packaging.py` builds the wheel and skips itself when `setuptools` and `wheel` are not installed; every other test runs. On Windows 11 with Python 3.12 the whole suite passes (2513 tests, 124 skipped, about 4.5 minutes). The demo writes to `demo_out\`.

## 7. Environment variables and secrets

Set provider keys for the current shell with `$env:ANTHROPIC_API_KEY = "sk-..."` (PowerShell) or `set ANTHROPIC_API_KEY=sk-...` (cmd). `${VAR}` and `${VAR:-default}` in `eval.yaml` read them. Keys are never written to disk or printed by the kit.

`summary.json` and `results.jsonl` are meant to be committed as the baseline. They store the dataset path relative to `eval.yaml` and trim tracebacks to repository-relative paths, so `C:\Users\<you>\...` does not end up in a client repository. Check the first committed baseline for absolute paths anyway.
