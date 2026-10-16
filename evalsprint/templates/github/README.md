# Eval Sprint: GitHub Actions templates

Two workflows that turn an eval suite into a merge gate for an LLM feature, plus the two small files that make the gate hard to bypass.

| File | Copy to | Purpose |
|---|---|---|
| `evals.yml` | `.github/workflows/evals.yml` | On every PR: check whether LLM code changed (`EVAL_PATHS`). If it did, run the suite on the PR's code, compare the result with the baseline **from the base branch in a separate job that runs none of the PR's code**, post one sticky comment; if not, skip the paid run. Either way the **LLM evals gate** job decides. |
| `evals-baseline.yml` | `.github/workflows/evals-baseline.yml` | Manually refresh the baseline. Opens a PR, so the reference only changes through review. Refuses an unhealthy run and a placeholder run. |
| `CODEOWNERS.example` | `.github/CODEOWNERS` | Requires the owners of the gate to approve any change to `evals/**` and to the two workflows. |
| `../evals/requirements.txt` | `evals/requirements.txt` | Pins the kit (and your target's dependencies) for CI. |

## What this gate is, and what it is not

Read this before you promise it to a client.

**It is** a regression detector against a reviewed reference. On every pull request that touches LLM code it runs a fixed test set, compares the result with the committed baseline and blocks the merge when the pass rate, a critical case, the cost, the latency or the error rate got worse, when baseline cases were removed or edited, or when the judges, the rubric, the `repeat` protocol or the source of a custom judge changed. The gate settings, the baseline and the kit that compares come from the **base branch**, in a job that does not run the pull request's code, and every way the pipeline can break (no key, no baseline, a failed job, a missing artifact) ends as a red check, never as a silent pass.

**How it decides, in one paragraph.** The gate compares each change with the committed baseline on the same cases. It blocks the change only when the cases that went from pass to fail outnumber those that went from fail to pass by at least max(3, n/50), and a one-sided exact sign test (McNemar) says that such an imbalance would be unusual by chance. The test is lenient on small suites (alpha about 24% at 45 cases, 12% at 100, 7% at 150) and tightens to about 5.8% beyond roughly 300 cases. It is a noise-aware threshold, not a guarantee. With 5% random noise per case and one attempt per case, an unchanged system is blocked about 11% of the time at 45 cases, 7% at 100 and 4% at 200, averaged over possible baselines; for one committed baseline the rate can be far higher or lower. A 10-point drop is caught about 67%, 80% and 93% of the time, and a 5-point drop only about 39%, 42% and 51%. Cases tagged critical sit outside this statistical rule: any single pass-to-fail flip blocks, so they must be repeated 5 times.

**What that paragraph leaves out** (section 9 has the numbers and how they were computed). The rates above are rounded: exactly, an unchanged system is blocked 10.6%, 7.1% and 4.0% of the time at 45, 100 and 200 cases, and a 10 point drop is caught 66.6%, 79.7% and 92.7% of the time. The false alarm is not flat: it peaks at 12.3% near 54 cases (15.1% with 10% noise), and between 60 and 149 cases the rule of v0.1 had a lower false alarm and a higher detection at 84 of the 90 sizes at 5% noise (every size except 60 to 62 and 78 to 80 cases; at 10% noise v0.2 is the stricter rule and blocks less often: over those sizes the mean false alarm is 8.3% against 11.5% and the mean detection of a 10 point drop 61.2% against 72.9%, v0.2 against v0.1), so the current rule is one smooth rule, not a better one at every size. The 4% at 200 cases holds only if the target does not crash (a 1% crash rate doubles it). And the committed baseline is one noisy draw: the chance that it leaves more than 15% of later unchanged pull requests blocked is 28.8%, 14.0% and 7.2% at 45, 100 and 200 cases. `evalsprint noise RUN` prices the whole gate (the pass-rate rule and the critical-case rule together) for the repeat protocol of a run, from the flip rate of its repeated cases, and prints "about" figures: it assumes that the noise measured on the repeated cases is the noise of every case, and it needs at least 20 repeated cases (with fewer it prints "not estimated" and no rate, which is what the 10 cases of the `evalsprint init` scaffold give: repeat every case of a suite of 20 or more cases with `run --repeat 3`, or use the tables).

**Two more measured sentences, for a suite with critical cases and for a target that crashes** (exact, at 5% noise per attempt; `scripts/gate_numbers.py`). With 8 critical cases in a 45-case suite, an unchanged change is blocked 36.67% of the time with one attempt per case, 5.79% with every case repeated 3 times, and 1.11% with every case 3 times and the critical ones 5 times (`repeat: {n: 3, per_tag: {safety: 5}}`); repeating every case 5 times alone gives 0.92% and costs more. A target that crashes on 5% of the calls makes an unchanged 100-case change blocked in 54% of the runs (38% of all runs are unhealthy and never compared); one retry inside the run (`target_retries: 1`, the default) brings it to 8%. The same retry also hides a candidate that starts to crash on some calls: with `target_retries: 1` and 100 cases, one whose target crashes on 10% of its calls is blocked in 11% of the runs and one that crashes on 20% of its calls in 39%, against 7% for an unchanged candidate. The gate now blocks a change of `target_retries`, prints the retries of both runs and warns when the candidate retried at least 10 percentage points more of its calls than the baseline, but that warning is weak for small increases (against a baseline that retried 5% of its calls it fires in 11 of 100 comparisons for a candidate that crashes on 10% of its calls and in 89 of 100 for 20%), so the crash and retry counts of the report must still be read next to the gate.

**It is not**

- **Proof that the feature is good.** It says "not worse than the baseline on these cases, as graded by this judge". A regression outside the test set, or one the judge cannot see, passes. Calibrate the judge against human labels (`evalsprint calibrate`) and keep adding cases from real failures.
- **A check that the candidate results are genuine.** The candidate is produced by the pull request's own code (its adapter, its prompts, its custom judges), next to your API keys. Splitting the comparison into its own job protects the gate settings, the baseline, the kit and the comparison from that code; it does not stop a change that *fakes its results* (an adapter that returns canned answers, a judge that always passes, a hand-written `results.jsonl`). Reviewing the pull request is what catches that (section 4).
- **A replacement for code review.** Editing the workflow file, the adapter, the code it calls and the new cases that only look easy are invisible to the checks; `CODEOWNERS` is part of the gate, not an extra.
- **Sensitive on a small test set.** With 45 cases it is a smoke test (an unchanged system is blocked about 1 run in 9 at 5% noise, a real 10 point drop is caught 2 times in 3; section 9). Repeat every case three times (`repeat: {n: 3}`, on by default in the template) until you have about 150 cases, and the critical cases five times (`repeat: {n: 3, per_tag: {safety: 5}}`). Between 60 and 149 cases the gate of v0.1 had a lower false alarm **and** a higher detection of a 10 point drop at 84 of the 90 sizes at 5% noise (at 10% noise v0.2 is the stricter rule and blocks less often); the current rule is one smooth rule without a cliff at 150, not a better one at every size.
- **A prompt-injection defense.** The judge's `injection_suspected` flag is a heuristic hint that a grader-directed instruction may be hiding in an answer. The independent review of evalsprint 0.1 measured a recall of 43% on injection attempts and 27.5% false positives on legitimate text; the heuristic is unchanged in 0.2 and was not measured again. Treat it as a reason to look, never as a filter and never as a guarantee.
- **A measurement of your suite's false alarm, in `evalsprint noise`.** It is a model: it prices the whole gate for the protocol of a run, "about", from at least 20 repeated cases, taking the noise measured on those cases for every case and the cases and attempts as independent (section 9). Hand its figures over with the flip rate and those assumptions.
- **A way to see every crash.** A retry (`target_retries`, 1 by default) hides a transient crash from the pass rate and from the error rate. A changed `target_retries` blocks, and the comment prints the retries of both runs, but the warning for a candidate that retried 10 percentage points more of its calls is weak for small increases: against a baseline that retried 5% of its calls it fires in 11 of 100 comparisons when the candidate's calls crash 10% of the time (89 of 100 at 20%). Read the crash and retry counts (section 9).
- **Coverage of small tags.** A non-critical tag with fewer than `tag_drop_min_cases` (20) paired cases never raises the tag-drop warning, and even at 20 cases five tags give a spurious warning in 3.3% of unchanged comparisons at 5% noise (14.8% at 10%). It is a warning, never a reason.

## 1. Add them to your repository

Install the kit, then let it write the files (nothing that already exists is overwritten without `--force`). The templates ship inside the package, so this works from a plain `pip install`:

```
evalsprint init --ci
```

That creates `evals/` (`eval.yaml`, `rubric.yaml`, `cases.jsonl` from the example set, `target_adapter.py`, `requirements.txt`) and, with `--ci`, the two workflows under `.github/workflows/` and `.github/CODEOWNERS.evals.example`. Copy that last file to `.github/CODEOWNERS` and replace `@your-org/eval-owners` (the same file is `CODEOWNERS.example` next to this README).

```
your-repo/
├── .github/
│   ├── CODEOWNERS
│   └── workflows/
│       ├── evals.yml
│       └── evals-baseline.yml
└── evals/
    ├── eval.yaml            # suite config: dataset, target, judges, repeat, gate
    ├── cases.jsonl          # the test set
    ├── rubric.yaml
    ├── target_adapter.py
    ├── requirements.txt     # the kit (pinned) plus whatever your target imports
    └── baseline/            # committed reference run: summary.json + results.jsonl (+ calibration.json)
```

Edit the lines marked `EDIT` in both workflows (in `evals.yml`: `EVAL_PATHS`, the folders that hold your prompts and LLM code; provider secrets; timeout). `evalsprint init` also adds `runs/` and `.label_secret` to your `.gitignore` (run folders hold your model's answers and the secret of the blind-labeling tokens; existing lines are never duplicated), but never `evals/baseline/`.

The scaffold runs out of the box, offline, so you can see the pipeline work before anything is wired up:

```
evalsprint describe -c evals/eval.yaml                               # checks the set and the judges, no key
JUDGE_PROVIDER=mock evalsprint run -c evals/eval.yaml --label smoke   # canned answers and a placeholder judge
```

The placeholder target (`offline_stub`) and judge (`offline_judge`) live in `evals/target_adapter.py`. They prove that the pipeline runs on your machine and **measure nothing**: never commit a baseline made with them. The default judge stays a real model call on purpose, so a CI run can never be graded by the placeholder by accident. A run is marked `placeholder: true` in its `summary.json` when its target callable is a function named `offline_stub` or when any LLM judge has the provider `mock` (both read after `${VAR}` and `--set` are expanded, so `JUDGE_PROVIDER=mock` does it): the report shows a banner, `evalsprint baseline` refuses to copy it without `--force`, the *Update eval baseline* workflow refuses it (no PR is opened) and `compare` warns when the baseline it is given is one.

**Pin the kit.** `evals/requirements.txt` must name an exact release tag or commit of the kit, so gate behaviour (defaults and statistics) only changes when you upgrade on purpose, in a reviewed PR. `evalsprint init` writes the pin of the version that created it:

```
evalsprint @ git+https://github.com/renanpgonzalez/eval-sprint@v0.2.0
```

The comparison job installs the kit from this file **as it is on the base branch**: a pull request that bumps or swaps the kit changes the candidate run, not the comparison.

## 2. Secrets and repository settings

- **Secrets** (Settings > Secrets and variables > Actions): `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`, whichever your target and judges use. Use a dedicated key with a monthly spend limit: the job that runs the pull request's code holds it (section 4).
- **Dependabot** pull requests come from your repository but GitHub gives their workflows only the secrets stored under Settings > Secrets and variables > **Dependabot**, not the Actions secrets. Add the API key there as well if Dependabot should be evaluated; otherwise the gate treats them like forks (next section). The check names this case: "this pull request was opened by Dependabot".
- **`EVALSPRINT_READ_TOKEN`** (only if the kit lives in a private repository): a fine-grained token with read-only *Contents* access to **that one repository**. Both workflows point `git` at it while `pip install` runs and remove it again afterwards. Without this secret nothing changes. **Who can read it:** in the job that runs the pull request's code the token **is present**, in the environment of the install step and in git's config, while pip installs what the pull request's own `evals/requirements.txt` names. A requirement, a build script or an edit to that file is code from the pull request, so the pull request can read the token. Removing it from git's config afterwards only stops it from outliving the step; the suite itself starts without it. The comparison job installs the base branch's requirements and runs no PR code, so its use of the token is safe from the pull request. So anyone who can open a pull request in your repository can read a token that can read the kit: keep it read-only, limited to the kit repository alone, and rotate it when the people who can open pull requests change.
- **Sensitive kits: vendor the wheel instead.** When the kit must stay private from the people who open pull requests (contractors, forks, a large team), do not give the `evals` job a token at all. Put the wheel of the kit in your repository (`evalsprint-0.2.0-py3-none-any.whl`, a release asset of the kit, or built from the tag with `pip wheel --no-deps --wheel-dir evals/vendor <checkout of the kit>`), commit it under `evals/vendor/`, replace the `evalsprint @ git+...` line of `evals/requirements.txt` with the path of the wheel, and delete `EVALSPRINT_READ_TOKEN` (its `env:` entry and the `git config` lines) from the install step of the `evals` job, and from the `compare` job, which then needs no secret either:
  ```
  ./evals/vendor/evalsprint-0.2.0-py3-none-any.whl
  ```
  The path is relative to the directory pip runs in, not to the requirements file: the workflows run `pip install -r evals/requirements.txt` from the repository root, so do the same locally. Tested offline with this recipe from a clean directory: from the repository root it installs `evalsprint 0.2.0`, from any other directory pip stops with `No such file or directory`. The wheel pins the kit, not its dependencies (PyYAML still comes from the package index); pinning by hash with `--require-hashes` works but then demands a hash for every transitive dependency, so this recipe does not use it. `CODEOWNERS.example` covers `evals/vendor/` through `/evals/`: a new wheel is a reviewed change, like a new pin.
- **Workflow permissions** (Settings > Actions > General): tick *Allow GitHub Actions to create and approve pull requests*, otherwise the baseline workflow cannot open its PR.
- **`BASELINE_PR_TOKEN`** (optional secret): a personal access token or GitHub App token with `contents` and `pull requests` write access. Pull requests opened with the default `GITHUB_TOKEN` do not trigger other workflows, so without it `evals.yml` does not run on the baseline PR, and with `LLM evals gate` required that PR can only be merged with an admin override.

## 3. Make the gate binding (branch protection)

A workflow that nobody is forced to respect is a suggestion. Set up these four things once:

1. **Require the check named `LLM evals gate`**, not `Run evals (PR code)` or `Compare with the baseline (base branch)`. Settings > Branches (or Rulesets) > your default branch > *Require status checks to pass*. The gate job always runs and always decides. If you required one of the other jobs instead, a pull request from a fork would show it as *skipped*, and GitHub treats a skipped required check as passed.
2. **Require code-owner review** and commit [CODEOWNERS.example](CODEOWNERS.example) as `.github/CODEOWNERS`. Section 4 explains why this is what actually stops a pull request from weakening its own gate, editing the workflow, or faking its results.
3. **Decide what a fork or Dependabot pull request means.** Forks never receive secrets, and Dependabot pull requests receive only Dependabot secrets, so the evals cannot run for them unless that key exists. By default the gate job stays green but writes a warning annotation and a job-summary note saying *nothing was verified*. To make that a red check instead, set `FAIL_WITHOUT_SECRETS: "true"` in the workflow's `env:`; a maintainer then runs *LLM evals* manually (Actions > Run workflow) on a branch of your repository that contains the same commit. GitHub attaches checks to commits, so that run normally shows up on the pull request too; confirm it on your first fork PR.
4. **Turn on `REQUIRE_BASELINE` once the first baseline is merged.** It ships as `"false"` so that the very first pull request, the one that installs the evals, is not blocked for lacking a reference. After the first baseline has been merged (section 6), set `REQUIRE_BASELINE: "true"` in `evals.yml`, in a reviewed PR. From then on a base branch without `evals/baseline/` fails the gate ("baseline missing") instead of passing without a comparison. Even while it is `"false"` the gate fails closed when the history of the base branch shows that `evals/baseline/` existed and is gone, so a merged pull request that deletes the baseline cannot switch the gate off for the pull requests after it. The pull request that restores a lost baseline is itself judged by the base branch, so it stays red until a code owner merges it with an admin override.

### Which pull requests run the evals: `EVAL_PATHS`, not a `paths:` filter

The workflow has **no `paths:` filter** on its trigger, on purpose. With `LLM evals gate` required, a pull request that a filter skips never receives the status and waits forever, and branch protection cannot express "required only when these paths change". Instead every pull request starts the workflow, and a cheap first job (`changes`, no API keys, runs no PR code) lists the files the PR changed against its base branch and matches them with the `EVAL_PATHS` list at the top of `evals.yml` (one glob per line: `*` stays inside a folder, `**` crosses folders, `?` is one character, `#` starts a comment, no `!` negation).

- If nothing matches, the paid run is skipped and the `LLM evals gate` job **passes with a notice and a job-summary note** ("no LLM code changed ... passes without a comparison"), so the required check still reports and nothing waits.
- `evals/**` and the two eval workflow files are always relevant, whatever the list says, so a pull request cannot skip the gate by editing the list from a docs-only change. Narrowing `EVAL_PATHS` is itself a workflow change: keep the workflow files under CODEOWNERS (section 4).
- A manual run is always relevant. If the job cannot tell what changed (the checkout is not a merge commit), it runs the evals and warns; if it fails, the gate fails with "path check failed", never green.
- A path missing from `EVAL_PATHS` means a change there is not evaluated. List everything that can change what the LLM feature outputs: prompts, model and parameter configuration, retrieval and tool code, the adapter's dependencies.

## 4. What comes from the base branch, and what does not

The pull request's code runs in **one** job, `evals`. Everything that judges its result lives in other jobs, each on its own runner, with its own checkout of the **base branch** and none of the pull request's files:

| Input | Comes from | Read by job | Why |
|---|---|---|---|
| `gate:` settings | base branch `evals/eval.yaml`, passed as `compare --gate-config` | `compare` | A PR that loosens its own gate is still judged by the old one. |
| Baseline, including `calibration.json` | base branch `evals/baseline/` | `compare` | A PR that swaps or poisons its own baseline is still compared with the reviewed one. |
| The kit that compares | base branch `evals/requirements.txt`, installed in the `compare` job from a clean checkout of the base commit, with no pip cache | `compare` | A PR that bumps, swaps or patches the kit changes the candidate run, not the code that judges it. |
| The candidate run (`results.jsonl`, `summary.json`, `calibration.json` when present, `report.html`) | produced by the PR's code in `evals`, handed over as the artifact `eval-candidate-<pr>` (an explicit list of files, never the folder) | `compare` | It is **data**: the kit parses it, nothing in it is executed. `compare` copies only those files, as regular files and not symlinks, before it reads them. |
| Test set, rubric, judges, adapter, prompts, LLM code | the pull request | `evals` | They are what the PR is allowed to change. Edits are *detected*, not prevented: see below. |
| This workflow file | the PR (GitHub runs the PR's copy) | | It cannot be protected by the workflow itself, only by review. |

Why two jobs. When the suite and the comparison shared one job, code the PR added (the adapter the suite imports, a custom judge, a `conftest.py`) ran in the same workspace, with the same `HOME` and the same installed kit as `evalsprint compare`: it could rewrite the trusted copy of the gate and the baseline, patch the installed `compare`, or make the install pick another kit. `tests/test_templates.py` (class `TamperTests`) runs exactly those three attacks against the previous single-job workflow, where they succeed, and against this one, where they change nothing because the `evals` job holds no trusted file and compares nothing, and `compare` runs on another runner against a kit installed from the base branch.

The kit narrows what a pull request can hide from that comparison:

- baseline cases that disappeared, and cases whose content changed in any field but the id, block the PR (`max_removed_cases` and `max_changed_cases` default to 0);
- a different judge block, rubric text or `pass_rule`, or different source code of a `custom` judge's file or of its `data_files` (`summary.judges_sha`), blocks the PR, with the reason "judge configuration or rubric changed: update the baseline in a reviewed PR" (`allow_judge_changes: false`);
- a different `repeat` protocol (`n` or `per_tag`; the older `tags` spelling counts as the same protocol as its `per_tag` form) blocks it too, because a bigger majority vote is an easier gate (`allow_repeat_change: false`); a different `target_retries` blocks it as well, because a retry hides a transient crash from the pass rate (`allow_target_retries_change: false`; a candidate that retried 10 percentage points more of its calls than the baseline is a warning, `retry_share_warn`); a different version of the kit, patch included, is a warning, or a reason with `require_same_kit_version: true`;
- a critical case that passed in the baseline and errored in the PR blocks it, because it could not be verified.

**What crosses from the PR's job to the trusted ones is treated as hostile text.** The PR's code controls every byte a job output, an artifact and a file in it can hold, so `compare` never uses them as they arrive:

- the exit code of the suite (a job output) must be 1 to 3 digits, else it counts as **255**, a failure: `case "$RUN_CODE" in ''|*[!0-9]*|????*) RUN_CODE=255;; esac`. Only values that passed that check are written to `$GITHUB_OUTPUT`, and the verdict is one of a closed set (pass, regression, configuration error, unhealthy, baseline missing, bootstrap, error), so a newline in the exit code cannot forge a second output line such as `verdict=pass`. The `gate` job checks the exit code it receives the same way;
- only the files on the explicit list (`results.jsonl`, `summary.json`, `calibration.json`, and `report.html` when there is no trusted baseline to take it from) are taken from the artifact, as regular non-symlink files, into a fresh folder;
- text of the PR's run that reaches the job summary or the comment (an unhealthy run's error table) is printed as numbers, never as the run's own strings.

`tests/test_templates.py` (classes `OutputInjectionTests` and `MutationTests`) replays the injection the independent review used, with the real `$GITHUB_OUTPUT` grammar: against the frozen workflow from before the fix (`tests/fixtures/ci/evals_unvalidated_output.yml`) the gate turns green on a real regression, against the shipped one it is blocked, and removing any one of the three checks makes the tests fail. That is an emulation of the runner, not an observation of one; confirm it on the first real pull request of your installation.

**What the checks cannot see, and the split does not change**

- **The results themselves.** The candidate's `results.jsonl` and `summary.json` (including the `judges_sha`, `repeat` and `kit_version` it reports about itself) are written by the PR's code. A change that returns canned good answers, a judge that always passes or a hand-written result file passes the comparison. The `gate.md` footer says so: *"The candidate results themselves come from the pull request's code."* Only review closes this.
- The code of the system under test, a hosted judge model that changes behind the same model name, new cases that only look easy, and the workflow files (GitHub runs the pull request's copy of a workflow, so a PR can delete the split itself). `CODEOWNERS.example` puts `evals/` (test set, judges, rubric, adapter, requirements) and both workflows behind the owners of the gate, so **CODEOWNERS is required, not optional**. Treat the check as evidence for reviewers, not as a replacement for them.
- **A genuine `0` with forged candidate files.** The exit code is validated, the files are a fixed list, but a PR that makes the suite exit 0 after writing a good-looking `results.jsonl` passes the comparison: that is the previous point, and review is what closes it.
- **API keys, and the kit's read token.** The job that runs the PR's code holds the API keys, and it holds `EVALSPRINT_READ_TOKEN` (if you set one) while it installs the pull request's requirements (section 2). Anyone who can open a pull request in your repository can write code that reads them. That is why forks get no secrets, why the API key should be a dedicated one with a spend limit and why a kit that must stay private should be vendored as a wheel.

When the base branch has no baseline **and never had one** (the PR that installs the evals, `REQUIRE_BASELINE: "false"`), there is nothing trusted to compare with: the suite runs, the report is attached, the gate job is green with a warning, and the comment says the gate is not enforced yet. When the base branch has no baseline but the history of `evals/baseline/` shows it existed, or `REQUIRE_BASELINE` is `"true"`, the gate is **red** with "baseline missing on the base branch": the paid run is skipped, and the message says this is not a quality regression. Fetching happens in the checkouts themselves (`fetch-depth: 0` in the job that reads the base branch, `persist-credentials: false` everywhere except the baseline workflow), so no job that runs PR code holds a token it could read back from `.git/config`; the baseline workflow, which pushes a branch, is the only checkout that keeps its credentials.

## 5. What happens on a pull request

| Job | Permissions | What it does |
|---|---|---|
| `check-secrets` | none | Is an API key available? (No for forks, and for Dependabot unless the key is also a Dependabot secret.) |
| `changes` | `contents: read` | Did a file matching `EVAL_PATHS` (or `evals/**`, or the eval workflows) change? No API keys, runs no PR code. |
| `trusted` | `contents: read` | Reads the **base branch**: does `evals/eval.yaml` and `evals/baseline/` exist there (and did they ever)? Records the base commit. No API keys, runs no PR code. |
| `evals` | `contents: read` | Only when LLM code changed, a key is available and the baseline exists. Installs what the PR's `evals/requirements.txt` asks for and runs the suite on the PR. It runs your code next to your API keys, so it cannot write to the PR or the repository, its checkout keeps no token, and it holds nothing trusted and compares nothing. It uploads the files of `runs/candidate` that `compare` reads, by name (artifact `eval-candidate-<pr>`). |
| `compare` | `contents: read` | Checks out the **base commit** `trusted` read (no PR files), installs the kit from that commit's `evals/requirements.txt` (no pip cache), downloads the candidate, runs `evalsprint compare` with the base branch's gate and baseline, builds the report, decides the verdict. No API keys, runs no PR code. |
| `comment` | `pull-requests: write` | Posts or updates the sticky comment. No checkout, no API keys, runs no PR code. |
| `gate` | none | Always runs. Green with a note when no LLM code changed, otherwise turns the verdict of `compare` into a green or red check. Red when any earlier job it depends on did not finish (the base-branch read, the key check, the path check, the comparison). This is the one to require. |

The verdict comes from `evalsprint`'s exit code, and each one has its own message:

| Exit code | Where | Verdict | PR comment heading | `LLM evals gate` |
|---|---|---|---|---|
| 0 | `compare` | pass | *No regression* | green |
| 1 | `compare` | regression | *Regression detected*, with the reasons | red, "candidate is worse than the baseline" |
| 2 | `run` or `compare` | configuration error | *Eval gate: configuration error* | red, "a setup problem, not a quality regression" |
| 3 | `run` | unhealthy run | *Eval infrastructure error, not a quality regression*, with the error counts | red, "errors from the target or the judge, not a quality regression; re-run when the provider has recovered" |
| none (the run is skipped) | base-branch read | baseline missing | *Eval gate: baseline missing on the base branch* | red, "not a quality regression: restore the baseline" |
| 130 or 143 | `run` | interrupted (Ctrl-C) or terminated (`SIGTERM`, for example from a `timeout` wrapped around the command) after saving the attempts that were running; a job that GitHub cancels never gets this far, and the gate is red because that job did not finish | *Eval job did not finish* (it reaches the gate as an unexpected exit code) | red, "not a quality regression"; re-run the job: CI does not resume, every job starts on a clean runner |
| anything else, a step that failed before the verdict, or a candidate that was never uploaded | | error | *Eval job did not finish* | red, "not a quality regression" |

Exit 3 means more than `max_error_rate` (default 5%) of the cases, or of all attempts when cases are repeated, ended in an error from the target or from the judge. Such a run is never compared with the baseline: errors are not a quality signal, and comparing would print a misleading delta. The files of the run are still written and the report is still built, so you can see what broke.

The last steps of `compare` always run, so the job summary, the artifacts and the PR comment exist even when the gate fails. Every artifact is built from an **explicit list of files**, never a whole folder, so nothing the run happens to leave next to its results (a scratch dump, a stray credential file) is published:

| Artifact | Files | Retention |
|---|---|---|
| `eval-candidate-<pr>` | `results.jsonl`, `summary.json`, `calibration.json` (when present), `report.html`: the run the PR's code produced, all that `compare` reads from that job | `ARTIFACT_RETENTION_DAYS` (14) |
| `eval-report-<pr>` | the same four files of the candidate, `gate.md`, `gate.json` | `ARTIFACT_RETENTION_DAYS` (14) |
| `eval-gate-<pr>` | `gate.md`: the comment text that `comment` posts | 1 day |

`ARTIFACT_RETENTION_DAYS` is one line in the `env:` of `evals.yml` (`"14"` by default; GitHub accepts 1 to 90, within the limit your repository or organization sets). `results.jsonl` holds your model's answers and the judges' reasons, so shorten it when the test set holds personal data (section 11).

### The sticky comment

The comment starts with `<!-- evalsprint-gate -->`. Each run looks through the PR's comments (`gh api`) for one that is written by `COMMENT_AUTHOR` (default `github-actions[bot]`) **and** starts with that marker, and edits it in place; otherwise it creates one. Comments from people, from other bots and from other workflows are never touched, even if they mention the marker. If you post with a personal or App token instead of `GITHUB_TOKEN`, set `COMMENT_AUTHOR` in the workflow. Comments longer than 60,000 characters are truncated (the job summary keeps the full text). Pull requests from forks have a read-only token: the step warns instead of failing, and the result is in the job summary. **What the comment prints from the run's files** (case ids, tags, run ids, kit versions, `judges_sha`, the judge's name and its calibration decision, all written by the pull request's code) is printed in code spans, with URLs, `@mentions`, issue references and HTML entities made inert, so a hostile tag name cannot notify anyone or plant a link; `gate.json` keeps the plain text. When either run retried a crashed target call, or `target_retries` differs, the comment adds one line with the configured value, the retries and their share of the calls for both runs.

## 6. Create the first baseline

Locally, with your API key exported and your real feature wired into `evals/target_adapter.py` (a run with `offline_stub` or `JUDGE_PROVIDER=mock` measures nothing):

```
evalsprint run -c evals/eval.yaml --label baseline --out runs/first
evalsprint baseline runs/first --to evals/baseline
git add evals/baseline && git commit -m "evals: first baseline"
```

If the run is interrupted (an outage, a closed laptop, a rate limit, Ctrl-C, or a `SIGTERM` from the machine), continue it instead of paying for the finished cases again: `evalsprint run -c evals/eval.yaml --resume runs/first` reuses every record whose case, judges, target and repeat protocol are unchanged. An interrupted run drains the attempts that were running, keeps what it has, prints the exact `--resume` command (it prints it when the final save fails, too) and exits **130** on Ctrl-C and **143** on `SIGTERM`. **What counts as "the target is unchanged".** Its callable, its `params` and the source file the callable lives in (`target.source_sha`, a short hash). **Edits in the modules that file imports are not detected**: if you changed those, do not resume, start a fresh run. When something changed, nothing is reused, and `summary.resumed_from.fingerprint_mismatch` says what: `judges_sha`, `target` (callable or params), `target_source` (the adapter file) or `repeat`. Add `--retry-errors` to run again the records that ended in a provider error and the crashed attempts inside repeated cases that otherwise finished (`resumed_from.retried_errors` counts the attempts that run again). A resume takes a lock in the run folder, the file `.resume.lock`, which exists only while a resume is running: a second `--resume` of the same folder at the same time stops with exit 2, and a `.resume.lock` left by a `kill -9` is harmless. **Transient target crashes are retried inside the run** (`target_retries`: default 1, at most 5, `0` switches it off). Every retry is a paid call, counted in the cost and in `n_target_retries`, so it is not free; and **a target that is not idempotent (it sends a message, writes a record, charges something) must have `target_retries: 0`**, or one crash can make the same call twice.

Or run the *Update eval baseline* workflow from your default branch and merge the PR it opens.

**Once that first baseline is merged, set `REQUIRE_BASELINE: "true"` in `evals.yml`** (section 3, item 4). Until you do, the only thing that stops a later loss of the baseline from going unnoticed is the history check.

Before committing, open `runs/first/summary.json` and check `n_errors` is 0 (or tiny), `n_scored` is close to `n_cases`, and the pass rate is believable. Everything is compared against this run. `evalsprint baseline` refuses a run with too many errors (per case, or per attempt when cases are repeated) and a placeholder run (`"placeholder": true`: the scaffold's offline target, or a judge on the mock provider) unless you pass `--force`; do not.

The baseline is **one noisy draw**, and every later pull request is compared with that draw (section 9 puts a number on what that costs). Make it with the same `repeat` protocol the pull requests will use: the majority of three attempts makes the draw far less noisy than one attempt, and a pull request whose protocol differs from the baseline's is blocked anyway.

If the judge is calibrated, keep the result next to the baseline so the gate can warn when the judge is not trustworthy:

```
evalsprint calibrate runs/first --labels labels.jsonl --json evals/baseline/calibration.json
```

`compare` loads `evals/baseline/calibration.json` automatically. Without it, the PR comment says *LLM judge not calibrated* on every run. With `require_calibration: true` in the gate, that warning becomes a regression, and so does a calibration whose decision is not `usable` or `trustworthy` or whose weighted kappa lower bound is below `min_judge_kappa`. The PR comment shows the decision next to the kappa (a `calibration.json` written by an older evalsprint is read as unweighted and marked legacy: re-run `evalsprint calibrate`). A calibration records the `judges_sha` it was made for: when the judges changed afterwards, the report and the comparison warn that it describes a different judge configuration, and the answer is to label again, not to ignore the warning.

**How many labels.** It depends on how often the humans pass the output, because the decision needs enough human *fails* to judge the judge. With a judge that agrees with the humans on every case (the best case, so read it as a floor) and every case labeled, the first number of cases from which it is `trustworthy` is 40, 48, 67 and 124 when the humans pass 50%, 70%, 80% and 90% of the outputs (the number of human fails in a run is random and fewer fails need more labels, so each figure is the smallest number for which at least 80% of the runs get there). Labeling a stratified sample of a larger run takes 40 to 130 labels over the same range, and a real judge with errors takes more: plan **about 150 labels**, and more when the humans pass 90% or more. A judge that is right 95% of the time on both kinds of verdict is not reliably `trustworthy` within 150 labels of a 200-case run at 70%, 80% or 90% human pass rates (the full table is in the sprint playbook, rendered from `calibration.labels_budget_table()`). `tnr_reliable`, the true-negative-rate line of the calibration, is a **screening flag**, not a guarantee: it says that rate was measured narrowly and its estimate is at least 0.75, which a judge whose true rate is 70% still passes in 2% to 28% of simulated runs, depending on the design. `evalsprint calibrate` accepts several label files (`--labels a.jsonl --labels b.jsonl`), so a second labeling round adds to the first instead of replacing it, and `evalsprint calibrate --export-fewshot` turns the clearest disagreements into rubric examples.

**After a change to the judge or the prompt**, do not label everything again: export only the cases whose current output has no label yet (`evalsprint label RUN --previous-labels old-labels.jsonl`), label those, and calibrate with both files. Only the changed outputs need a label when the earlier labels covered every unchanged case (a full earlier pass); after a sampled pass the export also draws unchanged cases that never had a label (labeling only the changed outputs after a sampled pass biases the true-negative rate).

## 7. The gate settings

The `gate:` block of `eval.yaml` (these are the defaults; every key is optional):

```yaml
gate:
  max_pass_rate_drop: auto        # auto: ONE rule for every test-set size. With n paired cases it trips when the net loss
                                  #       (pass -> fail minus fail -> pass) is at least max(3, ceil(n / 50)) cases AND the
                                  #       one-sided exact McNemar p < alpha(n) = 0.0575 + 0.1925 / (1 + (n / 85)^4.5)
                                  # a number (0.03 = 3 p.p.) is a fixed limit; null switches the rule off
  alpha: 0.05                     # one-sided; only used with a NUMBER above plus require_significance: true
  require_significance: false     # only used when max_pass_rate_drop is a number
  min_pass_rate: null             # absolute floor on the candidate run, e.g. 0.80
  critical_tags: []               # any pass -> fail in these tags blocks the PR; repeat those cases 5 times with repeat.per_tag (section 9)
  max_cost_increase_pct: null     # mean cost per request
  max_p95_latency_increase_pct: null
  max_error_rate: 0.05            # above this (per case or per attempt) the run is unhealthy (exit 3), never compared
  max_removed_cases: 0            # baseline case ids missing from the candidate
  max_changed_cases: 0            # same id, different content
  min_judge_kappa: 0.6            # compared with the LOWER bound of the calibration's weighted kappa interval
  require_calibration: false      # true: an uncalibrated, unreleased or low-kappa LLM judge is a regression, not a warning
  allow_judge_changes: false      # true: a changed judge block, rubric, pass_rule, custom judge source or data_files is a warning, not a regression
  allow_repeat_change: false      # true: a changed `repeat` protocol (n or per_tag) is a warning, not a regression
  allow_target_retries_change: false # true: a changed `target_retries` is a warning, not a regression
  retry_share_warn: 0.10          # warn when the candidate retried this much more of its target calls (a share, 0 to 1) than the baseline (null: never)
  require_same_kit_version: false # true: a different evalsprint version (patch included) is a regression reason (else a warning)
  tag_drop_warning_pp: 20         # warn when a non-critical tag's pass rate drops by this many points (null: never)
  tag_drop_min_cases: 20          # ... and the tag has at least this many paired cases (10 was too few: a no-change comparison of 10 tags of 10 cases raised a spurious warning 43% of the time)
```

**`retry_share_warn` is a weak warning.** The share of retried calls is the retries over the target calls of the run; the warning needs a candidate share at least this much above the baseline's. At 100 calls and `target_retries: 1`, against a baseline whose calls crash 5% of the time, it fires in 0.12% of the comparisons of an unchanged target, 11.2% when the candidate's calls crash 10% of the time, 88.8% at 20% and always at 50% (against a baseline that never retried: 55%, more than 99% and 100%); a limit of 0.075 would warn on an unchanged target 0.77% of the time and on 10% crashing calls 24.7% of the time, and a limit of 0.15 would almost never warn on an unchanged target but would warn on 10% crashing calls only 0.61% of the time. Raise `target_retries` only in a reviewed pull request: a changed value blocks (`allow_target_retries_change: false`).

`max_pass_rate_drop: null` switches the pass-rate rule off (the report says "gate mode: off"); every other rule still applies. The comparison reports the rule it applied (`gate_mode`: `auto`, `fixed` or `off`, plus `effective_min_net_loss`, `effective_alpha` and `effective_max_pass_rate_drop`). The gate given to `compare` is authoritative: the CI passes the base branch's copy with `--gate-config`, and it is never mixed with the gate stored in the run under test.

Because the gate is read from the base branch, **a PR that changes `gate:` is judged by the old settings, and its new settings apply to the pull requests after it.** That is the point. To tighten or relax the gate, change `evals/eval.yaml` in a reviewed PR and merge it first.

## 8. Accepting an intended change, and updating the baseline

Sometimes a change is meant to cost quality: a cheaper model, a shorter prompt. The gate will (correctly) block it. The template deliberately has no label or comment that overrides the gate, because whoever writes the PR could use it on themselves. Instead:

1. A code owner reviews the PR, reads the comparison and decides. The merge is then a decision by a person with the authority to override the required check (an admin merge, or a ruleset bypass), not a side effect of editing a file.
2. After merging, run *Update eval baseline* from the default branch. It runs the whole suite, copies the run to `evals/baseline` (keeping the committed `calibration.json`), and opens a PR whose description lists the pass rate against the previous baseline, the number of target and judge errors, the error rate per attempt when cases were repeated (a crashed attempt can hide behind a majority that passed), the calibration status and the flip rate when cases were repeated.
3. Review that PR and merge it. Later pull requests are compared with the new reference.

**Changing the test set, the judges or the repeat protocol** goes through the same door. Editing or deleting cases, editing the rubric, changing the judge model, `pass_rule`, the source (or a `data_files` entry) of a custom judge or `repeat` makes the gate fail with "update the baseline in a reviewed PR", on purpose: pass rates before and after such a change are not comparable, and it is also the cheapest way to hide a regression. After the admin merge, *Update eval baseline* lists in its pull request whether the judges changed (`judges_sha`, old to new) and how many cases were added, removed or edited. If the judges changed, label again (only the cases whose output changed: section 6) and re-run `evalsprint calibrate`: the `calibration.json` that was carried over describes the old judge. `evalsprint history runs/ -o history.html` draws the pass rate and cost per run and marks the runs where `judges_sha` changed, because the numbers on the two sides are not comparable.

While you iterate on a judge over several pull requests you can instead merge, in a first reviewed PR, `allow_judge_changes: true` in `evals/eval.yaml` (it applies from the next PR, because the gate is read from the base branch), so that judge changes become a warning. Set it back to `false` in the PR that refreshes the baseline. Do not leave it on: it removes the protection for every pull request in the meantime. The same goes for `allow_repeat_change`.

Do not regenerate the baseline on a PR branch to turn its own check green; the workflow reads the baseline from the base branch, so it would not work, and it would be the wrong thing to try. Cases added to `cases.jsonl` are reported as *candidate only* and are not gated until the baseline includes them.

`evals-baseline.yml` refuses to propose anything when the suite run is unhealthy (exit 3) or is a placeholder (the target is a function named `offline_stub`, or an LLM judge has the provider `mock`: the job fails with "Baseline refused" and opens no PR), never passes `--force`, and must be started from the default branch.

## 9. What the gate can and cannot catch

A gate that cries wolf gets disabled, and one that never fires is theater. These are the measured properties of the default gate (`max_pass_rate_drop: auto`). They come from an exact calculation, not a benchmark: baseline pass rate 80%, and a per-case verdict that flips with the stated probability in each run, independently, even for an identical system (the model that fits "committed baseline vs a fresh run of the same code"). The noise level is an assumption: measure yours with `evalsprint run --repeat 3` (the summary reports `flip_rate`), then run `evalsprint noise RUN` on that run: it prices the whole gate (the pass-rate rule, the critical-case rule and the two together) for the repeat protocol that run really used and prints the false alarm, the detection and the baseline-lottery tail below as "about" figures for YOUR number of cases and YOUR measured noise (it says so and asks for `run --repeat 3` when nothing was repeated). **How `noise` reads your run:** it turns the flip rate (the share of repeated cases whose attempts disagree) into the noise of one attempt, gives each case the noise of the majority of its own attempts (the `per_tag` cases with 5, the others with 3; a case that ran once with the noise of one attempt) and prices the gate of the run (the one stored in it, or the file that `--gate PATH` or `-c eval.yaml` names; the output says which one and which tags are critical) with the pass rate of that run. **It assumes that the noise measured on the repeated cases is the noise of every case.** **It needs at least 20 repeated cases**: with fewer it prints `not estimated`, the fix (`run --repeat 3`, and `per_tag` for the critical cases) and no rate, because one flaky case moves the flip rate by 5 points or more. The 10 cases of the `evalsprint init` scaffold are in that situation, so measure the noise on a suite of 20 or more cases with every case repeated, not with the critical tag alone. When no repeated case flipped, the figures are priced at the upper 95% bound of the noise and say so. An even number of attempts counts a tie as a fail: `noise` prices that exactly, but 3 or 5 attempts are the better choice. Reproduce the numbers of this section with `python scripts/gate_numbers.py` (standard library only; the kit repository ships it, the client repository does not). The detection figures are an exact calculation of a simplified model (a true drop hits a random share of the cases); a model in which the drop hits an exact number of cases gives figures from 2.4 points lower to 7.9 points higher on the sizes that were compared (30 to 300 cases), so read them as "about".

**The rule.** One rule for every size, with no switch and no cliff: it trips when the net loss *b - c* (cases that went pass to fail, minus fail to pass) is at least *k(n)* = max(3, ceil(n / 50)) and the one-sided exact McNemar p-value is below *alpha(n)* = 0.0575 + 0.1925 / (1 + (n / 85)^4.5). Small suites get a generous alpha (they have little data), large ones a strict alpha and a floor of 2 points:

| Paired cases | Net loss needed | alpha | Fewest clean losses (c = 0) that trip |
|---|---:|---:|---:|
| n = 45 | 3 | 0.2396 | 3 (6.7 p.p.) |
| n = 100 | 3 | 0.1200 | 4 (4.0 p.p.) |
| n = 150 | 3 | 0.0714 | 4 (2.7 p.p.) |
| n = 200 | 4 | 0.0615 | 5 (2.5 p.p.) |
| n = 300 | 6 | 0.0582 | 6 (2.0 p.p.) |
| n = 500 | 10 | 0.0576 | 10 (2.0 p.p.) |

False alarms (a PR that changes nothing gets blocked), by test-set size and per-case noise:

| Paired cases | 2% noise | 5% noise | 10% noise |
|---:|---:|---:|---:|
| 45 | 3.0% | 10.6% | 14.7% |
| 100 | 3.4% | 7.1% | 8.3% |
| 200 | 2.4% | 4.0% | 4.5% |
| 400 | 2.7% | 4.1% | 4.6% |

Detection (a true drop gets blocked), at 5% noise:

| Paired cases | 5 p.p. drop | 10 p.p. drop | 15 p.p. drop |
|---:|---:|---:|---:|
| 45 | 39% | 67% | 85% |
| 100 | 42% | 80% | 96% |
| 200 | 51% | 93% | 99.6% |
| 400 | 78% | 99.7% | 100% |

What to conclude:

- **A 45-case set is a smoke test.** An unchanged system is still blocked about 1 time in 9 at 5% noise (10.6%), a real 10 p.p. drop is caught 2 times in 3 and a 5 p.p. drop fewer than 4 times in 10. It cannot do much better with one attempt per case, and the false-alarm curve is not flat: at 5% noise it peaks at 12.3% near n = 54 and stays above 11% for n = 47 to 54 and 62 to 72. The previous rule peaked at 21.0% (n = 39). At 10% noise the peak is 15.1% (also at n = 54). The PR comment says so when fewer than 100 cases are paired.
- **Repeat every case three times below about 150 cases** (`repeat: {n: 3}`, the default in the template). The verdict of a case is the majority of its attempts, which removes most of the judge and target noise: at 5% noise per attempt and 45 cases the false-alarm rate falls from 10.6% to 0.3% and the detection of a 10 p.p. drop rises from 66.6% to 81.6%. The price is 3 times the target and judge calls for those cases. At n = 30 a repeat still cuts false alarms (6.5% to 0.1%) but does not help the detection of a small drop (5 p.p.: 27% to 21%): below about 40 cases the test set is too small for any gate. From 150 cases on, one attempt per case keeps false alarms at or under 5% **if the target does not crash** (see the crash bullet below), and then you can delete the `repeat:` block and pay 1x.

  | Paired cases | False alarm, 1 attempt to 3 | Detect 10 p.p., 1 to 3 | Detect 5 p.p., 1 to 3 |
  |---|---:|---:|---:|
  | n = 30 | 6.5% to 0.1% | 51.7% to 57.9% | 27.1% to 20.8% |
  | n = 45 | 10.6% to 0.3% | 66.6% to 81.6% | 39.1% to 40.3% |
  | n = 60 | 10.9% to 0.6% | 73.1% to 91.6% | 41.6% to 56.5% |
  | n = 100 | 7.1% to 0.3% | 79.7% to 97.5% | 42.4% to 68.7% |
  | n = 149 | 4.2% to 0.8% | 85.5% to 99.3% | 43.5% to 79.7% |

  This assumes the attempts are independent. Noise that repeats on every attempt of a case (a judge that always gets one answer wrong) is not reduced by repeating it, so these figures are the best case. The repeat protocol is part of the baseline: a pull request that changes it is blocked (`allow_repeat_change`).
- **100 to 200 cases** is the practical range for a sprint: at 5% noise false alarms are 7.1% at 100 cases and 4.0% at 200 with one attempt, and a 10 p.p. drop is caught 80% to 93% of the time. A 5 p.p. drop is still caught about half the time. Reliable 3 to 5 p.p. sensitivity needs several hundred paired cases, or repeats. Between 60 and 149 cases the new rule is a different trade from the previous one, not a strict improvement: the rule of v0.1 had a lower false alarm **and** a higher detection of a 10 p.p. drop at 84 of the 90 sizes at 5% noise (every size but 60 to 62 and 78 to 80). Averaged over those sizes at 5% noise the previous rule was slightly better on both axes (false alarms 5.1% against 7.0%, detection of a 10 p.p. drop 83.4% against 79.7%); at 10% noise the new rule is the stricter one (false alarms 8.3% against 11.5%, detection 61.2% against 72.9%). That is part of the price of removing the step at 150 cases; repeating the cases (above) is the way out.
- **From 150 cases on**, false alarms are at most 4.3% at 5% noise and 4.9% at 10% noise for every size up to 500, and the floor of the net loss grows by one case every 50 cases. Adding cases never costs much detection: between 30 and 500 cases the detection of a 10 p.p. drop at 5% noise never falls more than 2.6 p.p. below its best value at a smaller size (2.2 p.p. at 10% noise). A small sawtooth remains for a 5 p.p. drop (up to 5.4 p.p. at 5% noise, 8.8 p.p. at 10%). **"At most 5% from 150 cases on" assumes the target does not crash.** A target crash on a case that passed in the baseline counts as a failure, and a crash is not noise the majority vote removes from a single attempt: at 5% noise a target that crashes on 1% of the cases raises the false alarm from 10.6% to 14.7% at 45 cases, from 7.1% to 11.1% at 100 and from 4.0% to **8.0%** at 200, and on 5% of the cases to 34.6%, 36.2% and 42.4% (more than 5% is `UNHEALTHY` and never compared). Fix the crash in the adapter, let `target_retries` absorb the transient ones, and read the crash warnings of the comment.
- **The committed baseline is a lottery ticket.** The figures above average over the baselines you could have drawn: the baseline is ONE noisy run, and every later pull request is compared with that one. Given the baseline the false-alarm rate of later pull requests is exact, and it varies a lot: at 5% noise and a true pass rate of 80%, the chance that the baseline you committed makes more than 15% of the unchanged pull requests fail is **28.8% at 45 cases, 14.0% at 100 and 7.2% at 200** (median baseline: 7.0%, 1.3% and 0.6%; the 95th-percentile baseline: 42.9%, 31.3% and 22.6%). So an unchanged pull request that keeps being blocked is a reason to look at the baseline and at `evalsprint noise` (from 20 repeated cases), not only at the pull request. Re-drawing the baseline until the pull requests pass is not a fix (it is the same cheapest way to hide a regression that section 8 describes): build the baseline with the `repeat` protocol of the pull requests, which shrinks the noise of the draw (with a majority of three attempts the chance of more than 15% falls to under 0.15% at all three sizes), and review it like code.
- **Critical tags are not statistical.** One flipped case blocks the PR, so with 8 critical cases and 5% noise about 32% of unchanged PRs would be blocked on the critical rule alone with one attempt (36.67% for the whole gate at 45 cases). Repeat the critical cases, and only those: `repeat: {n: 3, per_tag: {safety: 5}}` in `eval.yaml` runs every case 3 times and the cases tagged `safety` (use the tag in your `critical_tags`) 5 times; a case takes the largest count among its tags. A majority of five attempts cuts the critical false-alarm rate from 32% to **0.92%** (5.6% with a majority of three) while a critical case that really broke, failing 80% of its attempts, is still caught **94%** of the time (unanimity of three was rejected: 0.10% false alarms, 51% detection). The total false alarm of the whole gate (pass-rate rule or critical rule, exact over their joint distribution) at 45 cases with 8 critical ones and 5% noise is **1.11%** with `{n: 3, per_tag: {safety: 5}}` against **5.79%** with `repeat: {n: 3}` on every case (the critical rule alone is 5.6% there); `n: 5` on every case is 0.92% but costs five times the whole suite, the critical cases at 5 with the rest at 1 is 9.37%, and no repeats at all is 36.67%. `compare` warns when critical cases ran fewer than five times and recommends the `per_tag` line. The old `{n: 5, tags: [safety]}` is read as `{n: 1, per_tag: {safety: 5}}`. `run --repeat N` changes `n` and keeps `per_tag`.
- **Judge noise** is the main source of noise. Put `samples: 3` on every LLM judge that gates a merge (the per-criterion majority, about 3 times that judge's cost) and leave `temperature` at the provider default; at temperature 0 the three samples are nearly identical and the vote adds little. Calibrate the judge before trusting it (`evalsprint calibrate`, about 150 labels).
- **Target noise.** Set temperature 0 or a seed on the system under test where you can. If re-running the job flips the verdict, your limit is tighter than your noise. A target crash in one attempt of a repeated case is a warning naming the case even when the majority passed (a flaky crash): retry provider errors inside your adapter.
- **Warnings** in the PR comment (a judge that is not calibrated, critical tags that were not repeated, no cost data, a baseline cost of zero, a critical tag that matches no case, a small test set, a non-critical tag whose pass rate fell by 20 points on at least 20 cases, a different kit version, a calibration made for another judge configuration, a baseline that is a placeholder run) are setup problems or slices, not noise. Read them.

## 10. What each CI run costs

A run costs the target's spend plus the judges' spend, both recorded in `runs/candidate/summary.json` (also in the uploaded artifact `eval-candidate-<pr>`):

```
python -c "import json; c = json.load(open('runs/candidate/summary.json'))['cost']; print(c['system_total_usd'] + c['judge_total_usd'])"
```

Monthly cost is that number times the number of runs, roughly the PRs that touch LLM paths times their pushes. As an illustration, 150 cases at $0.004 system + $0.006 judge is about $1.50 per run, so 40 runs a month is about $60. `repeat` and `samples` multiply it for the cases and judges they apply to: `repeat: {n: 3}` on 45 cases makes 135 attempts (with `per_tag: {safety: 5}` and 8 safety cases: 37 x 3 + 8 x 5 = 151). `EVAL_PATHS` (pull requests that touch no LLM code cost nothing) and `cancel-in-progress` (a new push supersedes the run in flight) already avoid most waste. `system_total_usd` only counts cases whose model has a price (`cost.n_priced` says how many), so check that number when a cost looks too low.

Ways to spend less: run only a smoke subset on PRs (`--tags smoke` in the run step) and the full suite nightly (but the gate's power depends on the number of paired cases, so a smaller PR subset is a less sensitive gate); put cheap heuristic judges (`contains`, `regex`, `json_valid`) first; move to a cheaper judge model only after calibrating it against human labels. A CI run starts from nothing on a fresh runner, so `--resume` does not apply to it: the `timeout-minutes` of the `evals` job is the cap on a runaway run.

## 11. Data handling

The eval run holds real inputs and outputs. The artifacts `eval-candidate-<pr>` and `eval-report-<pr>` (`ARTIFACT_RETENTION_DAYS`, 14 by default) contain `results.jsonl` and the HTML report, which show every case's input and the system's answer (up to 3,000 characters each); the committed `evals/baseline/results.jsonl` holds the same for the baseline, in git history for good; the PR comment names case ids and reasons but is written from the same run. When the test set was built from production traffic, all of that is production-derived data, and everyone who can read the repository or its workflow runs can read it (for a public repository, everyone).

Before the first run, agree with the client, in writing:

- whether production-derived cases may be committed at all, and how they are scrubbed first (names, emails, account numbers and identifiers, free text). Scrub the test set, not just the baseline: the baseline copies the inputs;
- how long artifacts are kept (`ARTIFACT_RETENTION_DAYS` in the `env:` of `evals.yml` sets `retention-days` of both run artifacts: `eval-candidate-<pr>` only has to outlive the workflow run, so 1 day is enough; `eval-gate-<pr>` already keeps 1 day). Each artifact is an explicit list of files, so for sensitive outputs drop `results.jsonl` and `report.html` from the `path:` list of the `eval-report-<pr>` upload (keep `gate.md` and `gate.json`). Also agree who can read the repository and its Actions runs, and whether the model providers may receive those inputs;
- who deletes what at the end of the engagement. A committed baseline cannot be removed from history by a later commit.

The kit already scrubs local file paths from recorded errors and tracebacks; it does not scrub personal data. The per-run secret of the blind-labeling tokens (`.label_secret` in the run folder) stays local: it is created with permissions 0600 and must be in `.gitignore`, as above.

## 12. Good to know

- **Provider outages** show up as exit 3 (*infrastructure error, not a quality regression*), for the judge as well as for the target, and never as a drop in the pass rate. Re-run the job when the provider recovers.
- **A hung adapter** holds the job until `timeout-minutes`. The kit has no per-case timeout for the target yet; set one inside your adapter.
- **A very long answer** is not sent to the judge: above `max_output_chars` (default 20,000 characters) the case fails with "output exceeds max_output_chars" at zero judge cost, and every other untrusted section is cut at `max_section_chars`.
- **The workflows run on Linux.** Nothing here needs Windows; see [docs/WINDOWS.md](../../docs/WINDOWS.md) for running the kit on a Windows laptop.
- **What was tested.** The workflows were executed step by step by an emulator of GitHub Actions (every job on its own workspace, home directory, installed kit and artifact store, with real bash and git, and stub `evalsprint`, `pip` and `gh` programs; the real `evalsprint` in the end-to-end and tamper tests) in `tests/test_templates.py`. That covers the job graph, a PR that edits its own gate and baseline, three attempts to tamper with the comparison from the PR's code (blocked here, successful against the old single-job workflow kept as a fixture), a PR that forges its own results (passes: the documented limit), a PR that injects a newline into its exit code to write a second job output (the `RUN_CODE` injection of the independent review: the frozen pre-fix workflow turns green on a real regression, this one is blocked, and taking out any one of its three checks makes the tests fail), the explicit artifact file lists, every exit code, the sticky comment, the fork and Dependabot paths, the `EVAL_PATHS` check (pull requests are checked out as merge commits built locally), a candidate that was never uploaded, a failed kit install, a base branch that lost its baseline, and the rule that only the baseline workflow keeps its credentials. The numbers of section 9 are recomputed from first principles by the test suite (`DocsNumbersTests`), so they cannot drift from the code. The workflows have **not** been run on a GitHub-hosted runner, `gh` itself was not run, and the real behavior of `upload-artifact`, `download-artifact` and job isolation is emulated, not observed: check the first real pull request, and on it that `changes` sees the right files, that `compare` downloads the candidate, and that a documentation-only pull request gets a green `LLM evals gate`.
