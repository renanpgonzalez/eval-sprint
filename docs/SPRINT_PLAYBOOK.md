# Eval Sprint playbook

How a 2-3 week Eval Sprint runs, what you need to provide, and what you get at the end.

This document is written to be shared with the client. It is the plan both sides agree to at kickoff.

## What an Eval Sprint is

You shipped a feature that calls an LLM. Today, "is it good?" is answered by spot-checking, and a prompt change or model upgrade is a leap of faith. An Eval Sprint replaces that with measurement:

| # | Deliverable | In one sentence |
|---|---|---|
| 1 | **Test set** | About 150 real and adversarial cases (100 at the least), tagged, that represent how your users actually use the feature. |
| 2 | **Calibrated LLM judge** | An automated grader whose agreement with your experts has been measured, not assumed. |
| 3 | **Quality and cost report** | Pass rate with confidence intervals, failures by category, real cost per request, latency, judge reliability. |
| 4 | **CI regression gate** | A pull-request check that fails when quality, cost or latency gets worse, or when a critical case breaks, and that a pull request cannot loosen for itself. |

Everything runs on the open, dependency-light Eval Sprint kit (Python, plain files, no hosted service), and it stays in your repository afterwards. You own it.

**What a sprint does not do.** It measures; it does not improve the feature. It surfaces the failure patterns and gives you a fast way to check every fix, but changing prompts, retrieval or models is your team's work (or a separate engagement). It also does not certify the feature as "accurate": it tells you how often it passes *your* criteria, with error bars, and how far the automated grader can be trusted. And the CI gate is a statistical instrument, not an oracle: an unchanged pull request is occasionally blocked and a small regression can slip through. We tell you how often, for the size of your test set (section 4), instead of promising otherwise. It compares the pull request's results with a reference your reviewers approved, in a job that never runs the pull request's code; it cannot tell whether those results were produced honestly, which is what code review is for (section 4, deliverable 4).

**Duration and price.** Two weeks for one feature with an accessible endpoint and usable logs, three weeks when data access, labeling or CI setup take longer. The price is fixed and agreed before kickoff, depending on the number of features, the state of your data and whether the CI gate is in scope. Optional monthly maintenance is described at the end.

## 1. Kickoff: questions we ask

Answer these in writing before the kickoff call if you can. Short answers are fine; "we don't know" is a useful answer.

### The feature

1. What does the feature do, in two sentences? Who uses it (customers, internal staff) and how often?
2. Which model(s), prompt(s) and retrieval or tools sit behind it? Is there one code path in production or several (per plan, per language, per tenant)?
3. Can we call it in a staging environment with realistic data? What does one call cost and how many calls per minute can it take? Is it safe to call twice with the same input (does it send a message, write a record, charge something)? The kit calls a target again once when it crashes (`target_retries: 1`) and that is a second paid call; if the feature is not idempotent we set `target_retries: 0`.
4. Is the output free text, structured (JSON), or an action taken by an agent?

### What "good" means

5. What is the worst thing this feature could do? (Wrong price, invented policy, leaked data, harmful advice, an unauthorized action.) These become critical cases.
6. What is a *bad but tolerable* answer? (Too long, off-tone, missing a link.) These become tracked, non-blocking criteria.
7. Which failures have you already seen in production? Support tickets, screenshots and Slack threads are gold: each one becomes a permanent regression case.
8. Does the feature ever need to say "I don't know" or refuse? In which situations?

### Data

9. Do you log inputs and outputs? For how long, and where? Can we get a sample of the last 30-90 days, anonymised?
10. Is there personal or regulated data in the logs (PII, health, payment, customer confidential)? What are your rules for sharing it, and do we need a DPA or NDA first? The CI artifacts and the committed baseline contain real inputs and outputs: may production-derived cases be committed, and how long may artifacts be kept?
11. Is there existing ground truth: FAQ answers, golden Q&A pairs, tickets resolved by humans?

### People and process

12. Who is the domain expert who can label outputs as pass/fail? We need about 5-8 hours of their time in week 2 (about 150 outputs at 2-3 minutes each). Who backs them up?
13. Who approves the definition of "good" (rubric), and who accepts the deliverables?
14. Where does your code live, what CI do you use (GitHub Actions, GitLab CI, other), and who can approve a new workflow?
15. What quality, cost and latency thresholds would block a release today, even informally?
16. Any upcoming change we should evaluate first? (A model upgrade, a prompt rewrite, a new knowledge base.) It often becomes the first comparison.

## 2. Access we need

Minimum set. Read-only wherever possible, and revocable at handover.

| What | Why | Notes |
|---|---|---|
| Anonymised sample of production inputs (and outputs, if logged), 500-2,000 rows | Source of the test set | Remove or mask PII before sharing where you can. If not, we work inside your environment. |
| A staging endpoint, or read access to the repository, to call the feature | The adapter calls the same code path as production | We never point the sprint at production without written agreement. |
| LLM provider API keys: one for the judge, one for the feature if we call it directly | Judge and system runs | Dedicated keys with a monthly spend limit. Typical judge cost is small: we report it. |
| Repository access with permission to open pull requests | Install the CI gate | Fork or branch access is enough. |
| A shared channel and one weekly 30-minute check-in | Decisions and unblocking | |
| 5-8 hours of a domain expert, in week 2 | About 150 human labels for calibration | The most common cause of delay. Book it at kickoff. |

## 3. Week by week

Two-week sprints compress weeks 2 and 3 into one; the steps are the same.

### Week 1: measure the current state

| Step | What happens | Output |
|---|---|---|
| Kickoff (day 1) | Walk through the questions above. Agree scope, thresholds to start with, and the acceptance criteria below. | Written scope |
| Adapter (days 1-2) | `evalsprint init --ci` writes the `evals/` folder and the workflows into your repository and the scaffold already runs offline. Write the small function that calls your feature and returns text, model, token usage and retrieved context. Run it on a handful of inputs. We decide together whether a crashed call may be repeated (`target_retries`: default 1, at most 5, `0` for a feature that is not idempotent). | `target_adapter.py`, first smoke run |
| Test set v1 (days 2-4) | Turn the log sample into about 150 cases (100 is the practical floor; below about 150 the gate needs `repeat: {n: 3}`, with the critical cases at 5 attempts through `per_tag`, and is a smoke test under 100): real traffic first, then reported failures, then known-hard categories (out of scope, injections, multi-part, ambiguous). Tag each case. Review a sample with you. | `cases.jsonl`, coverage table |
| Rubric v0 (days 3-4) | Write the criteria as yes/no questions, with what a fail looks like in your own examples. Add cheap deterministic checks (format, forbidden strings, length). | `rubric.yaml`, `eval.yaml` |
| First run (day 5) | Run the whole set. Share the first numbers and the failure taxonomy. | First report, findings call |

At the end of week 1 you already know how often the feature fails and in which categories, even before the judge is calibrated.

### Week 2: make the judge trustworthy

| Step | What happens | Output |
|---|---|---|
| Blind labeling (days 6-7) | Your expert labels about 150 outputs as pass/fail in a single-file HTML tool. The judge's verdict is hidden while they label. Autosaves; downloads a labels file. | `labels.jsonl` |
| Calibration (day 8) | Compare judge and human: agreement, Cohen's kappa with confidence interval, the decision taken on that interval (`trustworthy`, `usable`, `inconclusive`, `not_trustworthy`), and how often the judge misses a real failure or flags a good answer. Read every disagreement together. | Calibration report, `calibration.json` |
| Rubric iteration (days 8-9) | Fix the criteria that cause disagreements, add graded examples to the judge, re-run, re-measure. Usually one or two rounds. | `rubric.yaml` v1-v2 |
| Pass rule and thresholds (day 10) | Decide how judges combine into a verdict, and which cases are critical. | Agreed gate settings |

If agreement is still low after two rounds, or the interval is too wide to decide (`inconclusive`), we do not hide it. We report the measured agreement and the interval, restrict the judge to what it can grade reliably, and publish the *corrected* pass rate, which accounts for the judge's known error rates. Certifying a judge takes more labels than most sprints can afford, which is why the budget is **about 150 labels**, and more when your experts pass 90% or more of the outputs. The number depends on how often the humans pass the output, because the decision needs enough human *fails* to judge the true-negative rate. Best case, for a judge that agrees with every label (a real judge makes errors and needs more, so read the table as a floor, not a promise):

| Human pass rate | Every case labeled (usable / trustworthy / TNR screen) | 200-case run, stratified sample | 400-case run, stratified sample |
|---|---|---|---|
| 50% | 40 / 40 / 24 | 40 / 40 / 20 | 40 / 40 / 20 |
| 70% | 40 / 48 / 40 | 40 / 50 / 50 | 40 / 50 / 50 |
| 80% | 40 / 67 / 61 | 40 / 80 / 80 | 40 / 80 / 80 |
| 90% | 66 / 124 / 124 | 45 / 120 / 120 | 50 / 130 / 130 |
| 95% | 134 / 249 / 249 | 100 / more cases / more cases | 75 / 230 / 230 |

A strong but imperfect judge (95% TPR and TNR), stratified sample:

| Human pass rate | 200-case run | 400-case run |
|---|---|---|
| 70% | 60 / not reached / 60 | 60 / not reached / 60 |
| 80% | 100 / not reached / 100 | 100 / not reached / 100 |
| 90% | not reached / not reached / 150 | not reached / not reached / 200 |
| 95% | not reached / not reached / not reached | not reached / not reached / 300 |

How to read it: the number of human fails in a run is random (a 200-case run at 90% human pass has 20 fails on average and anywhere from about 8 to 35) and fewer fails need more labels, so every figure is the smallest number for which at least 80% of the runs get there, not the number for an average run. With **every case labeled**, the first column is the smallest run size from which, and for every larger size scanned, the judge is `usable`, `trustworthy` and passes the `tnr_reliable` screen (the bootstrap seed can move a figure by up to about 10 cases); the two right-hand columns are the fewest labels of a stratified sample (the labeling tool's default) of a 200- or 400-case run, rounded up to a multiple of 5, and `more cases` means that not even labeling every case of the run gets 80% of the runs there. In the strong-judge table the figure is the fewest labels on a grid of label counts (60, 100 and 150 on 200 cases; 60 to 300 on 400), and `not reached` means that no count simulated gets there; those are 200 draws per cell, not an exact enumeration. A judge that is right 95% of the time on both kinds of verdict is not reliably `trustworthy` within 150 labels of a 200-case run at 70%, 80% or 90% human pass rates. The figures are simulations with fixed seeds (`orchestration/2026-09-29_kit-v0.2/cycle4/loopback1/eval-scientist-judge/`), printed from `calibration.labels_budget_table()`; a test keeps this table equal to it.

A judge that agrees with every label is released at 40 labels (45 when exactly 90% of the labels are pass) and is called `trustworthy` from 40 to 96 labels when exactly that share of the labels is pass, or from 40 to 124 labels in 80% of the runs once the number of human fails is left to chance (the table above), but a merely good one needs many more: with 100 labels, a judge whose true kappa is about 0.7 is called `usable` less than half the time, and one that catches 90% of the bad outputs has its true-negative rate called reliable in about half the samples (98% at 150 labels; 68% at 150 for one that catches 80%). `tnr_reliable` is a **screening flag, not a guarantee**: it means the rate was measured narrowly and its estimate is at least 0.75, and a judge whose true rate is 70% still passes it in 2% to 28% of simulated runs. So `inconclusive` is a normal, honest outcome, and the report estimates how many more labels would settle it. When the rubric changes, `calibrate --export-fewshot` turns the clearest disagreements into examples for the judge and sets a holdout aside, so the next calibration is not graded on cases the judge was shown. After any change to the judge or the prompt, label only what has no label yet: the labeling export skips every case whose current output already has a label (`label --previous-labels`), and `calibrate` reads the old and the new label files together (`--labels` repeated). After a full earlier pass that is only the changed outputs; after a sampled pass the export also draws unchanged cases that never had a label, and labeling only the changed ones would bias the true-negative rate. A calibration remembers the judge configuration it was made for, and the report warns when the judges changed afterwards.

### Week 3: put it in the release process

| Step | What happens | Output |
|---|---|---|
| Baseline (day 11) | Freeze the current version's run as the reference, with the judge's `calibration.json` next to it. | Committed `evals/baseline/` |
| CI gate (days 11-13) | Install the workflows (`evalsprint init --ci`), `CODEOWNERS` and the required check. Prove it: a deliberate regression of 10 points or more is blocked, an unchanged pull request is compared and normally passes, and a simulated provider failure is reported as an infrastructure error, not a regression. Measure the noise floor, and run time and cost per PR. | Workflow, demo PRs, noise-floor note |
| Report (day 13) | Final quality and cost report, with calibration and a first before/after comparison if you had a change queued. | `report.html` |
| Handover (days 14-15) | Two-hour walkthrough with the engineers who will own it. Handover checklist below. | Signed-off handover |

## 4. Deliverables and acceptance criteria

We consider a deliverable done when every line below is true. Numbers are defaults; we adjust them at kickoff to fit the feature.

### 1. Test set

- About 150 cases, and at least 100 (more is better, see the detection numbers in section 4 below; even 200 catches a 5-point drop only about half the time); 45 to 99 only when the client's data does not allow more, and then the report says the gate is a smoke test. Below about 150 cases `eval.yaml` carries `repeat: {n: 3}` (every case is run three times and the majority decides), which costs three times the target and judge calls and is part of the baseline; the critical cases get `per_tag: {<critical tag>: 5}` on top of it. Unique ids, no duplicates, loads without errors (`evalsprint describe`, which also validates tags, check types and expected behaviors).
- At least half come from real production inputs; the rest are reported failures and deliberately hard cases.
- Every case has tags. Tags cover the main use categories and the critical risks you named at kickoff.
- At least 10% of cases test refusals, abstentions or out-of-scope requests (if the feature should ever decline).
- Personal data removed or masked; you confirmed the sample that we reviewed together.
- Each case states what a correct behavior is (a reference answer, expected behavior, or checks).

### 2. Calibrated judge

- About 150 outputs labeled blind by a domain expert (never fewer than 50; below 30 the report says `insufficient_data`; more when the expert passes 90% or more of the outputs, see the table in week 2), including some failures and some passes (the labeling page samples both on purpose; labels of only one kind cannot measure whether the judge misses bad outputs, and the report says to label the cases the judge failed).
- The report states agreement, Cohen's kappa with a confidence interval, the judge's true-positive rate (how often it passes a good answer) and true-negative rate (how often it catches a bad one, `tnr_reliable` unless there are at least 10 human fails and the interval is narrow enough: a screen, not a guarantee), and lists the disagreements.
- Target: the decision on the overall verdict is `usable` (lower bound of the kappa interval at least 0.6) or `trustworthy` (at least 0.8). The point estimate alone is not the criterion: it is too noisy at this sample size. If the target is not met after two rubric iterations (`inconclusive` or `not_trustworthy`), the report says so plainly, explains what the judge is and is not reliable for, gives the corrected pass rate with its interval, and estimates the labels still needed.
- `calibration.json` is committed next to the baseline, so the gate warns when the judge is uncalibrated.
- Judge cost per 100 cases is measured and documented.
- The rubric is committed with a one-page "how to change it" note.

### 3. Quality and cost report

- One self-contained HTML file that opens without a server.
- Overall pass rate with a confidence interval, and pass rate by tag.
- The failing cases, with the input, output and the judge's reasoning per criterion, so failure patterns can be read directly.
- Cost per request and per 1,000 requests computed from real token usage, plus the judge cost. Latency p50 and p95. Error rate.
- A calibration section (judge reliability) and, when a change was compared, a before/after with the cases that changed.
- Reproducible: one command regenerates it from the stored run.

### 4. CI regression gate

- Workflow installed in your repository, with `EVAL_PATHS` set to the folders that hold the prompts and LLM code (a pull request that touches none of them skips the paid run and the gate passes with a note; the workflow has no `paths:` filter, because that would leave the required check pending). The pull request's code runs in one job; the comparison runs in another that reads the gate settings, the baseline and the kit pin from the pull request's base branch and never executes the pull request's code, so a pull request cannot loosen its own gate, swap the baseline or patch the code that judges it. It **can** still fake its own results (an adapter that returns canned answers, a judge that always passes): reviewing `evals/` and the code the adapter calls is the control for that, and `REQUIRE_BASELINE` is switched to `"true"` after the first baseline is merged, so a lost baseline fails the gate instead of turning it off; `CODEOWNERS` puts `evals/` and the workflows under the named owners, and the check called `LLM evals gate` and code-owner review are required in branch protection.
- Baseline committed (with `calibration.json` when the judge is calibrated), with a documented way to refresh it through a reviewed pull request. Deliberate changes to the test set or the judges use that same way; after a judge change the calibration is redone, because the old one described the old judge.
- Gate settings written in `eval.yaml`: the allowed drop in pass rate (the default `auto` rule is fine: one rule for every test-set size), critical tags (repeated 5 times with a majority vote through `repeat: {n: 3, per_tag: {<critical tag>: 5}}`, which also repeats every other case 3 times below about 150 cases; not `n: 5` on every case), and cost and latency limits if you want them. The `repeat` setting, the judges and a custom judge's source are part of the baseline: changing any of them needs a reviewed baseline update.
- Demonstrated on pull requests: a deliberate regression of at least 10 percentage points is blocked with a readable reason; an unchanged pull request is compared and, in most runs, passes; a run with an invalid provider key ends in "infrastructure error, not a quality regression", never in a regression.
- **The gate is a noisy instrument, and we say how noisy.** For the size of your test set the handover states the exact rates from the tables (workflow README, section 9) together with the flip rate that a `--repeat 3` run measured (`summary.stability.flip_rate`: your own noise, not an assumed one). The output of `evalsprint noise` goes into the handover only with the flip rate and its assumptions, and not as the suite's rates: it prices the whole gate for the run's own repeat protocol, "about", from at least 20 repeated cases, and takes the noise measured on those cases for every case (the 10 cases of the scaffold give "not estimated"). At the assumed 5% per-case noise and one attempt per case an unchanged change is blocked about 1 time in 9 at 45 cases (10.6%), 7 times in 100 at 100 cases and 4 times in 100 at 200 cases, and a 10-point drop is caught 67% (45 cases), 80% (100) or 93% (200) of the time. A 5-point drop is caught about half the time at 200 cases and less at smaller sizes. With every case repeated 3 times (the setting for suites below about 150 cases) the same numbers at 45 cases are 0.3% false alarms and 82% detection of a 10-point drop, if the attempts are independent. **Four things we say out loud, because they are what a client would otherwise assume away.** (1) The false alarm is not flat: it peaks at 12.3% near 54 cases (15.1% at 10% noise), and between 60 and 149 cases the gate of evalsprint 0.1 had a lower false alarm and a higher detection at 84 of 90 sizes at 5% noise (every size except 60 to 62 and 78 to 80 cases; at 10% noise the current rule is the stricter one and blocks less often); the current rule removes the cliff at 150 cases, it is not better everywhere. (2) "At most 5% from 150 cases on" holds only if the target does not crash: a 1% crash rate doubles it at 200 cases (4.0% to 8.0%); and a retry hides crashes from the pass rate (the gate blocks a changed `target_retries` and warns when the candidate retried 10 percentage points more of its calls, a weak warning for small increases), so the crash and retry counts are read next to every verdict. (3) The committed baseline is one noisy draw, so these rates are averages over baselines you could have drawn: the chance that yours makes more than 15% of later unchanged pull requests fail is 28.8%, 14.0% and 7.2% at 45, 100 and 200 cases. We build it with the same repeat protocol as the pull requests, which shrinks that to under 0.15%, and we do not re-draw it until pull requests pass. (4) With critical cases, repeating everything 3 times blocks 5.79% of unchanged pull requests at 45 cases with 8 critical ones; repeating the critical cases 5 times and the rest 3 times blocks 1.11% (exact figures for the two rules together). The agreed policy for a blocked pull request that changed nothing relevant is to re-run the job once and to look at the cases that flipped if it blocks again.
- Run time and cost per pull request documented. If the full set is too slow or costly for every PR, a smaller subset is defined for PRs and the full set is scheduled, knowing that a smaller paired set makes a less sensitive gate.
- The check reports a readable summary in the PR (one sticky comment, updated in place): what changed, which cases, warnings such as an uncalibrated judge, and why the verdict was reached.
- For pull requests from forks, where secrets are not available, the behavior is agreed: a visible warning (default) or a failing check (`FAIL_WITHOUT_SECRETS`).

## 5. Handover checklist

- [ ] Repository contains `evals/` with `eval.yaml`, `rubric.yaml`, `cases.jsonl`, `target_adapter.py`, `requirements.txt` (the kit pinned), labels, baseline; `runs/` and `.label_secret` are in `.gitignore`.
- [ ] `evalsprint run` works from a clean checkout on your machine, not only on ours.
- [ ] API keys are stored as secrets in your CI; ours are revoked.
- [ ] CI workflow is merged, and the demo pull requests (the blocked regression, the unchanged pass, the provider-failure case) are documented. The baseline was made with the real target and judge, not the scaffold's placeholders (`summary.placeholder` is false; `baseline` refuses it).
- [ ] The kit's read token: either it is a fine-grained read-only token for the kit repository alone and you know the job that runs pull requests' code can read it while it installs requirements, or the wheel of the kit is vendored in `evals/vendor/` and that job has no token (the workflow README, section 2). Artifact retention (`ARTIFACT_RETENTION_DAYS`) is agreed.
- [ ] `.github/CODEOWNERS` is committed, code-owner review and the `LLM evals gate` check are required on the default branch, and the fork-PR policy is decided.
- [ ] Final report delivered; calibration numbers recorded in it.
- [ ] Walkthrough done: how to add a case, how to write a criterion, how to refresh the baseline, how to read a gate failure, how to change the model or price table, how to resume an interrupted run (`run --resume`: what it reuses and when it reuses nothing, `--retry-errors`, the lock a running resume holds), what `target_retries` is set to and why, how to read `evalsprint history`.
- [ ] Pricing table checked against your providers' current prices.
- [ ] Known limits written down: what the test set does not cover, what the judge is unreliable at, the size of the noise floor (`flip_rate`), the false-alarm and detection rates for your test-set size (with the 12.3% peak, the baseline lottery and the crash caveat), that the gate cannot tell whether a pull request's results are genuine, and that `injection_suspected` is a hint (43% recall and 27.5% false positives when measured on the first version), not a defense.
- [ ] Owner named on your side for each: test set, rubric, baseline, CI.

## 6. Optional monthly maintenance

For teams that want the eval to keep pace with the feature. Monthly, at a fixed price that depends on volume:

- **Test set refresh:** add new cases from recent production failures and new usage patterns (about 20 per month) and retire obsolete ones.
- **Re-calibration:** label 20-30 fresh outputs and re-measure judge agreement together with the earlier labels (a decision needs at least 30 labels and a release at least 40, so 20-30 fresh ones alone decide nothing), so drift is caught early, and refresh `calibration.json`. The `history` page shows whether the judge changed between runs. Rubric tuning if agreement falls.
- **Gate review:** look at every gate failure and near-miss; adjust thresholds that are too tight or too loose.
- **Change checks:** run one requested comparison per month (a model upgrade, a prompt rewrite, a provider change) and write a short recommendation.
- **Monthly report:** quality, cost, latency and reliability trend, with the three biggest failure categories.
- **Housekeeping:** keep the kit and price table up to date.

Not included: fixing the feature, on-call response, building new features, or evaluating additional features (each is scoped as a separate sprint).

## 7. Risks and how we handle them

| Risk | What we do |
|---|---|
| No logs, or logs cannot be shared | Build the test set from support tickets, documentation and workshop sessions with your team; label it as synthetic in the report. Expect a wider uncertainty on how representative it is. |
| Domain expert not available in week 2 | The sprint pauses on calibration, not on the rest. We continue the CI gate with heuristic checks and mark the judge as uncalibrated until labels arrive. |
| Feature output is non-deterministic | Measure the noise floor with `run --repeat 3` (the summary's `flip_rate`) and report it (`evalsprint noise` turns it into "about" figures for the whole gate once at least 20 cases are repeated). Below about 150 cases every case is repeated 3 times; critical cases are repeated 5 times (`per_tag`) and decided by majority (about 1% false alarms on 8 critical cases at 5% noise). The default gate is one rule for every size of the paired set; we state the remaining false-alarm rate instead of hiding it. |
| The target crashes on some calls, or crashes more after a change | A failed target call is repeated once inside the run (`target_retries: 1`; `0` for a target that is not idempotent) and counted in the cost; a crash that survives counts as a failure, and a crash on a case that passed in the baseline counts against the gate. A retry hides a transient crash from the pass rate, so a changed `target_retries` blocks until the baseline is refreshed in a reviewed pull request, the comparison prints the retries of both runs, and a candidate that retried 10 percentage points more of its calls than the baseline raises a warning. That warning is weak for small increases (against a baseline that retried 5% of its calls it fires in 11 of 100 comparisons when the candidate's calls crash 10% of the time, 89 of 100 at 20%), so the handover reads the crash and retry counts of every comparison and says so. |
| Small test set gives wide error bars | The report shows the intervals and the PR comment states the smallest drop the gate can reliably see. Under 100 cases the gate is a smoke test even with every case repeated 3 times; we say so and recommend growing the set to about 150. We only claim a difference when the paired statistics support it. |
| A pull request weakens its own gate (edits `gate:`, replaces the baseline, deletes or edits the cases it fails, loosens the judge or the rubric) | Gate settings, baseline and the kit pin are read from the base branch and the comparison runs in a job without the pull request's code; removed and edited cases and a changed judge block, rubric, `pass_rule`, custom-judge source or `repeat` protocol are detected and block by default ("update the baseline in a reviewed PR"). What the gate cannot see (the target's code, what a custom judge imports, the workflow files, how easy new cases are) is covered by `CODEOWNERS` and required code-owner review, which are therefore part of acceptance, not optional. |
| A pull request fakes its own results (an adapter that returns canned answers, a judge that always passes, a hand-written result file) | The gate does not catch it: the candidate results are produced by the pull request's own code, and only the comparison, the gate settings, the baseline and the kit are out of its reach (the exit code and the files that cross between the jobs are validated, which blocks a forged verdict but not a genuine `0` with forged files). Code-owner review of `evals/` and of the code the adapter calls is the control, the API key is a dedicated one with a spend limit (the job that runs the pull request's code holds it, and so it holds the kit's read token while it installs requirements: vendor the wheel for a kit that must stay private), and we tell you so at handover. |
| The provider or the judge API is down during a run | The run is reported as unhealthy (exit 3, "infrastructure error, not a quality regression"), never compared and never accepted as a baseline. Re-run when it recovers. A run that was cut short (Ctrl-C exits 130, a cancelled job or a stopped container exits 143) keeps the attempts it finished: `run --resume DIR` pays only for the rest. The resume notices a change to the adapter's own file but **not** to the modules that file imports: after editing those, start a fresh run. |
| Judge disagrees with your experts | Rubric iteration, graded examples, restricting the judge to criteria it grades reliably, and reporting the corrected pass rate. Never a silent override. |
| Cost per PR too high | Smaller PR subset (a less sensitive gate), cheaper judge model for PRs after calibrating it, full set nightly. `repeat` and `samples` multiply cost for the cases and judges they apply to. |
| Prices change | Prices live in a table you can override in `eval.yaml`; verify before quoting to your finance team. |

## 8. What we need from you each week

- **Week 1:** the data sample, access, answers to the kickoff questions, one review session of the draft test set (30-45 minutes).
- **Week 2:** the domain expert's labeling time, and an hour to review disagreements.
- **Week 3:** a reviewer for the CI pull request, and the engineers who will own the eval at the handover.
