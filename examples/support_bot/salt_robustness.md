# Salt robustness of the Ledgerly demo

Each claim is checked under 200 different values of `SIM_SEED` (every invented effect is redrawn) plus the default seed, which produced the numbers in the README. The salts come from 2 independent batches (salt00..salt99: 100; critic000..critic099: 100), run under the same code, so a count that differs between batches shows how much a count over that many salts can be trusted. A claim may be stated as a headline only if it holds in at least 90% of the pooled salts. Ranges are percentages (points for differences) across the pooled salts.

Kind: MECHANISM = code that really runs (retrieval, harness, statistics, prices); SCRIPTED = true by construction of the simulator; INVENTED = a rate picked for the story.

| # | Kind | Claim | Salts where it holds | Share | Holds per batch | Default seed | Verdict | Distribution of the underlying value |
|---|---|---|---|---|---|---|---|---|
| C01 | SCRIPTED | v1 (generic prompt) passes at most 2 of the 8 out-of-scope cases | 197/200 | 98% | 99, 98 | yes | HEADLINE OK | min 0.0, p10 0.0, median 12.5, p90 25.0, max 50.0 |
| C02 | SCRIPTED | v1 passes at most 3 of the 8 safety cases | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min 0.0, p10 0.0, median 0.0, p90 12.5, max 25.0 |
| C03 | SCRIPTED | v2 passes at least 7 of the 8 safety cases | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min 100.0, p10 100.0, median 100.0, p90 100.0, max 100.0 |
| C04 | SCRIPTED | v2 passes all 8 safety cases | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min 100.0, p10 100.0, median 100.0, p90 100.0, max 100.0 |
| C05 | SCRIPTED | v2 passes at least 6 of the 8 out-of-scope cases | 199/200 | 100% | 99, 100 | yes | HEADLINE OK | min 50.0, p10 87.5, median 100.0, p90 100.0, max 100.0 |
| C06 | MECHANISM+SCRIPTED | v2 pass rate is between 80% and 95% | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min 82.2, p10 88.9, median 91.1, p90 91.1, max 93.3 |
| C07 | SCRIPTED | v2 pass rate is at least 15 points above v1 | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min 22.2, p10 26.7, median 31.1, p90 35.6, max 42.2 |
| C08 | MECHANISM | the v1 -> v2 gate PASSES | 200/200 | 100% | 100, 100 | yes | HEADLINE OK |  |
| C09 | MECHANISM | the v1 -> v2 gate names no critical (safety) regression | 200/200 | 100% | 100, 100 | yes | HEADLINE OK |  |
| C10 | MECHANISM | v3 passes at most 4 of the 9 multi-part cases (v2: at least 7) | 198/200 | 99% | 100, 98 | yes | HEADLINE OK | min 22.2, p10 33.3, median 33.3, p90 33.3, max 44.4 |
| C11 | MECHANISM | the v2 -> v3 gate is a REGRESSION | 199/200 | 100% | 99, 100 | yes | HEADLINE OK |  |
| C12 | MECHANISM | the v2 -> v3 regression comes from the pass-rate drop (a pass-rate reason is listed) | 199/200 | 100% | 99, 100 | yes | HEADLINE OK |  |
| C13 | MECHANISM | v3 costs at least 50% less per request than v2 | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min 55.2, p10 57.6, median 58.0, p90 58.3, max 58.9 |
| C14 | MECHANISM | v2 costs less than 30% more per request than v1 | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min 16.8, p10 17.9, median 18.6, p90 19.4, max 20.7 |
| C15 | INVENTED | the v2 -> v3 gate names a critical safety regression | 26/200 | 13% | 12, 14 | no | do not state |  |
| C16 | INVENTED | v3 passes all 8 safety cases | 174/200 | 87% | 88, 86 | yes | range only | min 87.5, p10 87.5, median 100.0, p90 100.0, max 100.0 |
| C17 | SCRIPTED | refunds-002 ("how long do refunds take to arrive") passes on v1 and fails on v2 | 136/200 | 68% | 70, 66 | yes | range only |  |
| C18 | MECHANISM | attribution: with Sonnet and top_k 1 multi-part is still at most 4 of 9 | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min 22.2, p10 33.3, median 33.3, p90 33.3, max 44.4 |
| C19 | MECHANISM | attribution: with Haiku and top_k 3 multi-part is at least 7 of 9 | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min 77.8, p10 88.9, median 88.9, p90 88.9, max 88.9 |
| C20 | MECHANISM | attribution: v2 -> (Sonnet, top_k 1) is a REGRESSION | 199/200 | 100% | 99, 100 | yes | HEADLINE OK |  |
| C21 | INVENTED | Haiku with top_k 3 also fails the gate (safety flips or pass-rate drop) | 35/200 | 18% | 19, 16 | yes | do not state |  |
| C22 | INVENTED | flip rate on the repeated safety cases: v2 lower than v1 | 189/200 | 94% | 92, 97 | yes | HEADLINE OK | min 0.0, p10 0.0, median 12.5, p90 25.0, max 62.5 |
| C23 | INVENTED | flip rate of v2 is at most 25% of the 8 repeated cases | 189/200 | 94% | 93, 96 | yes | HEADLINE OK | min 0.0, p10 0.0, median 12.5, p90 25.0, max 62.5 |
| C24 | INVENTED | flip rate of v1 is at least 25% | 196/200 | 98% | 99, 97 | yes | HEADLINE OK | min 12.5, p10 37.5, median 50.0, p90 62.5, max 75.0 |
| C25 | INVENTED | flip rate of v3 is at least 25% | 188/200 | 94% | 94, 94 | yes | HEADLINE OK | min 0.0, p10 25.0, median 50.0, p90 62.5, max 87.5 |
| C26 | MECHANISM | no run has an error (errors are 0 in v1, v2 and v3) | 200/200 | 100% | 100, 100 | yes | HEADLINE OK |  |
| C27 | INVENTED | A/A: the gate does NOT flag v2 run twice as a regression | 198/200 | 99% | 99, 99 | yes | HEADLINE OK | min -8.9, p10 -2.2, median 0.0, p90 2.2, max 6.7 |
| C28 | INVENTED | A/A: the gate does NOT flag v1 run twice as a regression | 137/200 | 68% | 59, 78 | no | range only | min -11.1, p10 -4.4, median 0.0, p90 4.4, max 13.3 |
| C29 | INVENTED | A/A: the gate does NOT flag v3 run twice as a regression | 178/200 | 89% | 87, 91 | yes | range only | min -8.9, p10 -4.4, median 0.0, p90 4.4, max 6.7 |
| C30 | SCRIPTED | calibration (synthetic human) agreement on v2 is at least 85% | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min 90.0, p10 95.0, median 95.0, p90 97.5, max 100.0 |
| C31 | SCRIPTED | kappa_decision on the v2 calibration is 'inconclusive' or better than 'not_trustworthy' | 200/200 | 100% | 100, 100 | yes | HEADLINE OK |  |
| C32 | MECHANISM | the 40 labels do not settle the judge: decision is 'inconclusive' or 'insufficient_data' | 200/200 | 100% | 100, 100 | yes | HEADLINE OK |  |
| C33 | MECHANISM | the true-negative screening check is not passed (few failing cases in the sample) | 200/200 | 100% | 100, 100 | yes | HEADLINE OK |  |
| C34 | SCRIPTED | oracle and simulated judge agree on at least 85% of all 135 records | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min 90.4, p10 91.9, median 92.6, p90 93.3, max 94.8 |
| C35 | SCRIPTED | oracle and simulated judge disagree on at least 3 records (they are not the same rules) | 200/200 | 100% | 100, 100 | yes | HEADLINE OK |  |
| C36 | SCRIPTED | refunds-002 fails on v2 (the grounded prompt abstains on the paraphrase "arrive" for "reach") | 196/200 | 98% | 98, 98 | yes | HEADLINE OK |  |
| C37 | MECHANISM | the labels needed for a +/-0.10 kappa interval exceed the 45 labelable cases, so the calibration says to label all of them (labels_needed_capped) | 200/200 | 100% | 100, 100 | yes | HEADLINE OK |  |
| C38 | INVENTED | A/A with repeat n=3 on every case: the gate does NOT flag v1 run twice as a regression | 111/200 | 56% | 50, 61 | no | range only | min -6.7, p10 -4.4, median 0.0, p90 4.4, max 11.1 |
| C39 | INVENTED | A/A with repeat n=3 on every case: the gate does NOT flag v2 run twice as a regression | 198/200 | 99% | 100, 98 | yes | HEADLINE OK | min -2.2, p10 0.0, median 0.0, p90 0.0, max 2.2 |
| C40 | INVENTED | A/A with repeat n=3 on every case: the gate does NOT flag v3 run twice as a regression | 149/200 | 74% | 72, 77 | no | range only | min -6.7, p10 -2.2, median 0.0, p90 2.2, max 6.7 |
| C41 | MECHANISM | with repeat n=3 on every case the v2 -> v3 gate is still a REGRESSION | 200/200 | 100% | 100, 100 | yes | HEADLINE OK |  |
| C42 | INVENTED | A/A with repeat n=5 on every case: the gate does NOT flag v1 run twice as a regression | 144/200 | 72% | 64, 80 | no | range only | min -6.7, p10 -2.2, median 0.0, p90 2.2, max 8.9 |
| C43 | INVENTED | A/A with repeat n=5 on every case: the gate does NOT flag v2 run twice as a regression | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min -2.2, p10 0.0, median 0.0, p90 0.0, max 0.0 |
| C44 | INVENTED | A/A with repeat n=5 on every case: the gate does NOT flag v3 run twice as a regression | 184/200 | 92% | 90, 94 | yes | HEADLINE OK | min -4.4, p10 -2.2, median 0.0, p90 2.2, max 4.4 |
| C45 | MECHANISM | with repeat n=5 on every case the v2 -> v3 gate is still a REGRESSION | 200/200 | 100% | 100, 100 | yes | HEADLINE OK |  |
| C46 | INVENTED | A/A with repeat n=3 and per_tag safety 5: the gate does NOT flag v1 run twice as a regression | 143/200 | 72% | 63, 80 | no | range only | min -8.9, p10 -4.4, median 0.0, p90 4.4, max 8.9 |
| C47 | INVENTED | A/A with repeat n=3 and per_tag safety 5: the gate does NOT flag v2 run twice as a regression | 200/200 | 100% | 100, 100 | yes | HEADLINE OK | min -2.2, p10 0.0, median 0.0, p90 0.0, max 2.2 |
| C48 | INVENTED | A/A with repeat n=3 and per_tag safety 5: the gate does NOT flag v3 run twice as a regression | 183/200 | 92% | 90, 93 | yes | HEADLINE OK | min -6.7, p10 -2.2, median 0.0, p90 2.2, max 6.7 |
| C49 | MECHANISM | with repeat n=3 and per_tag safety 5 the v2 -> v3 gate is still a REGRESSION | 200/200 | 100% | 100, 100 | yes | HEADLINE OK |  |

## README headlines and their evidence

| Headline | Backing claims | All hold in at least 90% of the salts |
|---|---|---|
| one section (top_k 1) breaks multi-part questions: v3 at most 4 of 9, v2 at least 7 of 9 | C10 (198/200) | yes |
| the collapse comes from top_k, not from the model (Sonnet+top_k 1 still collapses, Haiku+top_k 3 does not) | C18, C19, C20 (200/200, 200/200, 199/200) | yes |
| the v2 -> v3 gate says REGRESSION, because of the pass-rate drop | C11, C12 (199/200, 199/200) | yes |
| v3 is at least 50% cheaper per request than v2; v2 costs less than 30% more than v1 | C13, C14 (200/200, 200/200) | yes |
| v1 passes at most 2 of 8 out-of-scope and 3 of 8 safety cases | C01, C02 (197/200, 200/200) | yes |
| v2 passes all 8 safety cases and at least 6 of 8 out-of-scope cases; its pass rate is 80-95% | C04, C05, C06 (200/200, 199/200, 200/200) | yes |
| v1 -> v2 gate PASSES, with a gain of at least 15 points and no critical regression | C07, C08, C09 (200/200, 200/200, 200/200) | yes |
| v2 fails refunds-002 (the paraphrase abstain) | C36 (196/200) | yes |
| calibration on the synthetic human: agreement at least 85%, the 40 labels leave the judge inconclusive (decided on the weighted kappa), TNR screening check not passed | C30, C32, C33 (200/200, 200/200, 200/200) | yes |
| the 45 labelable cases cannot give a +/-0.10 kappa interval: the calibration says to label all 45 and add cases or repeats | C37 (200/200) | yes |
| oracle and simulated judge agree on at least 85% of the records and differ on at least 3 | C34, C35 (200/200, 200/200) | yes |
| the same v2 configuration run twice does not trip the gate | C27 (198/200) | yes |

Every README headline is backed by claims that held in at least 90% of the salts: **yes**.
Quoted only as a range or a share, never as a headline (they did not reach 90%, or are properties of the invented rates): C15 (26/200), C16 (174/200), C17 (136/200), C21 (35/200), C22 (189/200), C24 (196/200), C25 (188/200), C28 (137/200), C29 (178/200), C38 (111/200), C39 (198/200), C40 (149/200), C42 (144/200), C43 (200/200), C44 (184/200), C46 (143/200), C47 (200/200), C48 (183/200).

## Pass rates across salts (percent)

| Variant | Pass rate | Answerable | Multi-part | Out of scope | Safety | Flip rate | Cost / 1k (USD) |
|---|---|---|---|---|---|---|---|
| v1 baseline | min 49, p10 56, median 58, p90 62, max 64 | min 75, p10 85, median 90, p90 90, max 90 | min 67, p10 78, median 89, p90 89, max 89 | min 0, p10 0, median 12, p90 25, max 50 | min 0, p10 0, median 0, p90 12, max 25 | min 12, p10 38, median 50, p90 62, max 75 | min 1.35, p10 1.37, median 1.37, p90 1.38, max 1.38 |
| v2 grounded | min 82, p10 89, median 91, p90 91, max 93 | min 80, p10 80, median 85, p90 85, max 90 | min 56, p10 78, median 89, p90 89, max 89 | min 50, p10 88, median 100, p90 100, max 100 | min 100, p10 100, median 100, p90 100, max 100 | min 0, p10 0, median 12, p90 25, max 62 | min 1.61, p10 1.62, median 1.63, p90 1.64, max 1.66 |
| v3 cost cut | min 67, p10 71, median 76, p90 78, max 80 | min 70, p10 75, median 80, p90 80, max 80 | min 22, p10 33, median 33, p90 33, max 44 | min 50, p10 75, median 88, p90 100, max 100 | min 88, p10 88, median 100, p90 100, max 100 | min 0, p10 25, median 50, p90 62, max 88 | min 0.67, p10 0.68, median 0.68, p90 0.69, max 0.73 |

## Gate verdicts across salts

- v1 -> v2: {'pass': 200}; gate modes seen: ['auto']; runs with a critical (safety) regression: 0/200
- v2 -> v3: {'regression': 199, 'pass': 1}; gate modes seen: ['auto']; runs with a critical (safety) regression: 26/200
- v2 -> abl_topk1_sonnet: {'regression': 199, 'pass': 1}; multi-part passes min 22, p10 33, median 33, p90 33, max 44; critical regressions in 0/200; cost/1k min 1.35, p10 1.36, median 1.37, p90 1.37, max 1.38
- v2 -> abl_haiku_topk3: {'regression': 35, 'pass': 165}; multi-part passes min 78, p10 89, median 89, p90 89, max 89; critical regressions in 26/200; cost/1k min 0.81, p10 0.81, median 0.82, p90 0.82, max 0.87

## A/A: the same configuration run twice under two salts

| Configuration | Gate says REGRESSION | Per batch | Pass-rate change between the two runs, points |
|---|---|---|---|
| v1 | 63/200 (32%) | 41/100, 22/100 | min -11.1, p10 -4.4, median 0.0, p90 4.4, max 13.3 |
| v2 | 2/200 (1%) | 1/100, 1/100 | min -8.9, p10 -2.2, median 0.0, p90 2.2, max 6.7 |
| v3 | 22/200 (11%) | 13/100, 9/100 | min -8.9, p10 -4.4, median 0.0, p90 4.4, max 6.7 |

This is the false-alarm rate of the gate on the simulator's INVENTED run-to-run noise, not on any real model.

## A/A and detection with four repeat protocols

All protocols run the same 45 cases through the same gate (the `auto` rule, `critical_tags: [safety]`); only `repeat` changes. **safety x5** is what `eval.yaml` runs, `repeat: {n: 1, per_tag: {safety: 5}}` (`{n: 5, tags: [safety]}` in its old spelling): the 8 safety cases run 5 times, everything else once. **all cases x3** is `repeat: {n: 3}`: every case, safety included, runs 3 times and the majority decides. **all cases x5** is `repeat: {n: 5}`. **safety x5 + rest x3** is `repeat: {n: 3, per_tag: {safety: 5}}`: every case 3 times, the safety cases 5 times. Attempts per run: safety x5 (eval.yaml) 77, all cases x3 135, all cases x5 225, safety x5 + rest x3 (per_tag) 151. The gate refuses to compare runs made under different protocols, so each A/A pair and each detection pair keeps one.

False alarm = the gate says REGRESSION on the SAME configuration run twice under two salts (a salt redraws every invented effect). Detection = the gate says REGRESSION on a change that really is worse. Every cell counts salts; the interval is Wilson, 95%.

| Protocol | v1 A/A false alarm | v2 A/A false alarm | v3 A/A false alarm | v2 -> v3 detected | v2 -> Haiku top_k 3 detected |
|---|---|---|---|---|---|
| safety x5 (eval.yaml), all 200 salts | 63/200 (32%; 25-38%) | 2/200 (1%; 0-4%) | 22/200 (11%; 7-16%) | 199/200 (100%; 97-100%) | 35/200 (18%; 13-23%) |
| safety x5 (eval.yaml), batch salt00..salt99 | 41/100 (41%; 32-51%) | 1/100 (1%; 0-5%) | 13/100 (13%; 8-21%) | 99/100 (99%; 95-100%) | 19/100 (19%; 13-28%) |
| safety x5 (eval.yaml), batch critic000..critic099 | 22/100 (22%; 15-31%) | 1/100 (1%; 0-5%) | 9/100 (9%; 5-16%) | 100/100 (100%; 96-100%) | 16/100 (16%; 10-24%) |
| all cases x3, all 200 salts | 89/200 (44%; 38-51%) | 2/200 (1%; 0-4%) | 51/200 (26%; 20-32%) | 200/200 (100%; 98-100%) | 62/200 (31%; 25-38%) |
| all cases x3, batch salt00..salt99 | 50/100 (50%; 40-60%) | 0/100 (0%; 0-4%) | 28/100 (28%; 20-37%) | 100/100 (100%; 96-100%) | 29/100 (29%; 21-39%) |
| all cases x3, batch critic000..critic099 | 39/100 (39%; 30-49%) | 2/100 (2%; 1-7%) | 23/100 (23%; 16-32%) | 100/100 (100%; 96-100%) | 33/100 (33%; 25-43%) |
| all cases x5, all 200 salts | 56/200 (28%; 22-35%) | 0/200 (0%; 0-2%) | 16/200 (8%; 5-13%) | 200/200 (100%; 98-100%) | 26/200 (13%; 9-18%) |
| all cases x5, batch salt00..salt99 | 36/100 (36%; 27-46%) | 0/100 (0%; 0-4%) | 10/100 (10%; 6-17%) | 100/100 (100%; 96-100%) | 13/100 (13%; 8-21%) |
| all cases x5, batch critic000..critic099 | 20/100 (20%; 13-29%) | 0/100 (0%; 0-4%) | 6/100 (6%; 3-12%) | 100/100 (100%; 96-100%) | 13/100 (13%; 8-21%) |
| safety x5 + rest x3 (per_tag), all 200 salts | 57/200 (28%; 23-35%) | 0/200 (0%; 0-2%) | 17/200 (8%; 5-13%) | 200/200 (100%; 98-100%) | 26/200 (13%; 9-18%) |
| safety x5 + rest x3 (per_tag), batch salt00..salt99 | 37/100 (37%; 28-47%) | 0/100 (0%; 0-4%) | 10/100 (10%; 6-17%) | 100/100 (100%; 96-100%) | 13/100 (13%; 8-21%) |
| safety x5 + rest x3 (per_tag), batch critic000..critic099 | 20/100 (20%; 13-29%) | 0/100 (0%; 0-4%) | 7/100 (7%; 3-14%) | 100/100 (100%; 96-100%) | 13/100 (13%; 8-21%) |

Which rule raised the false alarms, pooled over v1, v2 and v3 (600 A/A pairs per protocol). An alarm can come from the pass-rate rule (the `auto` net-loss-plus-McNemar rule on the whole set), from the critical-case rule (a safety case that went pass -> fail), or from both, so the last two columns can add up to more than the total.

| Protocol | Attempts per run | Any false alarm | Pass-rate rule fired | Critical-case rule fired |
|---|---|---|---|---|
| safety x5 (eval.yaml), all 200 salts | 77 | 87/600 (14%; 12-18%) | 28/600 (5%; 3-7%) | 72/600 (12%; 10-15%) |
| safety x5 (eval.yaml), batch salt00..salt99 | 77 | 55/300 (18%; 14-23%) | 16/300 (5%; 3-8%) | 46/300 (15%; 12-20%) |
| safety x5 (eval.yaml), batch critic000..critic099 | 77 | 32/300 (11%; 8-15%) | 12/300 (4%; 2-7%) | 26/300 (9%; 6-12%) |
| all cases x3, all 200 salts | 135 | 142/600 (24%; 20-27%) | 13/600 (2%; 1-4%) | 141/600 (24%; 20-27%) |
| all cases x3, batch salt00..salt99 | 135 | 78/300 (26%; 21-31%) | 6/300 (2%; 1-4%) | 78/300 (26%; 21-31%) |
| all cases x3, batch critic000..critic099 | 135 | 64/300 (21%; 17-26%) | 7/300 (2%; 1-5%) | 63/300 (21%; 17-26%) |
| all cases x5, all 200 salts | 225 | 72/600 (12%; 10-15%) | 2/600 (0%; 0-1%) | 72/600 (12%; 10-15%) |
| all cases x5, batch salt00..salt99 | 225 | 46/300 (15%; 12-20%) | 0/300 (0%; 0-1%) | 46/300 (15%; 12-20%) |
| all cases x5, batch critic000..critic099 | 225 | 26/300 (9%; 6-12%) | 2/300 (1%; 0-2%) | 26/300 (9%; 6-12%) |
| safety x5 + rest x3 (per_tag), all 200 salts | 151 | 74/600 (12%; 10-15%) | 7/600 (1%; 1-2%) | 72/600 (12%; 10-15%) |
| safety x5 + rest x3 (per_tag), batch salt00..salt99 | 151 | 47/300 (16%; 12-20%) | 2/300 (1%; 0-2%) | 46/300 (15%; 12-20%) |
| safety x5 + rest x3 (per_tag), batch critic000..critic099 | 151 | 27/300 (9%; 6-13%) | 5/300 (2%; 1-4%) | 26/300 (9%; 6-12%) |

Reading it: the noise here is the simulator's, an INVENTED structure (a per-attempt chance of ignoring a rule or of answering sloppily, redrawn by the salt), so the false-alarm rates are properties of those rates, not of any model. The critic's gate simulation (model B: 5% independent noise on every case, n=45) measured 10.9% false alarms falling to 0.3% with `repeat: {n: 3}` on every case; the second table is the comparable number here, the pass-rate rule alone. The critical-case rule is a separate rule: with `repeat: {n: 3}` on every case the safety cases run 3 times instead of 5, and a majority of 3 is noisier than a majority of 5.

## Calibration of the v2 run against the synthetic human (40 labels)

- kappa_decision across salts: {'inconclusive': 200}
- agreement (percent): min 90, p10 95, median 95, p90 98, max 100
- weighted kappa (whole run; the decision uses it): min 0.52, p10 0.71, median 0.71, p90 0.88, max 1.00
- lower bound of the weighted kappa CI: min -0.04, p10 0.11, median 0.11, p90 0.38, max 0.58
- unweighted kappa (labeled sample; secondary): min 0.55, p10 0.72, median 0.72, p90 0.87, max 1.00
- lower bound of the unweighted kappa CI: min -0.04, p10 0.00, median 0.00, p90 0.38, max 0.63
- labels needed for a +/-0.10 interval capped by the 45 labelable cases: 200/200 salts
- TNR screening check (`tnr_reliable`) not passed in 200/200 salts
- corrected pass rate minus observed (points): min -5.4, p10 -0.3, median -0.3, p90 2.2, max 4.4

## Oracle (synthetic human) vs simulated judge, all records of the three runs

- agreement (percent): min 90.4, p10 91.9, median 92.6, p90 93.3, max 94.8; kappa: min 0.74, p10 0.79, median 0.81, p90 0.84, max 0.86

| Type | Records (per salt) | Agreements, min | median | max |
|---|---|---|---|---|
| answerable | 60 | 55 | 57 | 59 |
| multi_part | 27 | 19 | 21 | 23 |
| out_of_scope | 24 | 24 | 24 | 24 |
| safety | 24 | 21 | 24 | 24 |

Literal phrases in the judge's detectors: 70; in the oracle's: 100; shared: 29 (Jaccard 0.21). Shared: adjusted, approved, backdated, can't, contact, couldn't, created, credit, credited, details, do not, don't, done, drafted, find, generated, have, information, issued, on its way, prepared, processed, refund, refunded, support, sure, teammate, will not, won't.

## Target blindness

- The target callable receives only: {"input": "q?", "params": {"prompt": "v2", "_attempt": 1}}
- Neither sim.py nor bot.py reads the reference answers or tags (the bot sees the question, the retrieved context and the prompt).
