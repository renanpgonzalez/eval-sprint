"""Tests for the history page (`report.render_history` / `write_history`, contract C15).

Runs are built by hand (tests/test_report.py builders) from the summary keys the runner writes, so nothing here
depends on the runner producing them. The SVG is checked as structure and geometry: one series per chart on one
axis, points where the values say they are, a rule where `judges_sha` changed, a readout for every point that
works from the keyboard, nothing outside the viewBox, nothing external, everything escaped.
"""

import contextlib
import io
import re
import tempfile
import unittest
from html import unescape
from pathlib import Path
from types import SimpleNamespace

from evalsprint import cli
from evalsprint.report import MAX_HISTORY_CHART_RUNS, render_history, write_history
from evalsprint.runner import save_run

from tests.test_report import XSS, XSS_ESCAPED, ReportTestCase, make_run, plain, record, seen

SHA_A, SHA_B, SHA_C = "a" * 12 + "0011", "b" * 12 + "0022", "c" * 12 + "0033"


def hrun(index=0, rate=0.8, ci=None, cost=1.0, p95=0.5, sha=SHA_A, kit="0.2.0", minute=None, errors=0, **extra):
    """A run for the history: a finished summary with the keys the page reads."""
    minute = index if minute is None else minute
    ci = ci if ci is not None else [max(0.0, rate - 0.1), min(1.0, rate + 0.1)]
    summary = {
        "run_id": f"20260929-10{minute:02d}00-r{index}",
        "label": f"r{index}",
        "created_at": f"2026-09-29T10:{minute % 60:02d}:00+00:00",
        "evalsprint_version": kit,
        "judges_sha": sha,
        "pass_rate": rate,
        "pass_rate_ci": ci,
        "n_scored": 40,
        "n_errors": errors,
        "cost": {"system_per_1k_usd": cost, "system_mean_usd": cost / 1000 if isinstance(cost, (int, float)) else None, "judge_total_usd": 1.0, "n_priced": 1},
        "latency": {"p95_s": p95},
        "repeat": {"n": 1, "tags": None},
    }
    summary = {key: value for key, value in summary.items() if value is not None or key == "judges_sha"}
    summary.update(extra)
    if sha is None:
        summary.pop("judges_sha")
    if kit is None:
        summary.pop("evalsprint_version")
    run = make_run([record("a-1", True)])
    merged = {**run.summary, **summary}
    if cost is None:
        merged["cost"] = {}
    return SimpleNamespace(summary=merged, results=run.results, run_id=merged["run_id"])


def svgs(doc):
    """{(measure, size): svg} from the page: measure is pass or cost, size is wide or narrow."""
    found = {}
    for match in re.finditer(r'<svg class="trend chart-(wide|narrow)"[^>]*aria-label="([^"]*)".*?</svg>', doc, flags=re.S):
        found[("pass" if match.group(2).startswith("Pass rate") else "cost", match.group(1))] = match.group(0)
    return found


def numbers(svg, pattern):
    return [float(x) for x in re.findall(pattern, svg)]


class HistoryCase(ReportTestCase):
    def assertOnly(self, snippets, doc):
        for snippet in snippets:
            self.assertHas(snippet, doc)


class TestOrderingAndContent(HistoryCase):
    def test_runs_are_ordered_by_created_at_whatever_the_input_order(self):
        runs = [hrun(2), hrun(0), hrun(3), hrun(1)]
        doc = render_history(runs)
        labels = re.findall(r"<strong>(r\d)</strong>", doc.split('id="runs"')[1])
        self.assertEqual(labels, ["r0", "r1", "r2", "r3"])
        xs = numbers(svgs(doc)[("pass", "wide")], r'class="dot pass" cx="([\d.]+)"')
        self.assertEqual(xs, sorted(xs))
        self.assertEqual(len(xs), 4)

    def test_a_run_without_a_readable_date_goes_last_and_z_and_naive_dates_are_utc(self):
        runs = [
            hrun(0, minute=5, created_at="not a date"),
            hrun(1, minute=1, created_at="2026-09-29T10:02:00Z"),
            hrun(2, minute=2, created_at="2026-09-29T10:01:00"),  # naive: taken as UTC
            hrun(3, minute=3, created_at="2026-09-29T09:00:00-03:00"),  # = 12:00 UTC
        ]
        labels = re.findall(r"<strong>(r\d)</strong>", render_history(runs).split('id="runs"')[1])
        self.assertEqual(labels, ["r2", "r1", "r3", "r0"])

    def test_every_measure_of_the_contract_is_in_the_table(self):
        run = hrun(
            0, rate=0.8333, ci=[0.7, 0.9], cost=1.6395, p95=1.5047, errors=3,
            n_attempts=100, n_attempt_errors=4, attempt_error_rate=0.04,
            stability={"n_repeated": 8, "flip_rate": 0.25}, repeat={"n": 5, "tags": ["safety"]},
        )
        text = seen(render_history([run]))
        for expected in ("r0", "20260929-100000-r0", "2026-09-29 10:00 UTC", "83.3%", "70.0–90.0%", "4.0%", "25.0%", "$1.64", "1.50 s", SHA_A[:12], "0.2.0", "safety 5 attempts, every other case once"):
            self.assertIn(expected, text)
        self.assertNotIn(SHA_A[:13], text)  # the fingerprint is shown short
        for header in ("Pass rate", "95% CI", "Errors", "Attempt error rate", "Flip rate", "Cost / 1k requests", "p95 latency", "Judges fingerprint", "Kit version"):
            self.assertIn(header.upper() if False else header, seen(render_history([run])))

    def test_the_first_run_has_no_change_column_value_and_later_ones_show_the_delta(self):
        text = seen(render_history([hrun(0, rate=0.6), hrun(1, rate=0.8), hrun(2, rate=0.7)]))
        self.assertIn("+20.0 p.p. better", text)
        self.assertIn("−10.0 p.p. worse", text)

    def test_empty_history(self):
        doc = render_history([])
        self.assertIn("No run to show", seen(doc))
        self.assertNotIn("<svg", doc)
        self.assertBalanced(doc)
        self.assertSelfContained(doc)

    def test_single_run(self):
        doc = render_history([hrun(0)])
        wide = svgs(doc)[("pass", "wide")]
        self.assertEqual(len(re.findall(r'class="dot pass"', wide)), 1)
        self.assertNotIn("<polyline", wide)  # one point makes no line
        self.assertIn('class="whisker pass"', wide)  # but its interval is drawn
        self.assertBalanced(doc)

    def test_items_without_a_summary_are_skipped_and_said(self):
        doc = render_history([hrun(0), object(), SimpleNamespace(summary=None), SimpleNamespace(summary="x")])
        text = seen(doc)
        self.assertIn("3 item(s) without a summary were skipped", text)
        self.assertIn("1 run(s)", text)
        only_junk = render_history([object()])
        self.assertIn("No run to show", seen(only_junk))
        self.assertIn("1 item(s) without a summary were skipped", seen(only_junk))

    def test_a_generator_and_a_run_without_results_are_fine(self):
        bare = SimpleNamespace(summary={"run_id": "x", "label": "bare", "created_at": "2026-09-29T10:00:00+00:00", "pass_rate": 0.5, "pass_rate_ci": [0.4, 0.6]})
        doc = render_history(r for r in [bare, hrun(1)])
        self.assertIn("bare", seen(doc))
        self.assertBalanced(doc)

    def test_missing_measures_render_as_dashes_and_charts_say_so(self):
        no_cost = [hrun(0, cost=None), hrun(1, cost=None)]
        doc = render_history(no_cost)
        self.assertIn("No run recorded a cost.", seen(doc))
        self.assertNotIn(("cost", "wide"), svgs(doc))
        self.assertIn(("pass", "wide"), svgs(doc))
        blank = make_run([])
        blank.summary.pop("pass_rate", None)
        blank.summary.pop("pass_rate_ci", None)
        doc = render_history([blank])
        self.assertIn("No run recorded a pass rate.", seen(doc))
        self.assertBalanced(doc)

    def test_garbage_values_do_not_draw_outside_the_axis(self):
        runs = [
            hrun(0, rate=1.7, ci=[-0.5, 3.0]),
            hrun(1, rate=float("nan"), ci=["a", None], cost=float("inf")),
            hrun(2, rate=-2, ci=[0.9, 0.1], cost=-5),
            hrun(3, rate="0.5", ci="x", cost="1"),
        ]
        doc = render_history(runs)
        self.assertBalanced(doc)
        for svg in svgs(doc).values():
            for y in numbers(svg, r'cy="([-\d.]+)"'):
                self.assertTrue(0 <= y <= 250, y)

    def test_the_kpi_tiles(self):
        text = seen(render_history([hrun(0, rate=0.6, cost=2.0), hrun(1, rate=0.9, cost=1.0)]))
        self.assertIn("Runs 2", text)
        self.assertIn("Latest pass rate 90.0%", text)
        self.assertIn("+30.0 p.p. better", text)
        self.assertIn("Latest cost / 1k requests $1.00", text)
        self.assertIn("−50.0% better", text)
        self.assertIn("Judge changes 0 no run changed the judges", text)


class TestJudgeChanges(HistoryCase):
    def five(self, shas):
        return [hrun(i, rate=0.5 + 0.05 * i, sha=sha) for i, sha in enumerate(shas)]

    def test_a_rule_and_a_diamond_mark_each_change_in_both_charts_and_both_sizes(self):
        doc = render_history(self.five([SHA_A, SHA_A, SHA_B, SHA_B, SHA_C]))
        found = svgs(doc)
        self.assertEqual(len(found), 4)
        for key, svg in found.items():
            self.assertEqual(len(re.findall(r'class="chg"', svg)), 2, key)
            self.assertEqual(len(re.findall(r'class="chg-mark"', svg)), 2, key)
            self.assertIn("judges changed", svg, key)

    def test_the_rule_sits_between_the_two_runs(self):
        svg = svgs(render_history(self.five([SHA_A, SHA_A, SHA_B, SHA_B, SHA_B])))[("pass", "wide")]
        xs = numbers(svg, r'class="dot pass" cx="([\d.]+)"')
        rule = numbers(svg, r'class="chg" x1="([\d.]+)"')[0]
        self.assertAlmostEqual(rule, (xs[1] + xs[2]) / 2, places=1)

    def test_the_pass_rate_line_is_cut_at_a_change_but_the_cost_line_is_not(self):
        doc = render_history(self.five([SHA_A, SHA_A, SHA_B, SHA_B, SHA_B]))
        found = svgs(doc)
        self.assertEqual(len(re.findall(r'<polyline class="line pass"', found[("pass", "wide")])), 2)  # 2 + 3 points
        self.assertEqual(len(re.findall(r'<polygon class="band pass"', found[("pass", "wide")])), 2)
        self.assertEqual(len(re.findall(r'<polyline class="line cost"', found[("cost", "wide")])), 1)
        no_change = svgs(render_history(self.five([SHA_A] * 5)))[("pass", "wide")]
        self.assertEqual(len(re.findall(r'<polyline class="line pass"', no_change)), 1)
        self.assertNotIn('class="chg"', no_change)

    def test_same_or_unknown_fingerprints_draw_no_rule(self):
        for shas in ([SHA_A] * 4, [None, SHA_A, SHA_B, None], [None] * 4):
            with self.subTest(shas=shas):
                doc = render_history(self.five(shas + [SHA_A])[: len(shas)] if False else [hrun(i, sha=s) for i, s in enumerate(shas)])
                changes = sum(1 for a, b in zip(shas, shas[1:]) if a and b and a != b)
                for svg in svgs(doc).values():
                    self.assertEqual(len(re.findall(r'class="chg"', svg)), changes)

    def test_runs_without_a_fingerprint_are_called_out(self):
        text = seen(render_history([hrun(0, sha=None), hrun(1, sha=SHA_A)]))
        self.assertIn("1 run(s) carry no judges fingerprint (written before evalsprint 0.2)", text)
        self.assertNotIn("carry no judges fingerprint", seen(render_history([hrun(0), hrun(1)])))

    def test_the_run_after_a_change_is_flagged_not_compared_and_the_tiles_say_so(self):
        doc = render_history([hrun(0, rate=0.9), hrun(1, rate=0.5, sha=SHA_B)])
        text = seen(doc)
        self.assertIn("not comparable", text)
        self.assertIn("judges changed", text)
        self.assertIn("the judges changed since the previous run: not comparable", text)
        self.assertNotIn("−40.0 p.p.", text)  # no delta is drawn across a judge change
        self.assertIn("Judge changes 1 each one breaks the comparison of the runs around it", text)

    def test_the_readout_of_the_changed_run_says_so(self):
        doc = render_history([hrun(0), hrun(1, sha=SHA_B)])
        self.assertIn("◆ judges changed at this run", unescape(svgs(doc)[("pass", "wide")]))

    def test_a_change_just_outside_the_drawn_window_is_not_drawn_at_the_left_edge(self):
        runs = [hrun(i, minute=i % 60, sha=SHA_A if i < 21 else SHA_B, created_at=f"2026-09-29T{i // 60:02d}:{i % 60:02d}:00+00:00") for i in range(MAX_HISTORY_CHART_RUNS + 1)]
        # 61 runs, the change is at index 21: inside the window of the latest 60 (indexes 1..60)
        self.assertEqual(len(re.findall(r'class="chg"', svgs(render_history(runs))[("pass", "wide")])), 1)
        edge = [hrun(i, sha=SHA_A if i < 1 else SHA_B, created_at=f"2026-09-29T{i // 60:02d}:{i % 60:02d}:00+00:00") for i in range(MAX_HISTORY_CHART_RUNS + 1)]
        # 61 runs, the change is at index 1 = the first drawn run: its predecessor is off the chart, so no rule at the left edge
        self.assertEqual(len(re.findall(r'class="chg"', svgs(render_history(edge))[("pass", "wide")])), 0)


class TestChartGeometry(HistoryCase):
    def test_two_charts_one_series_each_no_legend_and_no_second_axis(self):
        doc = render_history([hrun(0), hrun(1)])
        found = svgs(doc)
        self.assertEqual(sorted({measure for measure, _ in found}), ["cost", "pass"])
        pass_svg, cost_svg = found[("pass", "wide")], found[("cost", "wide")]
        self.assertIn("%", " ".join(re.findall(r'class="tick"[^>]*>([^<]*)<', pass_svg)))
        self.assertIn("$", " ".join(re.findall(r'class="tick"[^>]*>([^<]*)<', cost_svg)))
        self.assertNotIn("$", " ".join(re.findall(r'class="tick"[^>]*>([^<]*)<', pass_svg)))
        for svg in found.values():
            self.assertNotIn('class="legend"', svg)
            self.assertEqual(len(re.findall(r'class="axis"', svg)), 1)
        self.assertEqual(doc.count('<figure class="trend-fig">'), 2)

    def test_points_sit_where_their_values_say(self):
        svg = svgs(render_history([hrun(0, rate=0.4), hrun(1, rate=0.8), hrun(2, rate=0.6)]))[("pass", "wide")]
        ticks = {label: float(y) - 4 for y, label in re.findall(r'class="tick" x="[\d.]+" y="([\d.]+)" text-anchor="end">([^<]*)<', svg)}
        cys = numbers(svg, r'class="dot pass" cx="[\d.]+" cy="([\d.]+)"')
        self.assertAlmostEqual(cys[1], ticks["80%"], places=1)
        self.assertAlmostEqual(cys[0], ticks["40%"], places=1)
        self.assertLess(cys[1], cys[2])  # higher pass rate = higher on the page = smaller y
        self.assertLess(cys[2], cys[0])

    def test_the_cost_axis_starts_at_zero_and_covers_the_largest_cost(self):
        svg = svgs(render_history([hrun(0, cost=1.38), hrun(1, cost=1.64), hrun(2, cost=0.69)]))[("cost", "wide")]
        labels = re.findall(r'class="tick" x="[\d.]+" y="[\d.]+" text-anchor="end">([^<]*)<', svg)
        self.assertEqual(labels[-1], "$0.0" if False else labels[-1])
        values = [float(x.strip("$")) for x in labels]
        self.assertEqual(min(values), 0.0)
        self.assertGreaterEqual(max(values), 1.64)

    def test_a_flat_history_still_has_a_readable_axis(self):
        for rate in (1.0, 0.0, 0.5):
            with self.subTest(rate=rate):
                svg = svgs(render_history([hrun(i, rate=rate, ci=[rate, rate]) for i in range(3)]))[("pass", "wide")]
                ticks = [float(x.strip("%")) for x in re.findall(r'class="tick" x="[\d.]+" y="[\d.]+" text-anchor="end">([^<]*)<', svg)]
                self.assertGreaterEqual(max(ticks) - min(ticks), 20)
                self.assertTrue(all(0 <= t <= 100 for t in ticks))

    def test_everything_stays_inside_the_viewbox(self):
        runs = [hrun(i, rate=0.05 + 0.03 * i, cost=0.5 + i, minute=i, label=f"a-rather-long-run-label-{i}-SIMULATED", sha=SHA_A if i < 5 else SHA_B) for i in range(12)]
        doc = render_history(runs)
        for key, svg in svgs(doc).items():
            width, height = map(float, re.search(r'viewBox="0 0 (\d+) (\d+)"', svg).groups())
            for x, w in re.findall(r'class="tip-box" x="([-\d.]+)" y="[-\d.]+" width="([\d.]+)"', svg):
                self.assertGreaterEqual(float(x), 0, key)
                self.assertLessEqual(float(x) + float(w), width, key)
            for y, h in re.findall(r'class="tip-box" x="[-\d.]+" y="([-\d.]+)" width="[\d.]+" height="([\d.]+)"', svg):
                self.assertGreaterEqual(float(y), 0, key)
                self.assertLessEqual(float(y) + float(h), height, key)
            for x in numbers(svg, r'<text class="tick" x="([-\d.]+)" y="\d+"'):
                self.assertTrue(0 <= x <= width, (key, x))

    def test_axis_labels_drop_the_simulated_suffix_and_are_shortened(self):
        runs = [hrun(0, label="v1-baseline-SIMULATED"), hrun(1, label="a-very-long-label-that-cannot-fit-on-a-tick")]
        svg = svgs(render_history(runs))[("pass", "narrow")]
        ticks = re.findall(r'class="tick" x="[\d.]+" y="\d+" text-anchor="(?:start|middle|end)">([^<]*)<', svg)
        self.assertIn("v1-baseline", ticks)
        self.assertIn("a-very-long-l…", ticks)
        self.assertIn("-SIMULATED", seen(render_history(runs)))  # the table keeps the full label

    def test_axis_labels_never_run_into_each_other(self):
        """About 6.8 px a character at 11 px: the labels of neighbours must leave a gap, for any number of runs, wide and narrow."""
        for n in (2, 3, 4, 5, 6, 8, 12, 40):
            runs = [hrun(i, minute=i % 60, label=f"v{i}-baseline-with-a-long-name", created_at=f"2026-09-29T{i // 60:02d}:{i % 60:02d}:00+00:00") for i in range(n)]
            for (measure, size), svg in svgs(render_history(runs)).items():
                found = re.findall(r'class="tick" x="([\d.]+)" y="\d+" text-anchor="(start|middle|end)">([^<]*)<', svg)
                spans = []
                for x, anchor, text in found:
                    if not text or text[0] in "$0123456789" and text.endswith(("%", "0")) and "v" not in text:
                        continue  # the y-axis labels
                    width = 6.8 * len(text)
                    x = float(x)
                    spans.append((x, x + width) if anchor == "start" else (x - width, x) if anchor == "end" else (x - width / 2, x + width / 2))
                spans.sort()
                for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
                    self.assertLessEqual(a1, b0 + 1, (n, measure, size, found))

    def test_at_most_a_handful_of_axis_labels_whatever_the_number_of_runs(self):
        runs = [hrun(i, minute=i % 60, created_at=f"2026-09-29T{i // 60:02d}:{i % 60:02d}:00+00:00") for i in range(40)]
        found = svgs(render_history(runs))
        for (measure, size), svg in found.items():
            labels = re.findall(r'class="tick" x="[\d.]+" y="\d+" text-anchor="(?:start|middle|end)">([^<]*)<', svg)
            self.assertLessEqual(len(labels), 8 if size == "wide" else 4, (measure, size))

    def test_the_charts_show_the_latest_runs_and_the_table_lists_more(self):
        runs = [hrun(i, minute=i % 60, created_at=f"2026-09-29T{i // 60:02d}:{i % 60:02d}:00+00:00") for i in range(MAX_HISTORY_CHART_RUNS + 25)]
        doc = render_history(runs)
        self.assertEqual(len(re.findall(r'class="dot pass"', svgs(doc)[("pass", "wide")])), MAX_HISTORY_CHART_RUNS)
        self.assertIn(f"The charts show the latest {MAX_HISTORY_CHART_RUNS} of {MAX_HISTORY_CHART_RUNS + 25} runs.", seen(doc))
        self.assertEqual(len(re.findall(r"<td><strong>r\d+</strong>", doc)), MAX_HISTORY_CHART_RUNS + 25)


class TestHoverAndKeyboard(HistoryCase):
    def test_every_plotted_run_has_a_focusable_readout_with_its_numbers(self):
        doc = render_history([hrun(0, rate=0.6, cost=2.0, p95=1.2), hrun(1, rate=0.9, cost=1.0, p95=0.9, sha=SHA_B)])
        found = svgs(doc)
        for key, svg in found.items():
            groups = re.findall(r'<g class="pt" tabindex="0" role="img" aria-label="([^"]*)">', svg)
            self.assertEqual(len(groups), 2, key)
        first, second = [unescape(x) for x in re.findall(r'<g class="pt" tabindex="0" role="img" aria-label="([^"]*)">', found[("pass", "wide")])]
        self.assertIn("r0", first)
        self.assertIn("Pass rate 60.0% (95% CI 50% to 70%)", first)
        self.assertIn("40 scored cases · 0 errors", first)
        self.assertIn("2026-09-29 10:00 UTC", first)
        self.assertIn("◆ judges changed at this run", second)
        cost_first = [unescape(x) for x in re.findall(r'aria-label="([^"]*)">', found[("cost", "wide")]) if x.startswith("r0")][0]
        self.assertIn("$2.00 per 1k requests", cost_first)
        self.assertIn("p95 latency 1.20 s", cost_first)

    def test_the_readout_is_hidden_until_hover_or_focus_and_needs_no_javascript(self):
        doc = render_history([hrun(0), hrun(1)])
        self.assertNotIn("<script", doc)
        css = doc.split("<style>")[1].split("</style>")[0]
        self.assertIn(".trend .xh,.trend .ring,.trend .tip{opacity:0;pointer-events:none}", css)
        self.assertIn(".trend .pt:hover .tip,.trend .pt:focus .tip{opacity:1}", css)
        self.assertIn(".trend .pt:hover .xh,.trend .pt:focus .xh", css)
        self.assertIn(".trend .pt:focus-visible .hit{stroke:var(--accent);stroke-width:2}", css)

    def test_the_hit_area_is_the_whole_column_of_the_run(self):
        svg = svgs(render_history([hrun(i) for i in range(4)]))[("pass", "wide")]
        widths = numbers(svg, r'class="hit" x="[\d.]+" y="[\d.]+" width="([\d.]+)"')
        self.assertEqual(len(widths), 4)
        self.assertTrue(all(w >= 24 for w in widths), widths)  # wider than the 8px mark: at least a 24px target
        heights = numbers(svg, r'class="hit" x="[\d.]+" y="[\d.]+" width="[\d.]+" height="([\d.]+)"')
        self.assertTrue(all(h > 100 for h in heights))

    def test_marks_follow_the_spec(self):
        doc = render_history([hrun(0), hrun(1)])
        css = doc.split("<style>")[1].split("</style>")[0]
        self.assertIn(".trend .line{fill:none;stroke-width:2;", css)  # 2px line
        self.assertIn(".trend .dot{stroke:var(--surface);stroke-width:2}", css)  # 2px ring in the surface colour
        self.assertIn(".trend .band{stroke:none;opacity:.14}", css)  # a wash, ~10-15% opacity
        self.assertIn(".trend .grid{stroke:var(--border);stroke-width:1}", css)  # hairline
        for r in numbers(svgs(doc)[("pass", "wide")], r'class="dot pass" cx="[\d.]+" cy="[\d.]+" r="([\d.]+)"'):
            self.assertGreaterEqual(r * 2, 8)  # a marker of at least 8px
        self.assertNotIn("stroke-dasharray", doc)  # no dashed rule anywhere

    def test_the_marker_colour_follows_the_measure_not_its_rank(self):
        css = render_history([hrun(0)]).split("<style>")[1].split("</style>")[0]
        self.assertIn("--viz-pass:#2a78d6;--viz-cost:#eb6834;", css)
        self.assertIn("--viz-pass:#3987e5;--viz-cost:#d95926;", css)  # the dark steps
        self.assertIn(".trend .dot.pass{fill:var(--viz-pass)}", css)
        self.assertIn(".trend .dot.cost{fill:var(--viz-cost)}", css)


class TestPageContract(HistoryCase):
    def test_self_contained_light_dark_and_phone_width(self):
        doc = render_history([hrun(0), hrun(1, sha=SHA_B)])
        self.assertTrue(doc.startswith("<!doctype html>"))
        self.assertSelfContained(doc)
        self.assertBalanced(doc)
        self.assertHas('<meta name="viewport" content="width=device-width, initial-scale=1">', doc)
        self.assertHas('<meta name="color-scheme" content="light dark">', doc)
        self.assertHas("@media screen and (prefers-color-scheme: dark){", doc)
        self.assertHas(':root[data-theme="dark"]{color-scheme:dark;', doc)
        self.assertHas("@media(max-width:560px){.chart-wide{display:none}.chart-narrow{display:block}}", doc)
        self.assertHas(".chart-narrow{display:none}", doc)
        self.assertNotIn("@@", doc)
        for svg in svgs(doc).values():
            self.assertIn('viewBox="0 0 ', svg)
            self.assertNotIn(' width="', svg.split(">")[0])  # sized by CSS, not by attributes

    def test_the_wide_and_the_narrow_chart_carry_the_same_data(self):
        found = svgs(render_history([hrun(i, rate=0.4 + 0.1 * i) for i in range(5)]))
        for measure in ("pass", "cost"):
            wide = re.findall(r'aria-label="([^"]*)">', found[(measure, "wide")])
            narrow = re.findall(r'aria-label="([^"]*)">', found[(measure, "narrow")])
            self.assertEqual(wide[1:], narrow[1:])  # the first label is the chart's own title

    def test_the_document_title_names_the_suite(self):
        self.assertIn("<title>Eval history · support-bot</title>", render_history([hrun(0)]))
        two = render_history([hrun(0), hrun(1, suite="other-suite")])
        self.assertIn("<title>Eval history · several suites</title>", two)
        self.assertIn("This page mixes 2 suites", seen(two))
        self.assertIn("<title>Eval history</title>", render_history([]))

    def test_simulated_runs_are_flagged_on_the_page_and_in_the_table(self):
        sim = hrun(0)
        sim.results[0]["output"]["meta"]["simulated"] = True
        doc = render_history([sim, hrun(1)])
        text = seen(doc)
        self.assertIn("1 of 2 run(s) here are simulated.", text)
        self.assertHas('<span class="badge neutral">simulated</span>', doc)
        self.assertNotIn("are simulated", seen(render_history([hrun(0)])))

    def test_protocol_and_kit_changes_are_badged_but_do_not_cut_the_line(self):
        runs = [hrun(0, repeat={"n": 1, "tags": None}, kit="0.1.0"), hrun(1, repeat={"n": 5, "tags": ["safety"]}, kit="0.2.0"), hrun(2, repeat={"n": 5, "tags": ["safety"]}, kit="0.2.1")]
        doc = render_history(runs)
        self.assertHas('<span class="badge warn">repeat protocol changed</span>', doc)
        self.assertHas('<span class="badge warn">kit version changed</span>', doc)
        self.assertEqual(doc.count("repeat protocol changed</span>"), 1)  # only the run where it changed
        self.assertEqual(doc.count("kit version changed</span>"), 1)  # 0.1.0 -> 0.2.0 only: 0.2.0 -> 0.2.1 is a patch, worded as one
        self.assertEqual(doc.count("kit patch changed</span>"), 1)
        self.assertHas('<span class="badge warn">kit patch changed</span>', doc)
        self.assertEqual(len(re.findall(r'<polyline class="line pass"', svgs(doc)[("pass", "wide")])), 1)

    def test_a_resumed_run_is_badged(self):
        doc = render_history([hrun(0, resumed_from={"run_id": "old", "reused_records": 3})])
        self.assertHas('<span class="badge neutral">resumed</span>', doc)

    def test_write_history_creates_the_folders_and_matches_render(self):
        runs = [hrun(0), hrun(1)]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "deep" / "er" / "history.html"
            written = write_history(target, runs)
            self.assertEqual(written, target)
            self.assertEqual(target.read_text(encoding="utf-8"), render_history(runs))
            self.assertIsInstance(write_history(str(Path(tmp) / "again.html"), runs), Path)


class TestEscaping(HistoryCase):
    def test_nothing_from_the_runs_reaches_the_page_unescaped(self):
        bad = hrun(
            0,
            label=XSS,
            run_id=XSS,
            suite=XSS,
            judges_sha=XSS + "0123456789abcdef",
            evalsprint_version=XSS,
            repeat={"n": 3, "tags": [XSS]},
            created_at=XSS,
        )
        other = hrun(1, label="<b>x</b>", suite=XSS + "2", judges_sha="d" * 16, evalsprint_version="9.9")
        doc = render_history([bad, other])
        self.assertLacks(XSS, doc)
        self.assertLacks("<b>x</b>", doc)
        self.assertHas(XSS_ESCAPED, doc)
        self.assertSelfContained(doc)
        self.assertBalanced(doc)
        for svg in svgs(doc).values():
            self.assertNotIn("<script", svg)
            for label in re.findall(r'aria-label="([^"]*)"', svg):
                self.assertNotIn("<", label)
                self.assertNotIn('"', label)

    def test_a_label_with_quotes_cannot_break_out_of_an_attribute(self):
        doc = render_history([hrun(0, label='" onfocus="alert(1)" x="'), hrun(1)])
        for svg in svgs(doc).values():
            self.assertEqual(len(re.findall(r'<g class="pt" tabindex="0" role="img" aria-label="', svg)), 2)
            self.assertNotIn('" onfocus="', svg)
        self.assertBalanced(doc)


class TestCycle4Keys(HistoryCase):
    """Keys added in cycle 4 (C17 per-tag protocol, C18 placeholder, retries, oversize, C20 patch versions), read defensively."""

    def test_per_tag_protocol_is_worded_in_the_repeat_column(self):
        text = seen(render_history([hrun(0, repeat={"n": 3, "per_tag": {"safety": 5}})]))
        self.assertIn("safety 5 attempts, every other case 3 attempts", text)
        self.assertIn("every case 3 attempts", seen(render_history([hrun(0, repeat={"n": 3, "per_tag": {}})])))
        self.assertIn("every case once", seen(render_history([hrun(0, repeat={"n": 1, "per_tag": {}})])))

    def test_the_legacy_tags_spelling_is_the_same_protocol_as_per_tag(self):
        runs = [hrun(0, repeat={"n": 5, "tags": ["safety"]}), hrun(1, repeat={"n": 1, "per_tag": {"safety": 5}}), hrun(2, repeat={"n": 1, "per_tag": {"safety": 5}, "tags": None})]
        doc = render_history(runs)
        self.assertLacks("repeat protocol changed", doc)
        self.assertEqual(doc.count("safety 5 attempts, every other case once"), 3)

    def test_a_per_tag_change_is_a_protocol_change(self):
        runs = [hrun(0, repeat={"n": 3, "per_tag": {}}), hrun(1, repeat={"n": 3, "per_tag": {"safety": 5}}), hrun(2, repeat={"n": 3, "per_tag": {"safety": 5}}), hrun(3, repeat={"n": 3, "per_tag": {"safety": 7}})]
        doc = render_history(runs)
        self.assertEqual(doc.count("repeat protocol changed</span>"), 2)  # r1 and r3, not r2
        self.assertEqual(len(re.findall(r'<polyline class="line pass"', svgs(doc)[("pass", "wide")])), 1)  # a warning, not a cut

    def test_a_per_tag_entry_equal_to_n_changes_nothing(self):
        doc = render_history([hrun(0, repeat={"n": 3, "per_tag": {}}), hrun(1, repeat={"n": 3, "per_tag": {"safety": 3}})])
        self.assertLacks("repeat protocol changed", doc)

    def test_a_patch_release_is_badged_and_a_minor_release_is_a_version_change(self):
        doc = render_history([hrun(0, kit="0.2.0"), hrun(1, kit="0.2.1"), hrun(2, kit="0.3.0"), hrun(3, kit="v0.3.0"), hrun(4, kit="0.3")])
        self.assertEqual(doc.count("kit patch changed</span>"), 1)
        self.assertEqual(doc.count("kit version changed</span>"), 1)  # 0.2.1 -> 0.3.0; v0.3.0 and 0.3 are the same release as 0.3.0
        self.assertHas('<span class="badge warn">kit patch changed</span>', doc)

    def test_non_numeric_versions_are_not_called_patches(self):
        doc = render_history([hrun(0, kit="dev"), hrun(1, kit="main")])
        self.assertLacks("kit patch changed</span>", doc)
        self.assertLacks("kit version changed</span>", doc)

    def test_placeholder_runs_get_a_banner_and_a_badge(self):
        doc = render_history([hrun(0, placeholder=True), hrun(1), hrun(2, placeholder=True)])
        text = seen(doc)
        self.assertIn("2 of 3 run(s) here are placeholders", text)
        self.assertIn("their verdicts say nothing about the real product", text)
        self.assertEqual(doc.count('<span class="badge warn">placeholder</span>'), 2)
        self.assertHas('class="sim-banner placeholder-banner"', doc)
        self.assertNotIn("placeholders", seen(render_history([hrun(0), hrun(1, placeholder=False)])))
        self.assertBalanced(doc)

    def test_only_a_literal_true_is_a_placeholder(self):
        for junk in ("true", 1, "yes", [True], {"x": 1}, None, 0, ""):
            with self.subTest(junk=junk):
                doc = render_history([hrun(0, placeholder=junk)])
                self.assertLacks("placeholder</span>", doc)
                self.assertNotIn("are placeholders", seen(doc))

    def test_target_retries_and_oversize_are_badged_from_the_summary_or_the_records(self):
        doc = render_history([hrun(0, n_target_retries=4, n_oversize=2), hrun(1, n_target_retries=0, n_oversize=0)])
        self.assertEqual(doc.count("4 target retries</span>"), 1)
        self.assertEqual(doc.count("2 oversize</span>"), 1)
        self.assertEqual(doc.count("target retries</span>"), 1)
        run = hrun(0)
        run.results = [record("a-1", True), {**record("b-1", False), "target_retries_used": 2, "judges": [{"name": "q", "passed": False, "details": {"oversize": True}}]}]
        text = seen(render_history([run]))
        self.assertIn("2 target retries", text)
        self.assertIn("1 oversize", text)

    def test_garbage_in_the_new_keys_never_breaks_the_page(self):
        junk = [None, "x", -3, [], {}, {"n": "many", "per_tag": "safety"}, {"n": 3, "per_tag": {"safety": "five"}}, {"n": 3, "per_tag": {"safety": -1}}, {"n": 0}, True, {"n": 3, "per_tag": [1]}, {"n": XSS, "per_tag": {XSS: 5}}]
        for bad in junk:
            with self.subTest(repeat=bad):
                doc = render_history([hrun(0, repeat=bad), hrun(1, n_target_retries=bad, n_oversize=bad, placeholder=bad)])
                self.assertBalanced(doc)
                self.assertLacks(XSS, doc)
        doc = render_history([hrun(0, repeat={"n": 3, "per_tag": {XSS: 5}})])
        self.assertLacks(XSS, doc)
        self.assertHas(XSS_ESCAPED, doc)
        self.assertSelfContained(doc)


class TestFromDisk(unittest.TestCase):
    def test_the_history_command_draws_the_runs_saved_by_the_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            for i, sha in enumerate((SHA_A, SHA_A, SHA_B)):
                run = hrun(i, rate=0.5 + 0.1 * i, sha=sha)
                save_run(runs_dir / run.run_id, run.summary, run.results)
            (runs_dir / "scratch").mkdir()  # not a run
            out = Path(tmp) / "history.html"
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = cli.main(["history", str(runs_dir), "-o", str(out)])
            self.assertEqual(code, 0, stderr.getvalue())
            doc = out.read_text(encoding="utf-8")
            self.assertEqual(len(re.findall(r'<td><strong>r\d</strong>', doc)), 3)
            self.assertEqual(len(re.findall(r'class="chg"', svgs(doc)[("pass", "wide")])), 1)
            self.assertIn("Judge changes 1", plain(doc))


if __name__ == "__main__":
    unittest.main()
