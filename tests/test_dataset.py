import json
import tempfile
import unittest
from pathlib import Path

from evalsprint.dataset import describe, filter_cases, load_cases
from evalsprint.types import Case


class LoadCasesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "cases.jsonl"

    def write(self, *lines: str) -> Path:
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.path

    def test_loads_all_known_fields(self):
        row = {
            "id": "a-1",
            "input": "hi",
            "reference": "ref",
            "expected_behavior": "answer",
            "tags": ["x", "y"],
            "checks": [{"type": "contains", "values": ["z"]}],
            "metadata": {"source": "prod"},
        }
        (case,) = load_cases(self.write(json.dumps(row)))
        self.assertEqual(case.id, "a-1")
        self.assertEqual(case.input, "hi")
        self.assertEqual(case.reference, "ref")
        self.assertEqual(case.expected_behavior, "answer")
        self.assertEqual(case.tags, ["x", "y"])
        self.assertEqual(case.checks, [{"type": "contains", "values": ["z"]}])
        self.assertEqual(case.metadata, {"source": "prod"})

    def test_defaults_for_optional_fields(self):
        (case,) = load_cases(self.write('{"id": "a", "input": "q"}'))
        self.assertIsNone(case.reference)
        self.assertIsNone(case.expected_behavior)
        self.assertEqual((case.tags, case.checks, case.metadata), ([], [], {}))

    def test_null_optional_lists_become_empty(self):
        (case,) = load_cases(self.write('{"id": "a", "input": "q", "tags": null, "checks": null, "metadata": null}'))
        self.assertEqual((case.tags, case.checks, case.metadata), ([], [], {}))

    def test_blank_lines_and_comment_lines_are_skipped(self):
        cases = load_cases(self.write("// header comment", "", '{"id": "a", "input": "1"}', "   ", "  // indented comment", '{"id": "b", "input": "2"}'))
        self.assertEqual([c.id for c in cases], ["a", "b"])

    def test_order_is_file_order(self):
        cases = load_cases(self.write(*(json.dumps({"id": f"c{i}", "input": i}) for i in (3, 1, 2))))
        self.assertEqual([c.id for c in cases], ["c3", "c1", "c2"])

    def test_unknown_fields_land_in_metadata(self):
        (case,) = load_cases(self.write('{"id": "a", "input": "q", "must_include": ["refund"], "priority": 2, "metadata": {"src": "logs"}}'))
        self.assertEqual(case.metadata, {"src": "logs", "must_include": ["refund"], "priority": 2})

    def test_unknown_field_wins_over_same_key_in_metadata(self):
        (case,) = load_cases(self.write('{"id": "a", "input": "q", "k": "top", "metadata": {"k": "nested"}}'))
        self.assertEqual(case.metadata["k"], "top")

    def test_dict_input_and_non_string_id(self):
        (case,) = load_cases(self.write('{"id": 42, "input": {"question": "hi", "user": {"plan": "pro"}}}'))
        self.assertEqual(case.id, "42")
        self.assertEqual(case.input["user"]["plan"], "pro")

    def test_unicode_is_preserved(self):
        (case,) = load_cases(self.write(json.dumps({"id": "a", "input": "¿Dónde está mi pedido? 日本語 🚚"}, ensure_ascii=False)))
        self.assertEqual(case.input, "¿Dónde está mi pedido? 日本語 🚚")

    def test_duplicate_ids_report_line_and_id(self):
        path = self.write('{"id": "a", "input": "1"}', "", '{"id": "b", "input": "2"}', '{"id": "a", "input": "3"}')
        with self.assertRaises(ValueError) as ctx:
            load_cases(path)
        msg = str(ctx.exception)
        self.assertIn(":4:", msg)  # line number counts the blank line
        self.assertIn("duplicate", msg)
        self.assertIn("'a'", msg)

    def test_invalid_json_reports_line_number(self):
        path = self.write("// comment", '{"id": "a", "input": "1"}', "", '{"id": "b", "input": ')
        with self.assertRaises(ValueError) as ctx:
            load_cases(path)
        self.assertIn(f"{path}:4:", str(ctx.exception))

    def test_missing_id_or_input_reports_line_number(self):
        for bad in ('{"input": "no id"}', '{"id": "no-input"}'):
            with self.subTest(bad=bad):
                path = self.write('{"id": "ok", "input": "1"}', bad)
                with self.assertRaises(ValueError) as ctx:
                    load_cases(path)
                self.assertIn(":2:", str(ctx.exception))
                self.assertIn("'id' and 'input'", str(ctx.exception))

    def test_non_object_lines_are_value_errors_with_line_number(self):
        # regression: a bare number used to escape as TypeError
        for bad in ("42", "null", "true", "[1, 2]", '"just a string"'):
            with self.subTest(bad=bad):
                path = self.write('{"id": "ok", "input": "1"}', bad)
                with self.assertRaises(ValueError) as ctx:
                    load_cases(path)
                self.assertIn(":2:", str(ctx.exception))

    def test_empty_file_and_comment_only_file(self):
        for lines in ([""], ["// nothing here", ""]):
            with self.subTest(lines=lines):
                with self.assertRaises(ValueError) as ctx:
                    load_cases(self.write(*lines))
                self.assertIn("no cases", str(ctx.exception))

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_cases(Path(self._tmp.name) / "nope.jsonl")

    def test_accepts_str_path(self):
        self.write('{"id": "a", "input": "q"}')
        self.assertEqual(len(load_cases(str(self.path))), 1)

    def test_utf8_bom_is_tolerated(self):
        # regression: files saved by Excel / Notepad start with a BOM and used to fail on line 1
        self.path.write_bytes(b'\xef\xbb\xbf{"id": "a", "input": "1"}\r\n{"id": "b", "input": "2"}\r\n')
        self.assertEqual([c.id for c in load_cases(self.path)], ["a", "b"])

    def test_windows_line_endings(self):
        self.path.write_bytes(b'{"id": "a", "input": "1"}\r\n{"id": "b", "input": "2"}\r\n')
        self.assertEqual([c.id for c in load_cases(self.path)], ["a", "b"])


def sample_cases() -> list[Case]:
    return [
        Case("a", "q", tags=["billing", "core"], reference="r", expected_behavior="answer"),
        Case("b", "q", tags=["billing"], expected_behavior="answer"),
        Case("c", "q", tags=["shipping"], expected_behavior="abstain"),
        Case("d", "q", tags=[]),
        Case("e", "q", tags=["core", "refund"], reference="r"),
    ]


class FilterTests(unittest.TestCase):
    def ids(self, cases):
        return [c.id for c in cases]

    def test_no_filters_returns_everything(self):
        self.assertEqual(self.ids(filter_cases(sample_cases())), list("abcde"))

    def test_tags_match_any(self):
        self.assertEqual(self.ids(filter_cases(sample_cases(), tags=["billing"])), ["a", "b"])
        self.assertEqual(self.ids(filter_cases(sample_cases(), tags=["shipping", "refund"])), ["c", "e"])

    def test_unknown_tag_selects_nothing(self):
        self.assertEqual(filter_cases(sample_cases(), tags=["nope"]), [])

    def test_ids(self):
        self.assertEqual(self.ids(filter_cases(sample_cases(), ids=["e", "b", "zzz"])), ["b", "e"])  # dataset order, not request order

    def test_limit(self):
        self.assertEqual(self.ids(filter_cases(sample_cases(), limit=2)), ["a", "b"])
        self.assertEqual(len(filter_cases(sample_cases(), limit=99)), 5)

    def test_falsy_limit_means_no_limit(self):
        self.assertEqual(len(filter_cases(sample_cases(), limit=0)), 5)
        self.assertEqual(len(filter_cases(sample_cases(), limit=None)), 5)

    def test_filters_combine_with_and_semantics_and_limit_applies_last(self):
        self.assertEqual(self.ids(filter_cases(sample_cases(), tags=["core"], ids=["e"])), ["e"])
        self.assertEqual(self.ids(filter_cases(sample_cases(), tags=["core"], limit=1)), ["a"])

    def test_original_list_not_mutated(self):
        cases = sample_cases()
        filter_cases(cases, tags=["billing"], limit=1)
        self.assertEqual(len(cases), 5)


class DescribeTests(unittest.TestCase):
    def test_describe(self):
        d = describe(sample_cases())
        self.assertEqual(d["n_cases"], 5)
        self.assertEqual(d["tags"], {"billing": 2, "core": 2, "shipping": 1, "refund": 1})
        self.assertEqual(list(d["tags"])[:2], ["billing", "core"])  # most common first
        self.assertEqual(d["expected_behavior"], {"answer": 2, "abstain": 1, "unspecified": 2})
        self.assertEqual(d["with_reference"], 2)

    def test_case_round_trip(self):
        original = Case("a", {"q": 1}, reference="r", expected_behavior="answer", tags=["t"], checks=[{"type": "regex", "pattern": "x"}], metadata={"k": 1})
        self.assertEqual(Case.from_dict(original.to_dict()), original)


class SchemaValidationTests(unittest.TestCase):
    """The schema is checked when the case is loaded, so `describe` catches what used to surface mid-run (or never)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "cases.jsonl"

    def load(self, *rows, **kw):
        self.path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return load_cases(self.path, **kw)

    def assertRejected(self, row, contains, **kw):
        with self.assertRaises(ValueError) as ctx:
            self.load({"id": "ok", "input": "1"}, row, **kw)
        message = str(ctx.exception)
        self.assertIn(":2:", message)  # the line of the bad row
        self.assertIn(contains, message)
        return message

    def test_tags_must_be_a_list_of_strings(self):
        message = self.assertRejected({"id": "a", "input": "x", "tags": "safety"}, "'tags' must be a list")
        self.assertIn('["safety"]', message)  # says how to fix it (it used to become the tags s, a, f, e, t, y)
        self.assertRejected({"id": "a", "input": "x", "tags": ["a", 1]}, "'tags' must be a list")
        self.assertRejected({"id": "a", "input": "x", "tags": ["a", ""]}, "'tags' must be a list")
        self.assertRejected({"id": "a", "input": "x", "tags": {"a": 1}}, "'tags' must be a list")

    def test_id_must_be_a_non_empty_string(self):
        for bad in ("", "   ", None, True, ["a"], {"k": 1}, 1.5):
            with self.subTest(id=bad):
                self.assertRejected({"id": bad, "input": "x"}, "'id'")

    def test_integer_ids_are_converted(self):
        (case,) = self.load({"id": 42, "input": "x"})
        self.assertEqual(case.id, "42")

    def test_input_may_not_be_null(self):
        self.assertRejected({"id": "a", "input": None}, "'input' must not be null")

    def test_other_field_types(self):
        self.assertRejected({"id": "a", "input": "x", "expected_behavior": 3}, "'expected_behavior' must be a string")
        self.assertRejected({"id": "a", "input": "x", "metadata": "oops"}, "'metadata' must be an object")
        self.assertRejected({"id": "a", "input": "x", "checks": {"type": "contains", "values": ["x"]}}, "'checks' must be a list")

    def test_check_types_and_settings_are_validated_at_load_time(self):
        message = self.assertRejected({"id": "a", "input": "x", "checks": [{"type": "containz", "values": ["x"]}]}, "check type 'containz'")
        self.assertIn("did you mean 'contains'", message)
        self.assertRejected({"id": "a", "input": "x", "checks": [{"values": ["x"]}]}, "needs a 'type'")
        self.assertRejected({"id": "a", "input": "x", "checks": ["contains"]}, "a check must be an object")
        self.assertRejected({"id": "a", "input": "x", "checks": [{"type": "contains"}]}, "missing required setting(s) 'values'")
        message = self.assertRejected({"id": "a", "input": "x", "checks": [{"type": "contains", "value": ["x"]}]}, "unknown setting 'value'")
        self.assertIn("did you mean 'values'", message)
        self.assertRejected({"id": "a", "input": "x", "checks": [{"type": "regex", "pattern": "(unclosed"}]}, "check #0 (regex)")
        self.assertRejected({"id": "a", "input": "x", "checks": [{"type": "max_chars", "value": "lots"}]}, "'value' must be a whole number")
        self.assertRejected({"id": "a", "input": "x", "checks": [{"type": "contains", "values": ["x"], "mode": "alll"}]}, "'mode' must be 'any' or 'all'")
        self.assertRejected({"id": "a", "input": "x", "checks": [{"type": "contains", "values": []}]}, "'values'")
        self.assertRejected({"id": "a", "input": "x", "checks": [{"type": "contains", "values": ["x"], "when_tags": ["t"]}]}, "unknown setting 'when_tags'")  # meaningless per case
        self.assertRejected({"id": "a", "input": "x", "checks": [{"type": "contains", "values": ["x"]}, {"type": "nope"}]}, "check #1")

    def test_every_check_type_can_be_declared(self):
        checks = [
            {"type": "contains", "values": ["a"], "mode": "all", "case_insensitive": False},
            {"type": "not_contains", "values": "b"},
            {"type": "regex", "pattern": "^a", "flags": "im", "must_match": False},
            {"type": "max_chars", "value": 5},
            {"type": "min_chars", "value": "3"},
            {"type": "json_valid", "required_keys": ["a"]},
            {"type": "exact_match"},
        ]
        (case,) = self.load({"id": "a", "input": "x", "checks": checks})
        self.assertEqual(len(case.checks), 7)

    def test_expected_behavior_is_checked_against_configured_behaviors(self):
        rows = [{"id": "a", "input": "x", "expected_behavior": "answer"}, {"id": "b", "input": "x"}, {"id": "c", "input": "x", "expected_behavior": "refuse"}]
        self.assertEqual(len(self.load(*rows, behaviors=["answer", "abstain", "refuse"])), 3)  # a missing behavior is fine
        self.assertEqual(len(self.load(*rows)), 3)  # not configured: anything goes
        message = self.assertRejected({"id": "d", "input": "x", "expected_behavior": "abstian"}, "expected_behavior 'abstian'", behaviors=["answer", "abstain", "refuse"])
        self.assertIn("did you mean 'abstain'", message)
        self.assertIn("case 'd'", message)

    def test_lone_surrogates_are_replaced_everywhere_in_a_case(self):
        raw = (
            '{"id": "s\\ud83d", "input": {"q": "hello \\ud83d"}, "reference": "r\\ude00", "expected_behavior": "answer", '
            '"tags": ["t\\ud800"], "checks": [{"type": "contains", "values": ["x\\udc00"]}], "metadata": {"k": "m\\ud83d"}, "extra": "e\\ud83d"}'
        )
        (case,) = self.load(raw)
        self.assertEqual(case.id, "s\ufffd")
        self.assertEqual(case.input, {"q": "hello \ufffd"})
        self.assertEqual((case.reference, case.tags), ("r\ufffd", ["t\ufffd"]))
        self.assertEqual(case.checks[0]["values"], ["x\ufffd"])
        self.assertEqual(case.metadata, {"k": "m\ufffd", "extra": "e\ufffd"})
        json.dumps(case.to_dict(), ensure_ascii=False).encode("utf-8")  # storable

    def test_valid_surrogate_pairs_are_left_alone(self):
        (case,) = self.load('{"id": "a", "input": "truck \\ud83d\\ude9a"}')
        self.assertEqual(case.input, "truck \U0001f69a")

    def test_file_that_is_not_utf8_gets_a_clear_message(self):
        self.path.write_bytes(b'{"id": "a", "input": "caf\xe9"}\n')  # latin-1 e-acute
        with self.assertRaises(ValueError) as ctx:
            load_cases(self.path)
        self.assertIn("not valid UTF-8", str(ctx.exception))
        self.assertIn(str(self.path), str(ctx.exception))

    def test_validation_error_names_the_case_and_line(self):
        with self.assertRaises(ValueError) as ctx:
            self.load({"id": "a", "input": "x"}, "", {"id": "billing-7", "input": "x", "tags": "billing"})
        self.assertIn(":3:", str(ctx.exception))
        self.assertIn("billing-7", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
