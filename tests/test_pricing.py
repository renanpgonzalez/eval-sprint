import contextlib
import io
import unittest

from evalsprint import pricing
from evalsprint.pricing import PriceTable
from evalsprint.types import Usage


class LookupTests(unittest.TestCase):
    def test_exact_match(self):
        price = PriceTable().lookup("claude-haiku-4-5")
        self.assertEqual(price["input"], 1.00)
        self.assertEqual(price["output"], 5.00)

    def test_dated_model_id_matches_by_prefix(self):
        table = PriceTable()
        self.assertIs(table.lookup("claude-haiku-4-5-20251001"), table.models["claude-haiku-4-5"])

    def test_longest_prefix_wins(self):
        table = PriceTable()
        # gpt-5.4 ($2.50) and gpt-5.4-mini ($0.75) both prefix this id
        self.assertEqual(table.lookup("gpt-5.4-mini-2026-03-01")["input"], 0.75)
        self.assertEqual(table.lookup("gpt-5.4-2026-03-01")["input"], 2.50)

    def test_exact_match_beats_shorter_prefix(self):
        table = PriceTable()
        # claude-opus-5 is a prefix of claude-opus-5-5 but the exact entry must win
        self.assertEqual(table.lookup("claude-opus-5-5")["input"], 4.00)
        self.assertEqual(table.lookup("claude-opus-5")["input"], 5.00)

    def test_longest_prefix_with_custom_entries(self):
        table = PriceTable({"acme": {"input": 1.0, "output": 1.0}, "acme-pro": {"input": 9.0, "output": 9.0}})
        self.assertEqual(table.lookup("acme-pro-2027")["input"], 9.0)
        self.assertEqual(table.lookup("acme-lite")["input"], 1.0)

    def test_unknown_and_empty_model(self):
        table = PriceTable()
        self.assertIsNone(table.lookup("totally-unknown-model"))
        self.assertIsNone(table.lookup(None))
        self.assertIsNone(table.lookup(""))

    def test_prefix_is_not_substring(self):
        self.assertIsNone(PriceTable().lookup("my-claude-haiku-4-5"))


class CostTests(unittest.TestCase):
    def test_input_output(self):
        # haiku: $1 in / $5 out per 1M tokens
        cost = PriceTable().cost("claude-haiku-4-5", Usage(input_tokens=1000, output_tokens=500))
        self.assertAlmostEqual(cost, (1000 * 1.0 + 500 * 5.0) / 1e6)
        self.assertAlmostEqual(cost, 0.0035)

    def test_dated_model_cost_equals_base_model_cost(self):
        usage = Usage(input_tokens=1234, output_tokens=321)
        t = PriceTable()
        self.assertEqual(t.cost("claude-haiku-4-5-20251001", usage), t.cost("claude-haiku-4-5", usage))

    def test_cache_tokens_priced_separately(self):
        usage = Usage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
        cost = PriceTable().cost("claude-haiku-4-5", usage)
        self.assertAlmostEqual(cost, 0.10 + 1.25)

    def test_cache_prices_default_to_input_price(self):
        # gpt-4o-mini has no cache_read / cache_write entries
        usage = Usage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
        self.assertAlmostEqual(PriceTable().cost("gpt-4o-mini", usage), 0.15 * 2)

    def test_openai_style_cache_read_price(self):
        usage = Usage(input_tokens=600_000, output_tokens=0, cache_read_tokens=400_000)
        # gpt-5.4: input 2.50, cache_read 0.25
        self.assertAlmostEqual(PriceTable().cost("gpt-5.4", usage), 0.6 * 2.50 + 0.4 * 0.25)

    def test_zero_usage_is_zero_cost(self):
        self.assertEqual(PriceTable().cost("claude-haiku-4-5", Usage()), 0.0)

    def test_unknown_model_returns_none_and_warns_once(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.assertIsNone(PriceTable().cost("nonexistent-model-xyz-1", Usage(input_tokens=5)))
            self.assertIsNone(PriceTable().cost("nonexistent-model-xyz-1", Usage(input_tokens=5)))
        self.assertEqual(buf.getvalue().count("nonexistent-model-xyz-1"), 1)
        self.assertIn("no price", buf.getvalue())

    def test_none_model_returns_none_without_warning(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.assertIsNone(PriceTable().cost(None, Usage(input_tokens=5)))
        self.assertEqual(buf.getvalue(), "")


class OverrideTests(unittest.TestCase):
    def test_override_merges_with_default_entry(self):
        table = PriceTable({"claude-haiku-4-5": {"input": 2.0}})
        price = table.models["claude-haiku-4-5"]
        self.assertEqual(price["input"], 2.0)
        self.assertEqual(price["output"], 5.0)  # untouched field kept
        self.assertEqual(price["cache_read"], 0.10)

    def test_override_adds_new_model(self):
        table = PriceTable({"client-model-1": {"input": 3.0, "output": 6.0}})
        self.assertAlmostEqual(table.cost("client-model-1", Usage(input_tokens=1_000_000, output_tokens=1_000_000)), 9.0)

    def test_overrides_do_not_leak_between_tables_or_into_defaults(self):
        PriceTable({"claude-haiku-4-5": {"input": 99.0}, "leaky-model": {"input": 1.0}})
        fresh = PriceTable()
        self.assertEqual(fresh.models["claude-haiku-4-5"]["input"], 1.0)
        self.assertNotIn("leaky-model", fresh.models)
        self.assertEqual(pricing._defaults()["claude-haiku-4-5"]["input"], 1.0)

    def test_override_none_is_fine(self):
        self.assertEqual(PriceTable(None).models, PriceTable({}).models)

    def test_override_applies_to_dated_ids_via_prefix(self):
        table = PriceTable({"claude-haiku-4-5": {"input": 10.0}})
        cost = table.cost("claude-haiku-4-5-20251001", Usage(input_tokens=1_000_000))
        self.assertAlmostEqual(cost, 10.0)


class BatchTests(unittest.TestCase):
    def test_batch_flag_halves_cost(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        normal = PriceTable().cost("claude-haiku-4-5", usage)
        batched = PriceTable({"claude-haiku-4-5": {"batch": True}}).cost("claude-haiku-4-5", usage)
        self.assertAlmostEqual(normal, 6.0)
        self.assertAlmostEqual(batched, 3.0)

    def test_batch_false_is_a_no_op(self):
        usage = Usage(input_tokens=1_000_000)
        self.assertEqual(PriceTable({"claude-haiku-4-5": {"batch": False}}).cost("claude-haiku-4-5", usage), 1.0)


class DefaultTableTests(unittest.TestCase):
    def test_every_default_entry_has_numeric_input_and_output(self):
        table = PriceTable()
        self.assertGreater(len(table.models), 5)
        for model, price in table.models.items():
            for key in ("input", "output"):
                self.assertIsInstance(price.get(key), (int, float), f"{model}.{key}")
                self.assertGreater(price[key], 0, f"{model}.{key}")
            for key in ("cache_read", "cache_write"):
                if key in price:
                    self.assertGreater(price[key], 0)
                    if key == "cache_read":
                        self.assertLessEqual(price["cache_read"], price["input"], model)

    def test_every_default_model_resolves_to_itself(self):
        table = PriceTable()
        for model in table.models:
            self.assertIs(table.lookup(model), table.models[model])


if __name__ == "__main__":
    unittest.main()
