import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from evalsprint.pricing import PriceTable
from evalsprint.providers import Completion
from evalsprint.targets import Target, from_completion, relativize_path, scrub_text
from evalsprint.loader import load_callable
from evalsprint.types import Case, CaseResult, JudgeResult, TargetOutput, Usage, case_sha, derive_outcome, sanitize_obj, sanitize_text
from tests import TEMPLATES


def call(fn, params=None, case=None, prices=None):
    return Target(fn, params, prices)(case or Case(id="c1", input="hello"))


class ReturnShapeTests(unittest.TestCase):
    def test_plain_string(self):
        out = call(lambda i, p: "an answer")
        self.assertEqual(out.text, "an answer")
        self.assertIsNone(out.error)
        self.assertIsNone(out.cost_usd)  # no usage -> nothing to price
        self.assertEqual(out.usage, Usage())
        self.assertIsNotNone(out.latency_s)

    def test_dict_with_every_field(self):
        def fn(i, p):
            return {
                "text": "answer",
                "model": "claude-haiku-4-5",
                "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 2, "cache_write_tokens": 1},
                "cost_usd": 0.5,
                "latency_s": 1.25,
                "meta": {"context": ["doc1"]},
            }

        out = call(fn)
        self.assertEqual(out.text, "answer")
        self.assertEqual(out.model, "claude-haiku-4-5")
        self.assertEqual(out.usage, Usage(10, 5, 2, 1))
        self.assertEqual(out.cost_usd, 0.5)
        self.assertEqual(out.latency_s, 1.25)
        self.assertEqual(out.meta, {"context": ["doc1"]})

    def test_text_none_becomes_empty_string(self):
        self.assertEqual(call(lambda i, p: {"text": None}).text, "")

    def test_non_string_text_is_stringified(self):
        self.assertEqual(call(lambda i, p: {"text": 42}).text, "42")

    def test_dict_without_text_is_an_error(self):
        out = call(lambda i, p: {"answer": "oops"})
        self.assertEqual(out.text, "")
        self.assertIn("without 'text'", out.error)

    def test_wrong_return_type_is_an_error(self):
        for value in ([1, 2], None, 3, ("a",)):
            with self.subTest(value=value):
                out = call(lambda i, p, v=value: v)
                self.assertIn(type(value).__name__, out.error)
                self.assertEqual(out.text, "")

    def test_error_field_in_result_is_passed_through(self):
        out = call(lambda i, p: {"text": "", "error": "HTTP 500 from client API"})
        self.assertEqual(out.error, "HTTP 500 from client API")

    def test_meta_is_copied(self):
        meta = {"context": ["a"]}
        out = call(lambda i, p: {"text": "x", "meta": meta})
        out.meta["added"] = 1
        self.assertNotIn("added", meta)


class MalformedResultTests(unittest.TestCase):
    def test_malformed_results_fail_the_case_instead_of_crashing_the_run(self):
        # regression: these used to raise out of Target.__call__ and abort the whole run at case N of 500
        bad_results = {
            "non-numeric tokens": {"text": "x", "usage": {"input_tokens": "abc"}},
            "usage not a dict": {"text": "x", "usage": "lots"},
            "non-numeric latency": {"text": "x", "latency_s": "fast"},
            "meta not a dict": {"text": "x", "meta": "context string"},
            "non-numeric cost": {"text": "x", "cost_usd": "free"},
        }
        for label, result in bad_results.items():
            with self.subTest(label):
                out = call(lambda i, p, r=result: r)
                self.assertIn("malformed result", out.error)
                self.assertEqual(out.text, "")
                self.assertIsNotNone(out.latency_s)

    def test_numeric_strings_are_accepted(self):
        out = call(lambda i, p: {"text": "x", "usage": {"input_tokens": "12", "output_tokens": 3.0}, "latency_s": "0.5", "cost_usd": "0.25"})
        self.assertIsNone(out.error)
        self.assertEqual(out.usage, Usage(12, 3))
        self.assertEqual((out.latency_s, out.cost_usd), (0.5, 0.25))

    def test_cost_is_always_a_float(self):
        self.assertIsInstance(call(lambda i, p: {"text": "x", "cost_usd": 1}).cost_usd, float)


class ExceptionTests(unittest.TestCase):
    def test_exception_is_captured_as_error(self):
        def boom(i, p):
            raise ValueError("bad input shape")

        out = call(boom)
        self.assertEqual(out.text, "")
        self.assertEqual(out.error, "ValueError: bad input shape")
        self.assertIn("traceback", out.meta)
        self.assertIn("bad input shape", out.meta["traceback"])
        self.assertGreaterEqual(out.latency_s, 0.0)
        self.assertIsNone(out.cost_usd)

    def test_exception_with_empty_message(self):
        def boom(i, p):
            raise KeyError()

        self.assertTrue(call(boom).error.startswith("KeyError"))

    def test_exceptions_never_escape(self):
        for exc in (RuntimeError("x"), ZeroDivisionError(), TimeoutError("slow"), ConnectionError("down")):
            def boom(i, p, e=exc):
                raise e

            self.assertIsNotNone(call(boom).error)


class CostTests(unittest.TestCase):
    USAGE = {"input_tokens": 1000, "output_tokens": 500}

    def test_cost_computed_from_usage_when_absent(self):
        out = call(lambda i, p: {"text": "x", "model": "claude-haiku-4-5", "usage": self.USAGE})
        self.assertAlmostEqual(out.cost_usd, 0.0035)

    def test_explicit_cost_is_not_overridden(self):
        out = call(lambda i, p: {"text": "x", "model": "claude-haiku-4-5", "usage": self.USAGE, "cost_usd": 0.42})
        self.assertEqual(out.cost_usd, 0.42)

    def test_explicit_zero_cost_is_kept(self):
        out = call(lambda i, p: {"text": "x", "model": "claude-haiku-4-5", "usage": self.USAGE, "cost_usd": 0.0})
        self.assertEqual(out.cost_usd, 0.0)

    def test_model_falls_back_to_params(self):
        out = call(lambda i, p: {"text": "x", "usage": self.USAGE}, params={"model": "claude-haiku-4-5-20251001"})
        self.assertEqual(out.model, "claude-haiku-4-5-20251001")
        self.assertAlmostEqual(out.cost_usd, 0.0035)  # prefix match on the dated id

    def test_model_in_result_beats_params(self):
        out = call(lambda i, p: {"text": "x", "model": "gpt-4o-mini", "usage": self.USAGE}, params={"model": "claude-haiku-4-5"})
        self.assertEqual(out.model, "gpt-4o-mini")
        self.assertAlmostEqual(out.cost_usd, (1000 * 0.15 + 500 * 0.60) / 1e6)

    def test_unknown_model_leaves_cost_empty(self):
        with contextlib.redirect_stderr(io.StringIO()):  # the missing-price warning is expected
            out = call(lambda i, p: {"text": "x", "model": "mystery-model-77", "usage": self.USAGE})
        self.assertIsNone(out.cost_usd)

    def test_no_model_leaves_cost_empty(self):
        self.assertIsNone(call(lambda i, p: {"text": "x", "usage": self.USAGE}).cost_usd)

    def test_no_usage_leaves_cost_empty(self):
        self.assertIsNone(call(lambda i, p: {"text": "x", "model": "claude-haiku-4-5"}).cost_usd)

    def test_custom_price_table(self):
        prices = PriceTable({"client-model": {"input": 10.0, "output": 10.0}})
        out = call(lambda i, p: {"text": "x", "model": "client-model", "usage": {"input_tokens": 100_000, "output_tokens": 100_000}}, prices=prices)
        self.assertAlmostEqual(out.cost_usd, 2.0)


class LatencyTests(unittest.TestCase):
    def test_reported_latency_is_used(self):
        self.assertEqual(call(lambda i, p: {"text": "x", "latency_s": 2.5}).latency_s, 2.5)

    def test_reported_zero_latency_is_kept(self):
        self.assertEqual(call(lambda i, p: {"text": "x", "latency_s": 0}).latency_s, 0.0)

    def test_falls_back_to_wall_clock(self):
        def slow(i, p):
            time.sleep(0.03)
            return "x"

        out = call(slow)
        self.assertGreaterEqual(out.latency_s, 0.025)
        self.assertLess(out.latency_s, 5.0)

    def test_wall_clock_recorded_on_failure_too(self):
        def slow_boom(i, p):
            time.sleep(0.02)
            raise RuntimeError("x")

        self.assertGreaterEqual(call(slow_boom).latency_s, 0.015)


class CallContractTests(unittest.TestCase):
    def test_target_only_sees_case_input_and_a_params_copy(self):
        seen = {}

        def fn(input, params):
            seen["input"] = input
            seen["params"] = params
            params["mutated"] = True
            return "x"

        case = Case(id="c1", input={"q": "hi"}, reference="SECRET", tags=["t"], expected_behavior="answer")
        target = Target(fn, {"model": "m"})
        target(case)
        target(case)
        self.assertEqual(seen["input"], {"q": "hi"})
        self.assertEqual(target.params, {"model": "m"})  # fn's mutation did not leak

    def test_from_completion_helper(self):
        comp = Completion(text="hi", model="claude-haiku-4-5", usage=Usage(3, 4, 5, 6), latency_s=0.7)
        result = from_completion(comp, meta={"context": ["c"]})
        self.assertEqual(
            result,
            {
                "text": "hi",
                "model": "claude-haiku-4-5",
                "usage": {"input_tokens": 3, "output_tokens": 4, "cache_read_tokens": 5, "cache_write_tokens": 6},
                "latency_s": 0.7,
                "meta": {"context": ["c"]},
            },
        )
        out = call(lambda i, p: result)  # and Target accepts it as-is
        self.assertEqual(out.usage, Usage(3, 4, 5, 6))
        self.assertIsNotNone(out.cost_usd)

    def test_from_completion_zero_latency_becomes_none(self):
        self.assertIsNone(from_completion(Completion(text="x", model="m"))["latency_s"])

    def test_from_config_loads_callable_relative_to_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "my_adapter.py").write_text("def run(input, params):\n    return params['prefix'] + input\n")
            target = Target.from_config({"callable": "my_adapter.py:run", "params": {"prefix": ">> "}}, tmp, PriceTable())
            self.assertEqual(target(Case(id="c", input="hi")).text, ">> hi")

    def test_from_config_without_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "noparams_adapter.py").write_text("def run(input, params):\n    return str(params)\n")
            target = Target.from_config({"callable": "noparams_adapter.py:run"}, tmp, PriceTable())
            self.assertEqual(target(Case(id="c", input="x")).text, "{}")

    def test_from_config_bad_callable(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "empty_adapter.py").write_text("x = 1\n")
            with self.assertRaises(AttributeError):
                Target.from_config({"callable": "empty_adapter.py:missing"}, tmp, PriceTable())
            with self.assertRaises(ValueError):
                Target.from_config({"callable": "no_colon_here"}, tmp, PriceTable())


class _EndpointServer:
    """Tiny JSON endpoint standing in for a client's staging API."""

    def __init__(self, statuses=None, body=None):
        self.requests = []
        self.statuses = list(statuses or [])
        self.body = body or {"answer": "Use Settings > Billing.", "model": "claude-haiku-4-5", "usage": {"input_tokens": 50, "output_tokens": 10}, "sources": [{"text": "doc one"}, {"text": "doc two"}]}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("content-length", 0)))
                outer.requests.append({"headers": {k.lower(): v for k, v in self.headers.items()}, "body": json.loads(raw)})
                status = outer.statuses.pop(0) if outer.statuses else 200
                payload = json.dumps(outer.body if status == 200 else {"error": "nope"}).encode()
                self.send_response(status)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/ask"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class TemplateAdapterTests(unittest.TestCase):
    """templates/target_adapter.py is copied into client repos: it has to work as shipped."""

    @classmethod
    def setUpClass(cls):
        cls.adapter_path = TEMPLATES / "target_adapter.py"

    def fn(self, name):
        return load_callable(f"{self.adapter_path}:{name}", TEMPLATES)

    def setUp(self):
        env = mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
        env.start()
        self.addCleanup(env.stop)

    def test_offline_stub_is_a_valid_target(self):
        out = Target(self.fn("offline_stub"), {"model": "claude-sonnet-5-5"})(Case(id="c", input="How do I get an invoice?"))
        self.assertIsNone(out.error)
        self.assertTrue(out.text)
        self.assertGreater(out.cost_usd, 0)  # usage + model -> priced
        self.assertTrue(out.meta["context"])

    def test_offline_stub_accepts_dict_input(self):
        out = Target(self.fn("offline_stub"), {})(Case(id="c", input={"question": "hi", "user": {"plan": "pro"}}))
        self.assertIsNone(out.error)

    def test_call_http_success_maps_answer_usage_and_sources(self):
        with _EndpointServer() as srv:
            with mock.patch.dict(os.environ, {"TARGET_API_KEY": "secret-token"}):
                out = Target(self.fn("call_http"), {"url": srv.url})(Case(id="c", input={"question": "Where is the invoice?", "user": {"plan": "pro"}}))
            req = srv.requests[0]
        self.assertIsNone(out.error)
        self.assertEqual(out.text, "Use Settings > Billing.")
        self.assertEqual(out.model, "claude-haiku-4-5")
        self.assertEqual(out.usage, Usage(50, 10))
        self.assertAlmostEqual(out.cost_usd, (50 * 1.0 + 10 * 5.0) / 1e6)
        self.assertEqual(out.meta["context"], ["doc one", "doc two"])
        self.assertEqual(req["headers"]["authorization"], "Bearer secret-token")
        self.assertEqual(req["body"], {"question": "Where is the invoice?", "user": {"plan": "pro"}})

    def test_call_http_without_token_and_with_custom_keys(self):
        body = {"reply": "hello", "usage": {}}
        with _EndpointServer(body=body) as srv:
            with mock.patch.dict(os.environ, {}):
                os.environ.pop("TARGET_API_KEY", None)
                out = Target(self.fn("call_http"), {"url": srv.url, "question_key": "q", "answer_key": "reply"})(Case(id="c", input="hi"))
            req = srv.requests[0]
        self.assertEqual(out.text, "hello")
        self.assertEqual(req["body"], {"q": "hi"})
        self.assertNotIn("authorization", req["headers"])
        self.assertIsNone(out.cost_usd)

    def test_call_http_retries_transient_errors(self):
        with _EndpointServer(statuses=[503, 200]) as srv, mock.patch("time.sleep") as sleep:
            out = Target(self.fn("call_http"), {"url": srv.url})(Case(id="c", input="hi"))
            self.assertEqual(len(srv.requests), 2)
        self.assertIsNone(out.error)
        sleep.assert_called_once()

    def test_call_http_permanent_error_is_recorded_as_case_error(self):
        with _EndpointServer(statuses=[500, 500, 500]) as srv, mock.patch("time.sleep"):
            out = Target(self.fn("call_http"), {"url": srv.url})(Case(id="c", input="hi"))
        self.assertIn("HTTPError", out.error)
        out = Target(self.fn("call_http"), {"url": "http://127.0.0.1:9/never"})(Case(id="c", input="hi"))
        self.assertIsNotNone(out.error)

    def test_call_llm_with_mock_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "prompts").mkdir()
            Path(tmp, "prompts", "system.txt").write_text("You are terse.", encoding="utf-8")
            Path(tmp, "adapter_copy.py").write_text(self.adapter_path.read_text(encoding="utf-8"), encoding="utf-8")
            fn = load_callable("adapter_copy.py:call_llm", tmp)
            out = Target(fn, {"provider": "mock", "model": "claude-haiku-4-5"})(Case(id="c", input="What is 2+2?"))
        self.assertIsNone(out.error)
        self.assertEqual(out.text, "MOCK RESPONSE: What is 2+2?")
        self.assertEqual(out.model, "claude-haiku-4-5")
        self.assertGreater(out.usage.input_tokens, 0)

    def test_call_function_reports_a_clear_error_when_client_code_is_missing(self):
        out = Target(self.fn("call_function"), {})(Case(id="c", input="hi"))
        self.assertIn("ModuleNotFoundError", out.error)
        self.assertIn("my_app", out.error)

    def test_usage_from_anthropic_dict_and_object(self):
        helper = self.fn("usage_from_anthropic")
        raw = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 2}
        expected = {"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 3, "cache_write_tokens": 2}
        self.assertEqual(helper(raw), expected)

        class Obj:
            input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens = 10, 5, 3, 2

        self.assertEqual(helper(Obj()), expected)

        class Sparse:  # SDK objects may lack cache fields or carry None
            input_tokens, output_tokens, cache_read_input_tokens = 7, 1, None

        self.assertEqual(helper(Sparse()), {"input_tokens": 7, "output_tokens": 1, "cache_read_tokens": 0, "cache_write_tokens": 0})

    def test_usage_from_openai_subtracts_cached_tokens(self):
        helper = self.fn("usage_from_openai")
        raw = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 40}}
        self.assertEqual(helper(raw), {"input_tokens": 60, "output_tokens": 20, "cache_read_tokens": 40})
        self.assertEqual(helper({"prompt_tokens": 8, "completion_tokens": 2}), {"input_tokens": 8, "output_tokens": 2, "cache_read_tokens": 0})

        class Details:
            cached_tokens = 40

        class Obj:
            prompt_tokens, completion_tokens, prompt_tokens_details = 100, 20, Details()

        self.assertEqual(helper(Obj()), {"input_tokens": 60, "output_tokens": 20, "cache_read_tokens": 40})

        class NoDetails:
            prompt_tokens, completion_tokens, prompt_tokens_details = 5, 1, None

        self.assertEqual(helper(NoDetails()), {"input_tokens": 5, "output_tokens": 1, "cache_read_tokens": 0})

    def test_openai_helper_matches_provider_convention(self):
        # same raw usage as test_providers.OPENAI_OK -> Usage(60, 20, 40, 0), so cost is path-independent
        raw = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 40}}
        self.assertEqual(Usage.from_dict(self.fn("usage_from_openai")(raw)), Usage(60, 20, 40, 0))


class LoaderTests(unittest.TestCase):
    """loader.load_callable resolves `file.py:fn` and `package.module:fn` for targets, judges and responders."""

    def test_file_reference_relative_to_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "loader_case_a.py").write_text("def fn():\n    return 'a'\n")
            self.assertEqual(load_callable("loader_case_a.py:fn", tmp)(), "a")

    def test_same_file_is_loaded_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "loader_case_b.py").write_text("import itertools\ncounter = itertools.count()\ndef fn():\n    return next(counter)\n")
            first = load_callable("loader_case_b.py:fn", tmp)
            second = load_callable("loader_case_b.py:fn", tmp)
            self.assertIs(first, second)
            self.assertEqual((first(), second()), (0, 1))  # shared module state, not re-executed

    def test_file_can_import_its_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "loader_helper_c.py").write_text("VALUE = 41\n")
            Path(tmp, "loader_case_c.py").write_text("from loader_helper_c import VALUE\ndef fn():\n    return VALUE + 1\n")
            self.assertEqual(load_callable("loader_case_c.py:fn", tmp)(), 42)

    def test_dotted_module_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp, "loader_pkg_d")
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "mod.py").write_text("def fn():\n    return 'dotted'\n")
            self.assertEqual(load_callable("loader_pkg_d.mod:fn", tmp)(), "dotted")

    def test_upper_case_extension_is_a_file_reference(self):
        with tempfile.TemporaryDirectory() as tmp:  # BOT.PY is how a Windows user may have written it
            Path(tmp, "loader_case_f.PY").write_text("def fn():\n    return 'upper'\n")
            self.assertEqual(load_callable("loader_case_f.PY:fn", tmp)(), "upper")

    def test_a_file_that_failed_to_import_is_not_served_half_built(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "loader_case_g.py")
            path.write_text("raise RuntimeError('import boom')\ndef fn():\n    return 'never'\n")
            with self.assertRaises(RuntimeError):
                load_callable("loader_case_g.py:fn", tmp)
            path.write_text("def fn():\n    return 'fixed'\n")
            self.assertEqual(load_callable("loader_case_g.py:fn", tmp)(), "fixed")  # the second call re-imports, it does not get the broken module

    def test_threads_loading_the_same_file_share_one_module(self):
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "loader_case_h.py").write_text("import time\ntime.sleep(0.05)\nimport itertools\ncounter = itertools.count()\ndef fn():\n    return 'ok'\n")
            with ThreadPoolExecutor(max_workers=8) as pool:
                fns = list(pool.map(lambda _: load_callable("loader_case_h.py:fn", tmp), range(8)))
            self.assertEqual({id(fn) for fn in fns}, {id(fns[0])})  # one execution: nobody saw a half-initialised module

    def test_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "loader_case_e.py").write_text("x = 1\n")
            with self.assertRaises(ValueError):
                load_callable("no_colon", tmp)
            with self.assertRaises(AttributeError) as ctx:
                load_callable("loader_case_e.py:missing", tmp)
            self.assertIn("missing", str(ctx.exception))
            with self.assertRaises(FileNotFoundError):
                load_callable("does_not_exist.py:fn", tmp)
            with self.assertRaises(ModuleNotFoundError):
                load_callable("no_such_module_anywhere:fn", tmp)


class TypesTests(unittest.TestCase):
    def test_usage_arithmetic_and_parsing(self):
        self.assertEqual(Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40), Usage(11, 22, 33, 44))
        self.assertEqual(Usage.from_dict(None), Usage())
        self.assertEqual(Usage.from_dict({"input_tokens": None, "output_tokens": "7"}), Usage(0, 7, 0, 0))
        self.assertEqual(Usage.from_dict({"unknown_key": 5}), Usage())

    def test_output_sha_is_stable_and_short(self):
        self.assertEqual(TargetOutput(text="hello").sha, "2cf24dba5fb0")  # sha256("hello")[:12]
        self.assertEqual(TargetOutput(text="hello").sha, TargetOutput(text="hello", model="other").sha)
        self.assertNotEqual(TargetOutput(text="hello").sha, TargetOutput(text="Hello").sha)
        self.assertEqual(len(TargetOutput(text="").sha), 12)

    def test_case_result_to_dict_matches_contract(self):
        result = CaseResult(
            case=Case(id="c1", input="q", tags=["t"]),
            output=TargetOutput(text="a", model="m", usage=Usage(1, 2), cost_usd=0.5, latency_s=1.0, meta={"context": ["x"]}),
            judges=[JudgeResult("j", True, score=1.0, reasoning="ok", details={"k": 1}, cost_usd=0.1, usage=Usage(3, 4))],
            passed=True,
        )
        d = result.to_dict()
        self.assertEqual(list(d), ["case_id", "case", "case_sha", "output", "judges", "outcome", "error_source", "passed", "n_attempt_errors", "target_retries_used"])
        self.assertEqual((d["n_attempt_errors"], d["target_retries_used"]), (0, 0))
        self.assertEqual((d["outcome"], d["error_source"], d["passed"]), ("pass", None, True))  # derived when the runner did not set it
        self.assertEqual(d["case_sha"], Case(id="c1", input="q", tags=["t"]).sha)
        self.assertEqual(d["case_id"], "c1")
        self.assertEqual(d["output"]["usage"], {"input_tokens": 1, "output_tokens": 2, "cache_read_tokens": 0, "cache_write_tokens": 0})
        self.assertEqual(d["output"]["sha"], TargetOutput(text="a").sha)
        self.assertEqual(d["judges"][0], {"judge": "j", "passed": True, "score": 1.0, "reasoning": "ok", "details": {"k": 1}, "cost_usd": 0.1,
                                          "usage": {"input_tokens": 3, "output_tokens": 4, "cache_read_tokens": 0, "cache_write_tokens": 0}, "error": None})
        json.dumps(d)  # must be JSON-serialisable as is


    def test_from_record_is_the_inverse_of_to_dict(self):
        """`run --resume` rebuilds results from the lines of an earlier run: nothing may be lost or invented on the way."""
        case = Case(id="c1", input="q", reference="r", tags=["t"], metadata={"k": "v"})
        result = CaseResult(
            case=case,
            output=TargetOutput(text="a", model="m", usage=Usage(1, 2, 3, 4), cost_usd=0.5, latency_s=1.5, meta={"context": ["x"], "n": 2}),
            judges=[
                JudgeResult("j", True, score=0.75, reasoning="ok", details={"k": [1, 2]}, cost_usd=0.1, usage=Usage(3, 4)),
                JudgeResult("k", False, error="RuntimeError: down", cost_usd=0.02),
            ],
            passed=False,
            outcome="error",
            error_source="judge",
        )
        record = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(CaseResult.from_record(record, case).to_dict(), record)

    def test_from_record_keeps_attempts_and_stability(self):
        case = Case(id="c1", input="q")
        attempts = [
            {"outcome": "pass", "passed": True, "output_sha": "a", "error_source": None, "cost_usd": 1.0, "judge_cost_usd": 0.25},
            {"outcome": "error", "passed": False, "output_sha": "b", "error_source": "target", "cost_usd": None, "judge_cost_usd": 0.0},
            {"outcome": "pass", "passed": True, "output_sha": "a", "error_source": None, "cost_usd": 1.0, "judge_cost_usd": 0.25},
        ]
        result = CaseResult(case, TargetOutput(text="a"), [JudgeResult("j", True)], True, outcome="pass", attempts=attempts, stability=2 / 3)
        record = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(record["n_attempt_errors"], 1)
        rebuilt = CaseResult.from_record(record, case)
        self.assertEqual((rebuilt.attempts, rebuilt.stability), (attempts, 2 / 3))
        self.assertEqual(rebuilt.to_dict(), record)

    def test_from_record_derives_the_outcome_of_a_record_written_before_outcomes_existed(self):
        case = Case(id="c1", input="q")
        old = {"case_id": "c1", "output": {"text": "a"}, "judges": [{"judge": "j", "passed": True}], "passed": True}
        self.assertEqual(CaseResult.from_record(old, case).resolved(), ("pass", None))
        crashed = {"case_id": "c1", "output": {"text": "", "error": "boom"}, "judges": [], "passed": False}
        self.assertEqual(CaseResult.from_record(crashed, case).resolved(), ("error", "target"))

    def test_to_dict_counts_errored_attempts(self):
        case = Case(id="c1", input="q")
        attempts = [{"outcome": "error"}, {"outcome": "pass"}, {"outcome": "error"}]
        result = CaseResult(case, TargetOutput(text="a"), [JudgeResult("j", True)], True, outcome="pass", attempts=attempts, stability=1 / 3)
        self.assertEqual(result.to_dict()["n_attempt_errors"], 2)
        self.assertEqual(CaseResult(case, TargetOutput(text="", error="x"), [], False).to_dict()["n_attempt_errors"], 1)
        self.assertEqual(CaseResult(case, TargetOutput(text="ok"), [JudgeResult("j", True)], True).to_dict()["n_attempt_errors"], 0)


class SurrogateTests(unittest.TestCase):
    """A lone UTF-16 surrogate (what JSON.stringify emits for a truncated emoji) is a valid str that cannot be
    written as UTF-8. It used to lose a whole paid run at save time."""

    def test_output_text_error_meta_and_model_are_cleaned(self):
        out = call(lambda i, p: {"text": "ok \ud83d", "model": "m\ud83d", "meta": {"context": ["a\ude00", {"k": "b\ud800"}]}})
        self.assertEqual(out.text, "ok \ufffd")
        self.assertEqual(out.model, "m\ufffd")
        self.assertEqual(out.meta, {"context": ["a\ufffd", {"k": "b\ufffd"}]})
        for text in (out.text, out.model, json.dumps(out.meta, ensure_ascii=False)):
            text.encode("utf-8")  # must not raise

    def test_exception_message_is_cleaned(self):
        def boom(i, p):
            raise ValueError("bad \ud83d input")

        out = call(boom)
        self.assertEqual(out.error, "ValueError: bad \ufffd input")
        out.error.encode("utf-8")
        out.meta["traceback"].encode("utf-8")

    def test_sha_never_raises_on_a_surrogate(self):
        self.assertEqual(len(TargetOutput(text="x \ud83d").sha), 12)
        self.assertEqual(TargetOutput(text="x \ud83d").sha, TargetOutput(text="x \ufffd").sha)

    def test_valid_astral_characters_are_kept(self):
        out = call(lambda i, p: "truck \U0001f69a and \u65e5\u672c\u8a9e")
        self.assertEqual(out.text, "truck \U0001f69a and \u65e5\u672c\u8a9e")

    def test_sanitize_helpers(self):
        self.assertEqual(sanitize_text("a\ud83db"), "a\ufffdb")
        self.assertEqual(sanitize_text(None), None)
        self.assertEqual(sanitize_obj({"a\ud800": ["x\udc00", (1, "y\udfff")], "n": 3}), {"a\ufffd": ["x\ufffd", [1, "y\ufffd"]], "n": 3})


class AttemptTests(unittest.TestCase):
    def test_attempt_index_reaches_the_target_as_a_param(self):
        seen = []
        target = Target(lambda i, p: seen.append(dict(p)) or "x", {"model": "m"})
        target(Case(id="c", input="q"))
        target(Case(id="c", input="q"), attempt=0)
        target(Case(id="c", input="q"), attempt=2)
        self.assertEqual(seen, [{"model": "m"}, {"model": "m", "_attempt": 0}, {"model": "m", "_attempt": 2}])
        self.assertEqual(target.params, {"model": "m"})  # the stored params never carry it

    def test_a_target_may_ignore_the_attempt(self):
        self.assertEqual(Target(lambda i, p: "same", {})(Case(id="c", input="q"), attempt=1).text, "same")


class PathScrubbingTests(unittest.TestCase):
    """Tracebacks land in results.jsonl, which is committed as the baseline: no user directories in it."""

    def test_relativize_path(self):
        self.assertEqual(relativize_path("/work/proj/evals/bot.py", "/work/proj/evals"), "bot.py")
        self.assertEqual(relativize_path("/work/proj/evals/lib/util.py", "/work/proj/evals"), "lib/util.py")
        self.assertEqual(relativize_path("/elsewhere/other.py", "/work/proj/evals"), ".../other.py")
        self.assertEqual(relativize_path("/usr/lib/python3.11/json/decoder.py"), "<stdlib>/json/decoder.py")
        self.assertEqual(relativize_path("/venv/lib/python3.12/site-packages/yaml/loader.py"), "<site-packages>/yaml/loader.py")
        self.assertTrue(relativize_path(str(Path(__import__("evalsprint").__file__)), None).replace("\\", "/").startswith("evalsprint/"))

    def test_windows_paths(self):
        self.assertEqual(relativize_path("C:\\Users\\Renan Gonzalez\\proj\\evals\\bot.py", "C:\\Users\\Renan Gonzalez\\proj\\evals"), "bot.py")
        self.assertEqual(relativize_path("C:\\Users\\Renan Gonzalez\\other\\x.py", "C:\\Users\\Renan Gonzalez\\proj"), ".../x.py")
        self.assertEqual(relativize_path("c:\\users\\renan\\proj\\bot.py", "C:\\Users\\Renan\\proj"), "bot.py")  # drive-letter paths compare case-insensitively

    def test_relative_paths_and_pseudo_files_are_left_alone(self):
        for path in ("bot.py", "lib/util.py", "<frozen importlib._bootstrap>", "<string>", ".../x.py"):
            self.assertEqual(relativize_path(path, "/work/proj"), path)

    def test_scrub_text_is_idempotent_and_hides_the_base_dir_and_home(self):
        tb = '  File "/work/proj/evals/bot.py", line 3, in answer\n    open("/work/proj/evals/x.json")\nFileNotFoundError: /work/proj/evals/x.json'
        once = scrub_text(tb, "/work/proj/evals")
        self.assertNotIn("/work/proj", once)
        self.assertIn('File "bot.py"', once)
        self.assertEqual(scrub_text(once, "/work/proj/evals"), once)
        home = str(Path.home())
        if len(home) > 3:
            self.assertNotIn(home, scrub_text(f"cannot read {home}/secrets.txt"))
        self.assertEqual(scrub_text(""), "")

    def test_a_real_traceback_has_no_absolute_paths_and_skips_the_kits_own_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.py"
            path.write_text("def answer(input, params):\n    raise RuntimeError('boom at ' + __file__)\n", encoding="utf-8")
            fn = load_callable("adapter.py:answer", tmp)
            out = Target(fn, {}, base_dir=tmp)(Case(id="c", input="q"))
        tb = out.meta["traceback"]
        self.assertNotIn(tmp, tb)
        self.assertNotIn(tmp, out.error)
        self.assertIn('File "adapter.py"', tb)
        self.assertIn("RuntimeError: boom at", tb)
        self.assertNotIn("targets.py", tb)  # Target._call is not the user's code
        self.assertTrue(out.error.startswith("RuntimeError: boom at ./adapter.py") or "adapter.py" in out.error)

    def test_a_base_dir_that_is_not_in_resolved_form_still_scrubs_the_paths(self):
        # the traceback names the file by its resolved path; the same holds for "." and for a Windows 8.3 short name or a symlink
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "sub").mkdir()
            (Path(tmp) / "adapter.py").write_text("def answer(input, params):\n    raise RuntimeError('boom at ' + __file__)\n", encoding="utf-8")
            fn = load_callable("adapter.py:answer", tmp)
            out = Target(fn, {}, base_dir=Path(tmp) / "sub" / "..")(Case(id="c", input="q"))
            resolved = str(Path(tmp).resolve())
        self.assertIn('File "adapter.py"', out.meta["traceback"])
        for text in (out.meta["traceback"], out.error):
            self.assertNotIn(tmp, text)
            self.assertNotIn(resolved, text)

    def test_traceback_keeps_the_innermost_frames(self):
        def deep(n):
            if n == 0:
                raise ValueError("bottom")
            deep(n - 1)

        out = Target(lambda i, p: deep(12), {})(Case(id="c", input="q"))
        self.assertIn("ValueError: bottom", out.meta["traceback"])
        self.assertLessEqual(out.meta["traceback"].count("File "), 5)


class ModelTypeTests(unittest.TestCase):
    def test_a_non_string_model_fails_only_this_case(self):
        out = call(lambda i, p: {"text": "x", "model": ["a"]})
        self.assertIn("malformed result", out.error)
        self.assertIn("model", out.error)
        self.assertEqual(call(lambda i, p: {"text": "x", "model": 7}).text, "")  # not a string either


class OutcomeDerivationTests(unittest.TestCase):
    def test_target_error_wins_over_judge_errors(self):
        errored = JudgeResult("j", False, error="boom")
        self.assertEqual(derive_outcome("timeout", [errored], False), ("error", "target"))
        self.assertEqual(derive_outcome(None, [errored], False), ("error", "judge"))
        self.assertEqual(derive_outcome(None, [JudgeResult("j", True)], True), ("pass", None))
        self.assertEqual(derive_outcome(None, [JudgeResult("j", False)], False), ("fail", None))
        self.assertEqual(derive_outcome(None, [{"judge": "j", "passed": True, "error": "x"}], True), ("error", "judge"))  # record dicts too

    def test_case_result_outcome_overrides_and_attempts(self):
        base = dict(case=Case(id="c", input="q"), output=TargetOutput(text="a"), judges=[JudgeResult("j", True)], passed=True)
        self.assertEqual(CaseResult(**base, outcome="fail").to_dict()["passed"], False)  # passed is true only when outcome == pass
        d = CaseResult(**base, attempts=[{"outcome": "pass"}], stability=1.0).to_dict()
        self.assertEqual((d["attempts"], d["stability"]), ([{"outcome": "pass"}], 1.0))
        self.assertNotIn("attempts", CaseResult(**base).to_dict())


class CaseShaTests(unittest.TestCase):
    """case_sha covers the WHOLE case except its id: a judge-facing extra field is part of what a case tests."""

    BASE = dict(id="c1", input="q", reference="ref", expected_behavior="answer", tags=["core"], checks=[], metadata={"must_include": "tracking"})

    def test_stable_short_and_sensitive_to_everything_but_the_id(self):
        sha = Case(**self.BASE).sha
        self.assertEqual(len(sha), 12)
        self.assertEqual(sha, Case(**self.BASE).sha)
        edits = {
            "input": "q2",
            "reference": "ref2",
            "expected_behavior": "abstain",
            "tags": ["core", "extra"],
            "checks": [{"type": "contains", "values": ["x"]}],
            "metadata": {"must_include": "refund"},  # the field the judge is told to look for
        }
        for field, value in edits.items():
            with self.subTest(field=field):
                self.assertNotEqual(Case(**{**self.BASE, field: value}).sha, sha)

    def test_extra_fields_of_the_dataset_row_count(self):
        row = {"id": "c1", "input": "q", "must_include": "tracking"}
        self.assertNotEqual(Case.from_dict(row).sha, Case.from_dict({**row, "must_include": "refund"}).sha)
        self.assertNotEqual(Case.from_dict(row).sha, Case.from_dict({**row, "notes": "new field"}).sha)

    def test_the_id_is_left_out(self):
        self.assertEqual(Case(**self.BASE).sha, Case(**{**self.BASE, "id": "renamed"}).sha)  # a rename is not an edit of the content

    def test_key_order_of_dict_inputs_does_not_matter(self):
        self.assertEqual(Case(id="x", input={"a": 1, "b": 2}).sha, Case(id="x", input={"b": 2, "a": 1}).sha)
        self.assertEqual(case_sha({"id": "x", "input": {"a": 1, "b": 2}}), case_sha({"input": {"b": 2, "a": 1}, "id": "y"}))

    def test_a_case_and_its_stored_dict_hash_alike(self):
        case = Case(**self.BASE)
        self.assertEqual(case_sha(case), case_sha(case.to_dict()))
        self.assertEqual(case_sha(case), CaseResult(case=case, output=TargetOutput(text="a"), judges=[], passed=True).to_dict()["case_sha"])

    def test_contract_formula(self):
        import hashlib

        case = Case(**self.BASE)
        payload = {"input": "q", "reference": "ref", "expected_behavior": "answer", "tags": ["core"], "checks": [], "metadata": {"must_include": "tracking"}}
        blob = json.dumps(payload, sort_keys=True, default=str)  # whole case dict, no `id`
        self.assertEqual(case.sha, hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12])


if __name__ == "__main__":
    unittest.main()
