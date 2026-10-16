import json
import os
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from evalsprint import providers
from evalsprint.providers import (
    AnthropicProvider,
    BaseProvider,
    Completion,
    MockProvider,
    OpenAIProvider,
    ProviderError,
    estimate_tokens,
    get_provider,
    register_provider,
)
from evalsprint.types import Usage

ANTHROPIC_OK = {
    "id": "msg_1",
    "model": "claude-test-9",
    "content": [
        {"type": "text", "text": "Hello "},
        {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
        {"type": "text", "text": "world"},
    ],
    "usage": {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 2},
}
OPENAI_OK = {
    "id": "chatcmpl-1",
    "model": "gpt-test-9",
    "choices": [{"message": {"role": "assistant", "content": "Hi there"}}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 40}},
}


class FakeLLMServer:
    """A local HTTP server that records requests and replays scripted responses."""

    def __init__(self):
        self.requests: list[dict] = []
        self.script: list[tuple[int, dict, dict]] = []  # (status, json body, extra headers), consumed in order
        self.default_bodies = {"/v1/messages": ANTHROPIC_OK, "/chat/completions": OPENAI_OK, "/v1/chat/completions": OPENAI_OK}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("content-length", 0))
                raw = self.rfile.read(length)
                outer.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": json.loads(raw.decode("utf-8")),
                    }
                )
                if outer.script:
                    status, body, headers = outer.script.pop(0)
                else:
                    status, body, headers = 200, outer.default_bodies[self.path], {}
                payload = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):  # silence
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class HTTPTestCase(unittest.TestCase):
    def setUp(self):
        # Never route the loopback test server through a corporate/CI proxy.
        env = mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
        env.start()
        self.addCleanup(env.stop)
        for var in ("http_proxy", "HTTP_PROXY"):
            os.environ.pop(var, None)
        self.server = FakeLLMServer().__enter__()
        self.addCleanup(self.server.__exit__, None, None, None)
        sleep = mock.patch("evalsprint.providers.time.sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)


class AnthropicProviderTests(HTTPTestCase):
    def provider(self, **kw):
        return AnthropicProvider(api_key="sk-ant-test", base_url=self.server.url, **kw)

    def test_request_shape_and_headers(self):
        comp = self.provider().complete("You are terse.", [{"role": "user", "content": "hi"}], "claude-test-1", max_tokens=321, temperature=0.2)
        req = self.server.requests[0]
        self.assertEqual(req["path"], "/v1/messages")
        self.assertEqual(req["headers"]["x-api-key"], "sk-ant-test")
        self.assertEqual(req["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(req["headers"]["content-type"], "application/json")
        self.assertEqual(
            req["body"],
            {
                "model": "claude-test-1",
                "max_tokens": 321,
                "messages": [{"role": "user", "content": "hi"}],
                "system": "You are terse.",
                "temperature": 0.2,
            },
        )
        self.assertIsInstance(comp, Completion)

    def test_system_and_temperature_omitted_when_not_given(self):
        self.provider().complete(None, [{"role": "user", "content": "hi"}], "claude-test-1")
        body = self.server.requests[0]["body"]
        self.assertNotIn("system", body)
        self.assertNotIn("temperature", body)
        self.assertEqual(body["max_tokens"], 1024)

    def test_temperature_zero_is_sent(self):
        self.provider().complete("s", [{"role": "user", "content": "hi"}], "m", temperature=0.0)
        self.assertEqual(self.server.requests[0]["body"]["temperature"], 0.0)

    def test_text_blocks_joined_and_usage_normalised(self):
        comp = self.provider().complete("s", [{"role": "user", "content": "hi"}], "requested-model")
        self.assertEqual(comp.text, "Hello world")  # non-text blocks skipped
        self.assertEqual(comp.model, "claude-test-9")  # server-reported model wins
        self.assertEqual(comp.usage, Usage(input_tokens=11, output_tokens=7, cache_read_tokens=3, cache_write_tokens=2))
        self.assertGreaterEqual(comp.latency_s, 0.0)
        self.assertEqual(comp.raw["id"], "msg_1")

    def test_missing_usage_block_is_zero(self):
        self.server.script.append((200, {"content": [{"type": "text", "text": "x"}]}, {}))
        comp = self.provider().complete("s", [{"role": "user", "content": "hi"}], "m")
        self.assertEqual(comp.usage, Usage())
        self.assertEqual(comp.model, "m")

    def test_base_url_trailing_slash_is_stripped(self):
        p = AnthropicProvider(api_key="k", base_url=self.server.url + "/")
        p.complete("s", [{"role": "user", "content": "hi"}], "m")
        self.assertEqual(self.server.requests[0]["path"], "/v1/messages")

    def test_base_url_from_environment(self):
        with mock.patch.dict(os.environ, {"EVALSPRINT_ANTHROPIC_BASE_URL": self.server.url}):
            p = AnthropicProvider(api_key="k")
        p.complete("s", [{"role": "user", "content": "hi"}], "m")
        self.assertEqual(len(self.server.requests), 1)

    def test_api_key_from_environment_and_missing_key_error(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "from-env"}):
            self.assertEqual(AnthropicProvider(base_url=self.server.url).api_key, "from-env")
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ProviderError) as ctx:
                AnthropicProvider(base_url=self.server.url)
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_unknown_kwargs_are_ignored(self):
        p = AnthropicProvider(api_key="k", base_url=self.server.url, responder="sim.py:fn", base_dir="/x", nonsense=1)
        p.complete("s", [{"role": "user", "content": "hi"}], "m")
        self.assertEqual(len(self.server.requests), 1)


class OpenAIProviderTests(HTTPTestCase):
    def provider(self, **kw):
        return OpenAIProvider(api_key="sk-test", base_url=self.server.url, **kw)

    def test_request_shape_and_headers(self):
        self.provider().complete("Be brief.", [{"role": "user", "content": "hi"}], "gpt-test-1", max_tokens=99, temperature=0.5)
        req = self.server.requests[0]
        self.assertEqual(req["path"], "/chat/completions")
        self.assertEqual(req["headers"]["authorization"], "Bearer sk-test")
        self.assertNotIn("x-api-key", req["headers"])
        body = req["body"]
        self.assertEqual(body["model"], "gpt-test-1")
        self.assertEqual(body["max_completion_tokens"], 99)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["temperature"], 0.5)
        self.assertEqual(body["messages"], [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}])

    def test_no_system_means_no_system_message_and_no_temperature(self):
        self.provider().complete(None, [{"role": "user", "content": "hi"}], "m")
        body = self.server.requests[0]["body"]
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertNotIn("temperature", body)

    def test_caller_messages_are_not_mutated(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.provider().complete("sys", msgs, "m")
        self.assertEqual(msgs, [{"role": "user", "content": "hi"}])

    def test_cached_tokens_are_subtracted_from_input(self):
        comp = self.provider().complete("s", [{"role": "user", "content": "hi"}], "requested")
        self.assertEqual(comp.text, "Hi there")
        self.assertEqual(comp.model, "gpt-test-9")
        self.assertEqual(comp.usage, Usage(input_tokens=60, output_tokens=20, cache_read_tokens=40, cache_write_tokens=0))

    def test_usage_without_cache_details(self):
        self.server.script.append((200, {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": 8, "completion_tokens": 2}}, {}))
        comp = self.provider().complete("s", [{"role": "user", "content": "hi"}], "m")
        self.assertEqual(comp.usage, Usage(input_tokens=8, output_tokens=2))

    def test_null_content_becomes_empty_string(self):
        self.server.script.append((200, {"choices": [{"message": {"content": None}}]}, {}))
        comp = self.provider().complete("s", [{"role": "user", "content": "hi"}], "m")
        self.assertEqual(comp.text, "")

    def test_openai_compatible_base_url_needs_no_key(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            p = OpenAIProvider(base_url=self.server.url + "/v1")
        p.complete("s", [{"role": "user", "content": "hi"}], "local-model")
        req = self.server.requests[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertNotIn("authorization", req["headers"])

    def test_missing_key_for_official_endpoint_raises(self):
        env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "OPENAI_BASE_URL")}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ProviderError) as ctx:
                OpenAIProvider()
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_custom_api_key_env(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}):
            p = OpenAIProvider(base_url=self.server.url, api_key_env="OPENROUTER_API_KEY")
        p.complete("s", [{"role": "user", "content": "hi"}], "m")
        self.assertEqual(self.server.requests[0]["headers"]["authorization"], "Bearer or-key")

    def test_unknown_kwargs_are_ignored(self):
        p = OpenAIProvider(api_key="k", base_url=self.server.url, responder=lambda **kw: "x", base_dir="/x", whatever=True)
        p.complete("s", [{"role": "user", "content": "hi"}], "m")
        self.assertEqual(len(self.server.requests), 1)


class RetryTests(HTTPTestCase):
    def provider(self, **kw):
        return AnthropicProvider(api_key="k", base_url=self.server.url, **kw)

    def call(self, provider):
        return provider.complete("s", [{"role": "user", "content": "hi"}], "m")

    def test_retries_on_429_then_succeeds(self):
        self.server.script.append((429, {"error": "rate limited"}, {"retry-after": "0"}))
        comp = self.call(self.provider())
        self.assertEqual(comp.text, "Hello world")
        self.assertEqual(len(self.server.requests), 2)
        self.assertEqual(self.sleep.call_count, 1)

    def test_retries_on_5xx_and_overloaded(self):
        for status in (500, 502, 503, 504, 529):
            with self.subTest(status=status):
                self.server.requests.clear()
                self.sleep.reset_mock()
                self.server.script.append((status, {"error": "temporary"}, {}))
                self.assertEqual(self.call(self.provider()).text, "Hello world")
                self.assertEqual(len(self.server.requests), 2)

    def test_retry_after_header_is_respected(self):
        self.server.script.append((429, {"error": "slow down"}, {"retry-after": "7"}))
        self.call(self.provider())
        waited = self.sleep.call_args[0][0]
        self.assertGreaterEqual(waited, 7.0)
        self.assertLess(waited, 7.6)  # 7 + up to 0.5 jitter

    def test_exponential_backoff_without_retry_after(self):
        for _ in range(3):
            self.server.script.append((503, {"error": "busy"}, {}))
        self.call(self.provider())
        waits = [c[0][0] for c in self.sleep.call_args_list]
        self.assertEqual(len(waits), 3)
        for attempt, waited in enumerate(waits):
            self.assertGreaterEqual(waited, 2**attempt)
            self.assertLess(waited, 2**attempt + 0.6)

    def test_gives_up_after_max_retries(self):
        for _ in range(10):
            self.server.script.append((429, {"error": "still limited"}, {"retry-after": "0"}))
        with self.assertRaises(ProviderError) as ctx:
            self.call(self.provider(max_retries=2))
        self.assertIn("HTTP 429", str(ctx.exception))
        self.assertEqual(len(self.server.requests), 3)  # first try + 2 retries

    def test_400_raises_without_retry(self):
        self.server.script.append((400, {"error": {"message": "max_tokens too large"}}, {}))
        with self.assertRaises(ProviderError) as ctx:
            self.call(self.provider())
        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertIn("max_tokens too large", str(ctx.exception))
        self.assertEqual(len(self.server.requests), 1)
        self.sleep.assert_not_called()

    def test_auth_errors_are_not_retried(self):
        for status in (401, 403, 404):
            with self.subTest(status=status):
                self.server.requests.clear()
                self.server.script.append((status, {"error": "nope"}, {}))
                with self.assertRaises(ProviderError):
                    self.call(self.provider())
                self.assertEqual(len(self.server.requests), 1)

    def test_openai_provider_retries_too(self):
        self.server.script.append((429, {"error": "x"}, {"retry-after": "0"}))
        p = OpenAIProvider(api_key="k", base_url=self.server.url)
        self.assertEqual(p.complete("s", [{"role": "user", "content": "hi"}], "m").text, "Hi there")
        self.assertEqual(len(self.server.requests), 2)

    def test_network_error_is_retried_then_raised(self):
        with socket.socket() as s:  # grab a port nobody listens on
            s.bind(("127.0.0.1", 0))
            dead_port = s.getsockname()[1]
        p = AnthropicProvider(api_key="k", base_url=f"http://127.0.0.1:{dead_port}", max_retries=1)
        with self.assertRaises(ProviderError) as ctx:
            p.complete("s", [{"role": "user", "content": "hi"}], "m")
        self.assertIn("network error", str(ctx.exception))
        self.assertEqual(self.sleep.call_count, 1)


class MockProviderTests(unittest.TestCase):
    MSGS = [{"role": "user", "content": "What is 2+2?"}]

    def test_default_echo(self):
        comp = MockProvider().complete("sys", self.MSGS, "any-model")
        self.assertEqual(comp.text, "MOCK RESPONSE: What is 2+2?")
        self.assertEqual(comp.model, "any-model")
        self.assertGreaterEqual(comp.usage.input_tokens, 1)
        self.assertGreaterEqual(comp.usage.output_tokens, 1)

    def test_default_echo_truncates_and_handles_empty(self):
        long = MockProvider().complete(None, [{"role": "user", "content": "x" * 500}], "m")
        self.assertEqual(long.text, "MOCK RESPONSE: " + "x" * 200)
        self.assertEqual(MockProvider().complete(None, [], "m").text, "MOCK RESPONSE: ")

    def test_string_responder(self):
        p = MockProvider(responder=lambda system, messages, model: "four")
        comp = p.complete("s", self.MSGS, "m")
        self.assertEqual(comp.text, "four")
        # usage is estimated from prompt/completion size when the responder gives none
        self.assertEqual(comp.usage.output_tokens, estimate_tokens("four"))
        self.assertEqual(comp.usage.input_tokens, max(1, (len("s") + len("What is 2+2?")) // 4))

    def test_dict_responder_with_usage_and_latency(self):
        def responder(system, messages, model):
            return {"text": "ok", "usage": {"input_tokens": 500, "output_tokens": 50, "cache_read_tokens": 10}, "latency_s": 1.5}

        comp = MockProvider(responder=responder).complete("s", self.MSGS, "m")
        self.assertEqual(comp.text, "ok")
        self.assertEqual(comp.usage, Usage(500, 50, 10, 0))
        self.assertEqual(comp.latency_s, 1.5)

    def test_dict_responder_without_text_gives_empty_text(self):
        comp = MockProvider(responder=lambda **kw: {"usage": {"input_tokens": 1, "output_tokens": 1}}).complete("s", self.MSGS, "m")
        self.assertEqual(comp.text, "")

    def test_responder_gets_keyword_arguments(self):
        seen = {}

        def responder(system, messages, model):
            seen.update(system=system, messages=messages, model=model)
            return "x"

        MockProvider(responder=responder).complete("SYS", self.MSGS, "the-model")
        self.assertEqual(seen, {"system": "SYS", "messages": self.MSGS, "model": "the-model"})

    def test_responder_loaded_from_file_relative_to_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "sim.py").write_text("def reply(system, messages, model):\n    return 'from file: ' + messages[-1]['content']\n")
            comp = MockProvider(responder="sim.py:reply", base_dir=tmp).complete(None, self.MSGS, "m")
        self.assertEqual(comp.text, "from file: What is 2+2?")

    def test_responder_exception_propagates(self):
        def boom(**kw):
            raise RuntimeError("responder failed")

        with self.assertRaises(RuntimeError):
            MockProvider(responder=boom).complete(None, self.MSGS, "m")


class RegistryTests(unittest.TestCase):
    def test_unknown_provider_name(self):
        with self.assertRaises(ValueError) as ctx:
            get_provider("does-not-exist")
        msg = str(ctx.exception)
        self.assertIn("does-not-exist", msg)
        for known in ("anthropic", "openai", "mock"):
            self.assertIn(known, msg)

    def test_instances_are_cached_by_name_and_options(self):
        f = lambda **kw: "x"  # noqa: E731
        a = get_provider("mock", responder=f)
        self.assertIs(a, get_provider("mock", responder=f))
        self.assertIsNot(a, get_provider("mock", responder=lambda **kw: "y"))
        self.assertIsInstance(get_provider("mock"), MockProvider)

    def test_register_custom_provider(self):
        class Echo(BaseProvider):
            name = "echo-test"

            def __init__(self, **_):
                pass

            def complete(self, system, messages, model, max_tokens=1024, temperature=None):
                return Completion(text="echo", model=model)

        register_provider("echo-test", Echo)
        try:
            self.assertEqual(get_provider("echo-test").complete(None, [], "m").text, "echo")
        finally:
            providers._PROVIDERS.pop("echo-test", None)

    def test_base_provider_is_abstract(self):
        with self.assertRaises(NotImplementedError):
            BaseProvider().complete(None, [], "m")

    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens(""), 1)
        self.assertEqual(estimate_tokens("a" * 400), 100)


if __name__ == "__main__":
    unittest.main()
