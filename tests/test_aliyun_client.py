"""Offline contract tests for the Alibaba Cloud Model Studio client."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx


# Importing app.config normally loads the project .env.  Keep these tests fully
# isolated from production credentials and give the module a harmless fake key.
with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-key"}, clear=False):
    with patch("dotenv.load_dotenv", return_value=False):
        from app.services.aliyun_client import (
            AliyunAPIError,
            AliyunModelClient,
            TranscriptionResult,
            TranscriptionSentence,
        )


class AliyunModelClientTests(unittest.IsolatedAsyncioTestCase):
    """Exercise request contracts through httpx.MockTransport only."""

    async def asyncSetUp(self) -> None:
        self.clients: list[AliyunModelClient] = []

    async def asyncTearDown(self) -> None:
        for client in self.clients:
            await client.close()

    def make_client(
        self,
        handler: httpx.MockTransport,
        *,
        max_retries: int = 0,
        sleep=None,
    ) -> AliyunModelClient:
        client = AliyunModelClient(
            api_key="offline-test-key",
            base_url="https://model-studio.invalid/compatible-mode/v1",
            native_base_url="https://native-model-studio.invalid/api/v1",
            timeout=1.0,
            max_retries=max_retries,
            max_concurrency=1,
            transport=handler,
            sleep=sleep or self.no_sleep,
        )

        async def load_result_with_mock_transport(url: str) -> dict:
            return await client._request_json("GET", url, authorize=False)

        client._result_loader = load_result_with_mock_transport
        self.clients.append(client)
        return client

    @staticmethod
    async def no_sleep(_delay: float) -> None:
        return None

    async def test_chat_request_path_payload_and_content_list_parsing(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                request.url.path,
                "/compatible-mode/v1/chat/completions",
            )
            self.assertEqual(request.headers["Authorization"], "Bearer offline-test-key")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "qwen-test")
            self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
            self.assertEqual(payload["max_tokens"], 321)
            self.assertEqual(payload["temperature"], 0.2)
            self.assertIs(payload["enable_search"], True)
            self.assertIs(payload["enable_thinking"], False)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "first"},
                                    {"type": "text", "text": "second"},
                                ]
                            }
                        }
                    ]
                },
            )

        client = self.make_client(httpx.MockTransport(handler))
        result = await client.chat(
            model="qwen-test",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=321,
            temperature=0.2,
            enable_search=True,
            enable_thinking=False,
        )

        self.assertEqual(result, "first\nsecond")

    async def test_chat_sends_dynamic_completion_and_thinking_controls(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["max_completion_tokens"], 3500)
            self.assertNotIn("max_tokens", payload)
            self.assertIs(payload["enable_thinking"], True)
            self.assertEqual(payload["thinking_budget"], 1024)
            self.assertIs(payload["preserve_thinking"], False)
            self.assertEqual(
                payload["response_format"],
                {"type": "json_object"},
            )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "bounded output"}}]},
            )

        client = self.make_client(httpx.MockTransport(handler))
        result = await client.chat(
            model="qwen3.8-max",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=None,
            max_completion_tokens=3500,
            enable_thinking=True,
            thinking_budget=1024,
            preserve_thinking=False,
            response_format={"type": "json_object"},
        )

        self.assertEqual(result, "bounded output")

    async def test_chat_can_omit_output_limit_for_prompt_controlled_long_form(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertNotIn("max_tokens", payload)
            self.assertNotIn("max_completion_tokens", payload)
            self.assertEqual(payload["thinking_budget"], 16384)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "complete long form"}}]},
            )

        client = self.make_client(httpx.MockTransport(handler))
        result = await client.chat(
            model="qwen3.8-max",
            messages=[{"role": "user", "content": "cover every source point"}],
            max_tokens=None,
            max_completion_tokens=None,
            enable_thinking=True,
            thinking_budget=16384,
        )

        self.assertEqual(result, "complete long form")

    def test_transcription_parser_preserves_old_text_and_immutable_timestamps(self) -> None:
        payload = {
            "transcripts": [
                {
                    "channel_id": 1,
                    "text": "API 提供的完整文本。",
                    "sentences": [
                        {
                            "text": "API 提供的",
                            "begin_time": 25,
                            "end_time": 650,
                        },
                        {
                            "text": "完整文本。",
                            "begin_time": 700.0,
                            "end_time": 1400,
                        },
                    ],
                }
            ]
        }

        self.assertEqual(
            AliyunModelClient._extract_transcription_text(payload),
            "API 提供的完整文本。",
        )
        result = AliyunModelClient._extract_transcription_result(payload)
        self.assertEqual(result.text, "API 提供的完整文本。")
        self.assertEqual(
            result.sentences,
            (
                TranscriptionSentence("API 提供的", 25, 650, 1),
                TranscriptionSentence("完整文本。", 700, 1400, 1),
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            result.text = "不可变"  # type: ignore[misc]

    async def test_responses_request_path_and_text_source_parsing(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/compatible-mode/v1/responses")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "qwen-research")
            self.assertEqual(payload["input"], "research this")
            self.assertEqual(payload["max_output_tokens"], 999)
            self.assertIs(payload["enable_thinking"], True)
            self.assertEqual(payload["tools"], [{"type": "web_search"}])
            return httpx.Response(
                200,
                json={
                    "output_text": "overview",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "details"},
                                {"type": "output_text", "text": "overview"},
                            ],
                        },
                        {
                            "type": "web_search_call",
                            "action": {
                                "sources": [
                                    {"url": "https://example.test/one"},
                                    {"url": "https://example.test/one"},
                                    {"url": "https://example.test/two"},
                                ]
                            },
                        },
                    ],
                },
            )

        client = self.make_client(httpx.MockTransport(handler))
        result = await client.responses(
            model="qwen-research",
            input_data="research this",
            tools=[{"type": "web_search"}],
            max_output_tokens=999,
        )

        self.assertEqual(
            result,
            "overview\n\ndetails\n\n## API 检索来源\n"
            "- https://example.test/one\n- https://example.test/two",
        )

    async def test_asr_uses_chat_path_and_embeds_audio_as_data_uri(self) -> None:
        audio_bytes = b"offline mp3 bytes"

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                request.url.path,
                "/compatible-mode/v1/chat/completions",
            )
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "qwen-asr-test")
            self.assertIs(payload["stream"], False)
            self.assertEqual(
                payload["asr_options"],
                {"enable_itn": True, "language": "zh"},
            )
            self.assertEqual(len(payload["messages"]), 1)
            self.assertEqual(payload["messages"][0]["role"], "user")
            audio = payload["messages"][0]["content"][0]
            self.assertEqual(audio["type"], "input_audio")
            self.assertEqual(
                audio["input_audio"]["data"],
                "data:audio/mpeg;base64,b2ZmbGluZSBtcDMgYnl0ZXM=",
            )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "transcript"}}]},
            )

        client = self.make_client(httpx.MockTransport(handler))
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "segment.mp3"
            audio_path.write_bytes(audio_bytes)
            result = await client.transcribe_audio(
                model="qwen-asr-test",
                audio_path=str(audio_path),
                context="context",
                language="zh",
            )

        self.assertEqual(result, "transcript")

    async def test_429_retries_once_and_honors_retry_after(self) -> None:
        requests = 0
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            if requests == 1:
                return httpx.Response(
                    429,
                    headers={"Retry-After": "0"},
                    json={"error": {"message": "rate limited"}},
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "recovered"}}]},
            )

        client = self.make_client(
            httpx.MockTransport(handler),
            max_retries=2,
            sleep=record_sleep,
        )
        result = await client.chat(model="qwen-test", messages=[])

        self.assertEqual(result, "recovered")
        self.assertEqual(requests, 2)
        self.assertEqual(delays, [0.0])

    async def test_401_is_not_retried(self) -> None:
        requests = 0
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            return httpx.Response(
                401,
                json={"error": {"message": "invalid credential"}},
            )

        client = self.make_client(
            httpx.MockTransport(handler),
            max_retries=3,
            sleep=record_sleep,
        )

        with self.assertRaises(AliyunAPIError) as raised:
            await client.chat(model="qwen-test", messages=[])

        self.assertEqual(raised.exception.status_code, 401)
        self.assertIn("HTTP 401", str(raised.exception))
        self.assertNotIn("offline-test-key", str(raised.exception))
        self.assertEqual(requests, 1)
        self.assertEqual(delays, [])

    async def test_chat_rejects_empty_content(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "   "}}]},
            )
        )
        client = self.make_client(transport)

        with self.assertRaisesRegex(AliyunAPIError, "空内容"):
            await client.chat(model="qwen-test", messages=[])

    async def test_responses_rejects_empty_output(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"output": []})
        )
        client = self.make_client(transport)

        with self.assertRaisesRegex(AliyunAPIError, "Responses API 返回了空内容"):
            await client.responses(model="qwen-test", input_data="anything")

    async def test_responses_tolerates_null_optional_collections(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "output": [
                        {"type": "message", "content": None},
                        {"type": "web_search_call", "content": None, "action": None},
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "usable text"}],
                        },
                        {"type": "web_search_call", "action": {"sources": None}},
                    ]
                },
            )
        )
        client = self.make_client(transport)

        result = await client.responses(model="qwen-test", input_data="anything")

        self.assertEqual(result, "usable text")

    async def test_file_transcription_submits_then_polls_sdk_results(self) -> None:
        requests: list[tuple[str, str]] = []
        poll_count = 0
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal poll_count
            requests.append((request.method, str(request.url)))
            if request.method == "POST":
                self.assertEqual(request.url.host, "native-model-studio.invalid")
                self.assertEqual(
                    request.url.path,
                    "/api/v1/services/audio/asr/transcription",
                )
                self.assertEqual(request.headers["Authorization"], "Bearer offline-test-key")
                self.assertEqual(request.headers["X-DashScope-Async"], "enable")
                self.assertEqual(
                    json.loads(request.content),
                    {
                        "model": "qwen-audio-3.0-asr-flash-filetrans",
                        "input": {"file_urls": ["https://media.invalid/audio.mp3"]},
                        "parameters": {
                            "channel_id": [0],
                            "language_hints": ["zh", "en"],
                        },
                    },
                )
                return httpx.Response(200, json={"output": {"task_id": "task-123"}})

            if request.url.host == "native-model-studio.invalid":
                self.assertEqual(request.method, "GET")
                self.assertEqual(request.url.path, "/api/v1/tasks/task-123")
                self.assertEqual(request.headers["Authorization"], "Bearer offline-test-key")
                poll_count += 1
                if poll_count == 1:
                    return httpx.Response(
                        200,
                        json={
                            "output": {
                                "task_id": "task-123",
                                "task_status": "PENDING",
                                "results": [],
                            }
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "output": {
                            "task_id": "task-123",
                            "task_status": "SUCCEEDED",
                            "results": [
                                {
                                    "file_url": "https://media.invalid/audio.mp3",
                                    "subtask_status": "SUCCEEDED",
                                    "transcription_url": (
                                        "https://transcript-result.invalid/task-123.json"
                                    ),
                                }
                            ],
                        }
                    },
                )

            self.assertEqual(request.url.host, "transcript-result.invalid")
            self.assertEqual(request.url.path, "/task-123.json")
            self.assertNotIn("Authorization", request.headers)
            return httpx.Response(
                200,
                json={
                    "transcripts": [
                        {
                            "channel_id": 0,
                            "sentences": [
                                {
                                    "text": "第一句。",
                                    "begin_time": 120,
                                    "end_time": 980,
                                },
                                {
                                    "text": "第二句。",
                                    "begin_time": "1000",
                                    "end_time": "1880",
                                },
                            ],
                        }
                    ]
                },
            )

        client = self.make_client(
            httpx.MockTransport(handler),
            sleep=record_sleep,
        )
        result = await client.transcribe_file_url_detailed(
            model="qwen-audio-3.0-asr-flash-filetrans",
            file_url="https://media.invalid/audio.mp3",
            language_hints=["zh", "en"],
            channel_ids=[0],
            poll_interval=0.25,
            timeout=2.0,
        )

        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "第一句。第二句。")
        self.assertEqual(
            result.sentences,
            (
                TranscriptionSentence("第一句。", 120, 980, 0),
                TranscriptionSentence("第二句。", 1000, 1880, 0),
            ),
        )
        self.assertEqual(result.segments, result.sentences)
        self.assertEqual(result.sentences[0].begin_time, 120)
        self.assertEqual(result.sentences[0].end_time, 980)
        with self.assertRaises(FrozenInstanceError):
            result.sentences[0].text = "不可变"  # type: ignore[misc]
        self.assertEqual(poll_count, 2)
        self.assertEqual(delays, [0.25])
        self.assertEqual(len(requests), 4)

    async def test_file_transcription_accepts_legacy_result_shape(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"output": {"task_id": "legacy-task"}})
            if request.url.host == "native-model-studio.invalid":
                return httpx.Response(
                    200,
                    json={
                        "output": {
                            "task_status": "SUCCEEDED",
                            "result": {
                                "transcription_url": (
                                    "https://transcript-result.invalid/legacy.json"
                                )
                            },
                        }
                    },
                )
            return httpx.Response(200, json={"text": "legacy transcript"})

        client = self.make_client(httpx.MockTransport(handler))
        result = await client.transcribe_file_url(
            model="qwen-filetrans-test",
            file_url="https://media.invalid/legacy.mp3",
            poll_interval=0.1,
            timeout=1.0,
        )

        self.assertEqual(result, "legacy transcript")

    async def test_file_transcription_failed_task_raises_without_retrying(self) -> None:
        requests = 0
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            if request.method == "POST":
                return httpx.Response(200, json={"output": {"task_id": "failed-task"}})
            return httpx.Response(
                200,
                json={
                    "output": {
                        "task_status": "FAILED",
                        "message": "unsupported audio format",
                    }
                },
            )

        client = self.make_client(
            httpx.MockTransport(handler),
            sleep=record_sleep,
        )

        with self.assertRaises(AliyunAPIError) as raised:
            await client.transcribe_file_url(
                model="qwen-filetrans-test",
                file_url="https://media.invalid/bad.mp3",
                poll_interval=0.1,
                timeout=1.0,
            )

        self.assertIn("FAILED", str(raised.exception))
        self.assertIn("unsupported audio format", str(raised.exception))
        self.assertEqual(requests, 2)
        self.assertEqual(delays, [])

    async def test_file_transcription_pending_task_times_out(self) -> None:
        polls = 0
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal polls
            if request.method == "POST":
                return httpx.Response(200, json={"output": {"task_id": "slow-task"}})
            polls += 1
            return httpx.Response(
                200,
                json={"output": {"task_status": "PENDING", "results": []}},
            )

        client = self.make_client(
            httpx.MockTransport(handler),
            sleep=record_sleep,
        )

        with self.assertRaises(AliyunAPIError) as raised:
            await client.transcribe_file_url(
                model="qwen-filetrans-test",
                file_url="https://media.invalid/slow.mp3",
                poll_interval=0.5,
                timeout=1.0,
            )

        self.assertIn("等待超时", str(raised.exception))
        self.assertIn("PENDING", str(raised.exception))
        self.assertEqual(polls, 2)
        self.assertEqual(delays, [0.5])

    async def test_filetrans_submit_does_not_retry_ambiguous_failure(self) -> None:
        submit_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal submit_count
            self.assertEqual(request.method, "POST")
            submit_count += 1
            return httpx.Response(503, json={"message": "ambiguous failure"})

        client = self.make_client(
            httpx.MockTransport(handler),
            max_retries=3,
        )
        with self.assertRaises(AliyunAPIError):
            await client.transcribe_file_url(
                model="qwen-filetrans-test",
                file_url="https://media.invalid/audio.mp3",
                poll_interval=0.1,
                timeout=1.0,
            )

        self.assertEqual(submit_count, 1)

    async def test_filetrans_uses_real_wall_clock_timeout(self) -> None:
        client = self.make_client(httpx.MockTransport(lambda _request: None))

        async def slow_request(method: str, _url: str, **_kwargs):
            if method == "POST":
                return {"output": {"task_id": "slow-task"}}
            await asyncio.sleep(0.1)
            return {"output": {"task_status": "PENDING", "results": []}}

        with patch.object(
            client,
            "_request_json",
            new_callable=AsyncMock,
            side_effect=slow_request,
        ), self.assertRaisesRegex(AliyunAPIError, "墙钟上限"):
            await client.transcribe_file_url(
                model="qwen-filetrans-test",
                file_url="https://media.invalid/audio.mp3",
                poll_interval=0.01,
                timeout=0.02,
            )


if __name__ == "__main__":
    unittest.main()
