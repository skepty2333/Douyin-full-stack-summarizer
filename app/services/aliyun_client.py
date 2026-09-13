"""Unified asynchronous client for Alibaba Cloud Model Studio APIs."""
from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import httpx

from app.config import (
    AI_MAX_CONCURRENCY,
    AI_MAX_RETRIES,
    AI_REQUEST_TIMEOUT_SECONDS,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    DASHSCOPE_NATIVE_BASE_URL,
    MODEL_USAGE_LOG_ENABLED,
)
from app.database.model_usage_store import ModelUsageStore, get_model_usage_store


logger = logging.getLogger(__name__)
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_MAX_TRANSCRIPTION_JSON_BYTES = 32 * 1024 * 1024


@dataclass(slots=True)
class _RequestMetrics:
    """Mutable counters for one logical provider call."""

    request_count: int = 0
    retry_count: int = 0
    provider_request_id: str = ""


class AliyunAPIError(RuntimeError):
    """A sanitized Model Studio API error safe to expose in logs."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class TranscriptionSentence:
    """One immutable FileTrans sentence with optional millisecond offsets."""

    text: str
    begin_time_ms: Optional[int] = None
    end_time_ms: Optional[int] = None
    channel_id: Optional[int] = None

    @property
    def begin_time(self) -> Optional[int]:
        """Expose the source JSON field name as a compatibility alias."""
        return self.begin_time_ms

    @property
    def end_time(self) -> Optional[int]:
        """Expose the source JSON field name as a compatibility alias."""
        return self.end_time_ms


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Immutable FileTrans text plus any sentence-level timing metadata."""

    text: str
    sentences: tuple[TranscriptionSentence, ...] = ()

    @property
    def segments(self) -> tuple[TranscriptionSentence, ...]:
        """Use ``segments`` when sentence boundaries are treated generically."""
        return self.sentences


class AliyunModelClient:
    """One reusable, concurrency-limited client for all Qwen model calls."""

    def __init__(
        self,
        *,
        api_key: str = DASHSCOPE_API_KEY,
        base_url: str = DASHSCOPE_BASE_URL,
        native_base_url: str = DASHSCOPE_NATIVE_BASE_URL,
        timeout: float = AI_REQUEST_TIMEOUT_SECONDS,
        max_retries: int = AI_MAX_RETRIES,
        max_concurrency: int = AI_MAX_CONCURRENCY,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        result_loader: Optional[
            Callable[[str], Awaitable[dict[str, Any]]]
        ] = None,
        usage_store: Optional[ModelUsageStore] = None,
        usage_store_factory: Optional[Callable[[], ModelUsageStore]] = None,
    ):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.native_base_url = native_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._sleep = sleep
        self._result_loader = result_loader
        self._usage_store = usage_store
        self._usage_store_factory = usage_store_factory
        self._client = httpx.AsyncClient(
            base_url=f"{self.base_url}/",
            follow_redirects=True,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 20.0)),
            limits=httpx.Limits(max_connections=max_concurrency, max_keepalive_connections=max_concurrency),
            transport=transport,
            headers={"Content-Type": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _check_configuration(self) -> None:
        if not self.api_key or self.api_key.startswith("your_"):
            raise AliyunAPIError("DASHSCOPE_API_KEY 未配置")
        if not self.base_url.startswith("https://"):
            raise AliyunAPIError("DASHSCOPE_BASE_URL 必须使用 https://")
        if not self.native_base_url.startswith("https://"):
            raise AliyunAPIError("DASHSCOPE_NATIVE_BASE_URL 必须使用 https://")

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        payload: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        authorize: bool = True,
        retry: bool = True,
        metrics: Optional[_RequestMetrics] = None,
    ) -> dict[str, Any]:
        self._check_configuration()
        headers = dict(extra_headers or {})
        if authorize:
            headers["Authorization"] = f"Bearer {self.api_key}"

        attempt_limit = self.max_retries if retry else 0
        async with self._semaphore:
            for attempt in range(attempt_limit + 1):
                try:
                    if metrics is not None:
                        metrics.request_count += 1
                    request_kwargs: dict[str, Any] = {"headers": headers}
                    if payload is not None:
                        request_kwargs["json"] = payload
                    response = await self._client.request(method, url, **request_kwargs)
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt >= attempt_limit:
                        raise AliyunAPIError(
                            f"百炼网络请求失败（已重试 {attempt} 次）: {type(exc).__name__}"
                        ) from exc
                    if metrics is not None:
                        metrics.retry_count += 1
                    await self._wait_before_retry(attempt, None)
                    continue

                if response.status_code < 400:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise AliyunAPIError("百炼返回了无效 JSON", response.status_code) from exc
                    if not isinstance(data, dict):
                        raise AliyunAPIError("百炼返回格式异常", response.status_code)
                    if metrics is not None:
                        request_id = (
                            response.headers.get("x-request-id")
                            or response.headers.get("x-dashscope-request-id")
                            or data.get("request_id")
                            or data.get("id")
                        )
                        if isinstance(request_id, str) and request_id.strip():
                            metrics.provider_request_id = request_id.strip()[:256]
                    return data

                message = self._extract_error(response)
                if response.status_code not in _RETRYABLE_STATUS or attempt >= attempt_limit:
                    raise AliyunAPIError(message, response.status_code)

                logger.warning(
                    "百炼请求失败，准备重试: status=%s attempt=%s/%s",
                    response.status_code,
                    attempt + 1,
                    attempt_limit,
                )
                if metrics is not None:
                    metrics.retry_count += 1
                await self._wait_before_retry(attempt, response)

        raise AliyunAPIError("百炼请求异常退出")

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        metrics: Optional[_RequestMetrics] = None,
    ) -> dict[str, Any]:
        return await self._request_json("POST", path, payload=payload, metrics=metrics)

    async def _record_model_call(
        self,
        *,
        model: str,
        operation: str,
        api_kind: str,
        status: str,
        started_at: str,
        started_monotonic: float,
        metrics: _RequestMetrics,
        response_payload: Optional[dict[str, Any]] = None,
        audio_seconds: Optional[float] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        """Best-effort telemetry: observability must never break model output."""

        if self._usage_store is None and self._usage_store_factory is None:
            return
        finished_at = datetime.now(timezone.utc).isoformat()
        latency_ms = max(0, round((time.monotonic() - started_monotonic) * 1000))
        http_status = error.status_code if isinstance(error, AliyunAPIError) else None
        if error is None:
            error_type = ""
            error_message = ""
        elif isinstance(error, AliyunAPIError):
            error_type = type(error).__name__
            error_message = (
                f"百炼调用失败（HTTP {error.status_code}）"
                if error.status_code is not None
                else "百炼调用失败（详情见服务日志）"
            )
        elif isinstance(error, asyncio.CancelledError):
            error_type = type(error).__name__
            error_message = "调用被取消"
        elif isinstance(error, asyncio.TimeoutError):
            error_type = type(error).__name__
            error_message = "调用超时"
        else:
            error_type = type(error).__name__
            error_message = "调用失败（详情见服务日志）"

        def write() -> None:
            store = self._usage_store
            if store is None and self._usage_store_factory is not None:
                store = self._usage_store_factory()
            if store is None:
                return
            store.record(
                model=model,
                operation=operation,
                api_kind=api_kind,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=latency_ms,
                request_count=metrics.request_count,
                retry_count=metrics.retry_count,
                provider_request_id=metrics.provider_request_id,
                http_status=http_status,
                error_type=error_type,
                error_message=error_message,
                response_payload=response_payload,
                audio_seconds=audio_seconds,
            )

        try:
            write_task = asyncio.create_task(asyncio.to_thread(write))
            await asyncio.shield(write_task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("写入模型调用日志失败，继续返回模型结果")

    def _native_url(self, path: str) -> str:
        return f"{self.native_base_url}/{path.lstrip('/')}"

    async def _wait_before_retry(
        self, attempt: int, response: Optional[httpx.Response]
    ) -> None:
        retry_after: Optional[float] = None
        if response is not None:
            raw_retry_after = response.headers.get("Retry-After", "").strip()
            try:
                retry_after = float(raw_retry_after) if raw_retry_after else None
            except ValueError:
                retry_after = None
        delay = retry_after if retry_after is not None else min(30.0, 2 ** attempt + random.random())
        await self._sleep(max(0.0, delay))

    @staticmethod
    def _extract_error(response: httpx.Response) -> str:
        detail = ""
        try:
            data = response.json()
            error = data.get("error", data) if isinstance(data, dict) else data
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("code") or "")
            else:
                detail = str(error)
        except ValueError:
            detail = response.text
        detail = " ".join(detail.split())[:300]
        suffix = f": {detail}" if detail else ""
        return f"百炼请求失败（HTTP {response.status_code}）{suffix}"

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AliyunAPIError("百炼响应缺少 choices[0].message.content") from exc
        if isinstance(content, list):
            text_parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("text")
            ]
            content = "\n".join(text_parts)
        if not isinstance(content, str) or not content.strip():
            raise AliyunAPIError("百炼返回了空内容")
        return content.strip()

    @staticmethod
    def _extract_response_text(data: dict[str, Any]) -> str:
        """Extract text and source URLs from an OpenAI-compatible Responses payload."""
        direct = data.get("output_text")
        text_parts: list[str] = [direct] if isinstance(direct, str) and direct.strip() else []
        source_urls: list[str] = []

        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                text_value = content.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value)
            action = item.get("action") or {}
            if isinstance(action, dict):
                for source in action.get("sources") or []:
                    if isinstance(source, dict):
                        url = source.get("url")
                        if isinstance(url, str) and url.startswith(("http://", "https://")):
                            source_urls.append(url)

        unique_text = []
        for value in text_parts:
            value = value.strip()
            if value and value not in unique_text:
                unique_text.append(value)
        if not unique_text:
            raise AliyunAPIError("百炼 Responses API 返回了空内容")

        result = "\n\n".join(unique_text)
        unique_sources = list(dict.fromkeys(source_urls))[:5]
        if unique_sources:
            result += "\n\n## API 检索来源\n" + "\n".join(f"- {url}" for url in unique_sources)
        return result

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: Optional[int] = 8192,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.3,
        enable_search: bool = False,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        preserve_thinking: Optional[bool] = None,
        response_format: Optional[dict[str, Any]] = None,
        operation: str = "chat",
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_completion_tokens is not None:
            payload["max_completion_tokens"] = max_completion_tokens
        elif max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if enable_search:
            payload["enable_search"] = True
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            payload["thinking_budget"] = thinking_budget
        if preserve_thinking is not None:
            payload["preserve_thinking"] = preserve_thinking
        if response_format is not None:
            payload["response_format"] = response_format
        started_at = datetime.now(timezone.utc).isoformat()
        started_monotonic = time.monotonic()
        metrics = _RequestMetrics()
        data: Optional[dict[str, Any]] = None
        try:
            data = await self._post("chat/completions", payload, metrics=metrics)
            content = self._extract_content(data)
        except BaseException as exc:
            await self._record_model_call(
                model=model,
                operation=operation,
                api_kind="chat_completions",
                status="cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                started_at=started_at,
                started_monotonic=started_monotonic,
                metrics=metrics,
                response_payload=data,
                error=exc,
            )
            raise
        await self._record_model_call(
            model=model,
            operation=operation,
            api_kind="chat_completions",
            status="success",
            started_at=started_at,
            started_monotonic=started_monotonic,
            metrics=metrics,
            response_payload=data,
        )
        return content

    @staticmethod
    def _extract_embeddings(data: dict[str, Any], expected: int) -> list[list[float]]:
        items = data.get("data")
        if not isinstance(items, list) or len(items) != expected:
            raise AliyunAPIError("百炼 embeddings 返回条数与请求不符")
        vectors: list[Optional[list[float]]] = [None] * expected
        for item in items:
            if not isinstance(item, dict):
                raise AliyunAPIError("百炼 embeddings 返回格式异常")
            index = item.get("index")
            vector = item.get("embedding")
            if (
                not isinstance(index, int)
                or not (0 <= index < expected)
                or vectors[index] is not None
                or not isinstance(vector, list)
                or not vector
                or not all(isinstance(value, (int, float)) for value in vector)
            ):
                raise AliyunAPIError("百炼 embeddings 返回向量无效")
            vectors[index] = [float(value) for value in vector]
        return [vector for vector in vectors if vector is not None]

    async def embed(
        self,
        *,
        model: str,
        texts: Sequence[str],
        dimensions: Optional[int] = None,
        operation: str = "embedding",
    ) -> list[list[float]]:
        """Embed a small batch of texts through the OpenAI-compatible endpoint."""
        inputs = [str(text) for text in texts]
        if not inputs:
            return []
        if any(not text.strip() for text in inputs):
            raise ValueError("embedding 输入不能为空文本")
        payload: dict[str, Any] = {
            "model": model,
            "input": inputs,
            "encoding_format": "float",
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions
        started_at = datetime.now(timezone.utc).isoformat()
        started_monotonic = time.monotonic()
        metrics = _RequestMetrics()
        data: Optional[dict[str, Any]] = None
        try:
            data = await self._post("embeddings", payload, metrics=metrics)
            vectors = self._extract_embeddings(data, expected=len(inputs))
        except BaseException as exc:
            await self._record_model_call(
                model=model,
                operation=operation,
                api_kind="embeddings",
                status="cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                started_at=started_at,
                started_monotonic=started_monotonic,
                metrics=metrics,
                response_payload=data,
                error=exc,
            )
            raise
        await self._record_model_call(
            model=model,
            operation=operation,
            api_kind="embeddings",
            status="success",
            started_at=started_at,
            started_monotonic=started_monotonic,
            metrics=metrics,
            response_payload=data,
        )
        return vectors

    async def responses(
        self,
        *,
        model: str,
        input_data: Any,
        tools: Optional[list[dict[str, Any]]] = None,
        max_output_tokens: int = 16384,
        enable_thinking: bool = True,
        operation: str = "responses",
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "input": input_data,
            "max_output_tokens": max_output_tokens,
            "enable_thinking": enable_thinking,
            "store": False,
        }
        if tools:
            payload["tools"] = tools
        started_at = datetime.now(timezone.utc).isoformat()
        started_monotonic = time.monotonic()
        metrics = _RequestMetrics()
        data: Optional[dict[str, Any]] = None
        try:
            data = await self._post("responses", payload, metrics=metrics)
            content = self._extract_response_text(data)
        except BaseException as exc:
            await self._record_model_call(
                model=model,
                operation=operation,
                api_kind="responses",
                status="cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                started_at=started_at,
                started_monotonic=started_monotonic,
                metrics=metrics,
                response_payload=data,
                error=exc,
            )
            raise
        await self._record_model_call(
            model=model,
            operation=operation,
            api_kind="responses",
            status="success",
            started_at=started_at,
            started_monotonic=started_monotonic,
            metrics=metrics,
            response_payload=data,
        )
        return content

    async def transcribe_audio(
        self,
        *,
        model: str,
        audio_path: str,
        context: str = "",
        language: Optional[str] = None,
        audio_duration_seconds: Optional[float] = None,
        operation: str = "asr_fallback",
    ) -> str:
        path = Path(audio_path)
        audio_bytes = await asyncio.to_thread(path.read_bytes)
        encoded = await asyncio.to_thread(base64.b64encode, audio_bytes)
        data_uri = f"data:audio/mpeg;base64,{encoded.decode('ascii')}"

        # The dedicated qwen3-asr service currently rejects system messages on
        # workspace endpoints, even though some generic API docs list them.
        # Keep ``context`` in the public signature for future compatibility,
        # but send the proven minimal ASR payload.
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": data_uri},
                    }
                ],
            }
        ]
        asr_options: dict[str, Any] = {"enable_itn": True}
        if language:
            asr_options["language"] = language
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "asr_options": asr_options,
        }
        started_at = datetime.now(timezone.utc).isoformat()
        started_monotonic = time.monotonic()
        metrics = _RequestMetrics()
        data: Optional[dict[str, Any]] = None
        try:
            data = await self._post("chat/completions", payload, metrics=metrics)
            content = self._extract_content(data)
        except BaseException as exc:
            await self._record_model_call(
                model=model,
                operation=operation,
                api_kind="audio_chat_completions",
                status="cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                started_at=started_at,
                started_monotonic=started_monotonic,
                metrics=metrics,
                response_payload=data,
                audio_seconds=audio_duration_seconds,
                error=exc,
            )
            raise
        await self._record_model_call(
            model=model,
            operation=operation,
            api_kind="audio_chat_completions",
            status="success",
            started_at=started_at,
            started_monotonic=started_monotonic,
            metrics=metrics,
            response_payload=data,
            audio_seconds=audio_duration_seconds,
        )
        return content

    @staticmethod
    def _extract_transcription_urls(output: dict[str, Any]) -> list[str]:
        """Support both current SDK results and the older singular result shape."""
        urls: list[str] = []
        results = output.get("results")
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                subtask_status = str(item.get("subtask_status") or "").upper()
                value = item.get("transcription_url")
                if subtask_status in {"", "SUCCEEDED"} and isinstance(value, str):
                    urls.append(value)

        result = output.get("result")
        if isinstance(result, dict):
            value = result.get("transcription_url")
            if isinstance(value, str):
                urls.append(value)

        direct = output.get("transcription_url")
        if isinstance(direct, str):
            urls.append(direct)
        return list(dict.fromkeys(urls))

    @staticmethod
    def _extract_transcription_text(data: dict[str, Any]) -> str:
        """Extract text from a downloaded DashScope transcription JSON file."""
        return AliyunModelClient._extract_transcription_result(data).text

    @staticmethod
    def _timestamp_ms(value: Any) -> Optional[int]:
        """Normalize the integer millisecond offsets returned by FileTrans."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
        return None

    @classmethod
    def _extract_transcription_result(
        cls, data: dict[str, Any]
    ) -> TranscriptionResult:
        """Extract text and retain FileTrans ``sentences`` timing metadata."""
        direct = data.get("text")
        direct_text = direct.strip() if isinstance(direct, str) and direct.strip() else ""

        containers = [data]
        output = data.get("output")
        if isinstance(output, dict):
            containers.append(output)

        parts: list[str] = []
        sentences_out: list[TranscriptionSentence] = []
        for container in containers:
            transcripts = container.get("transcripts")
            if not isinstance(transcripts, list):
                continue
            for transcript in transcripts:
                if not isinstance(transcript, dict):
                    continue
                raw_channel_id = transcript.get("channel_id")
                channel_id = (
                    raw_channel_id
                    if isinstance(raw_channel_id, int) and not isinstance(raw_channel_id, bool)
                    else None
                )
                raw_sentences = transcript.get("sentences")
                sentence_text: list[str] = []
                if isinstance(raw_sentences, list):
                    for sentence in raw_sentences:
                        if not isinstance(sentence, dict):
                            continue
                        raw_text = sentence.get("text")
                        if not isinstance(raw_text, str) or not raw_text.strip():
                            continue
                        normalized_text = raw_text.strip()
                        sentence_text.append(normalized_text)
                        sentences_out.append(
                            TranscriptionSentence(
                                text=normalized_text,
                                begin_time_ms=cls._timestamp_ms(
                                    sentence.get("begin_time")
                                ),
                                end_time_ms=cls._timestamp_ms(sentence.get("end_time")),
                                channel_id=channel_id,
                            )
                        )

                text = transcript.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                    continue
                if sentence_text:
                    parts.append("".join(sentence_text))

        unique_parts = list(dict.fromkeys(parts))
        text = direct_text or "\n\n".join(unique_parts)
        if not text:
            raise AliyunAPIError("百炼转写结果中没有文本")
        return TranscriptionResult(text=text, sentences=tuple(sentences_out))

    @staticmethod
    def _validate_transcription_result_url(url: str) -> None:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not hostname:
            raise AliyunAPIError("百炼返回了不安全的 transcription_url")
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise AliyunAPIError("百炼返回了不安全的 transcription_url")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return
        if not address.is_global:
            raise AliyunAPIError("百炼返回了不安全的 transcription_url")

    @classmethod
    def _load_transcription_json_sync(cls, result_url: str, timeout: float) -> dict[str, Any]:
        """Read an OSS signed URL without normalizing its query string."""
        cls._validate_transcription_result_url(result_url)
        request = Request(result_url, headers={"Accept": "application/json"}, method="GET")
        with urlopen(request, timeout=timeout) as response:
            cls._validate_transcription_result_url(response.geturl())
            content_length = response.headers.get("Content-Length", "").strip()
            if content_length.isdigit() and int(content_length) > _MAX_TRANSCRIPTION_JSON_BYTES:
                raise AliyunAPIError("百炼转写结果超过 32MB 安全上限")
            raw = response.read(_MAX_TRANSCRIPTION_JSON_BYTES + 1)
        if len(raw) > _MAX_TRANSCRIPTION_JSON_BYTES:
            raise AliyunAPIError("百炼转写结果超过 32MB 安全上限")
        try:
            data = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AliyunAPIError("百炼转写结果不是有效 JSON") from exc
        if not isinstance(data, dict):
            raise AliyunAPIError("百炼转写结果格式异常")
        return data

    async def _download_transcription_json(
        self,
        result_url: str,
        *,
        metrics: Optional[_RequestMetrics] = None,
    ) -> dict[str, Any]:
        if self._result_loader is not None:
            if metrics is not None:
                metrics.request_count += 1
            data = await self._result_loader(result_url)
            if not isinstance(data, dict):
                raise AliyunAPIError("百炼转写结果格式异常")
            return data

        for attempt in range(self.max_retries + 1):
            try:
                if metrics is not None:
                    metrics.request_count += 1
                return await asyncio.to_thread(
                    self._load_transcription_json_sync,
                    result_url,
                    min(self.timeout, 60.0),
                )
            except HTTPError as exc:
                if exc.code not in _RETRYABLE_STATUS or attempt >= self.max_retries:
                    raise AliyunAPIError(
                        f"百炼转写结果下载失败（HTTP {exc.code}）", exc.code
                    ) from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt >= self.max_retries:
                    raise AliyunAPIError(
                        f"百炼转写结果下载失败: {type(exc).__name__}"
                    ) from exc
            if metrics is not None:
                metrics.retry_count += 1
            await self._wait_before_retry(attempt, None)
        raise AliyunAPIError("百炼转写结果下载异常退出")

    async def transcribe_file_url(
        self,
        *,
        model: str,
        file_url: str,
        language_hints: Optional[list[str]] = None,
        channel_ids: Optional[list[int]] = None,
        poll_interval: float = 2.0,
        timeout: float = 1800.0,
        audio_duration_seconds: Optional[float] = None,
        operation: str = "asr_filetrans",
    ) -> str:
        """Run FileTrans under a real wall-clock deadline."""
        result = await self.transcribe_file_url_detailed(
            model=model,
            file_url=file_url,
            language_hints=language_hints,
            channel_ids=channel_ids,
            poll_interval=poll_interval,
            timeout=timeout,
            audio_duration_seconds=audio_duration_seconds,
            operation=operation,
        )
        return result.text

    async def transcribe_file_url_detailed(
        self,
        *,
        model: str,
        file_url: str,
        language_hints: Optional[list[str]] = None,
        channel_ids: Optional[list[int]] = None,
        poll_interval: float = 2.0,
        timeout: float = 1800.0,
        audio_duration_seconds: Optional[float] = None,
        operation: str = "asr_filetrans",
    ) -> TranscriptionResult:
        """Run FileTrans and return text with immutable sentence timestamps."""
        if poll_interval <= 0 or timeout <= 0:
            raise ValueError("poll_interval 和 timeout 必须大于 0")
        started_at = datetime.now(timezone.utc).isoformat()
        started_monotonic = time.monotonic()
        metrics = _RequestMetrics()
        try:
            result = await asyncio.wait_for(
                self._transcribe_file_url_detailed_impl(
                    model=model,
                    file_url=file_url,
                    language_hints=language_hints,
                    channel_ids=channel_ids,
                    poll_interval=poll_interval,
                    timeout=timeout,
                    metrics=metrics,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            error = AliyunAPIError("百炼文件转写等待超时（已达到墙钟上限）")
            await self._record_model_call(
                model=model,
                operation=operation,
                api_kind="native_filetrans",
                status="error",
                started_at=started_at,
                started_monotonic=started_monotonic,
                metrics=metrics,
                audio_seconds=audio_duration_seconds,
                error=error,
            )
            raise error from exc
        except BaseException as exc:
            await self._record_model_call(
                model=model,
                operation=operation,
                api_kind="native_filetrans",
                status="cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                started_at=started_at,
                started_monotonic=started_monotonic,
                metrics=metrics,
                audio_seconds=audio_duration_seconds,
                error=exc,
            )
            raise

        measured_seconds = audio_duration_seconds
        if measured_seconds is None:
            timed_ends = [
                sentence.end_time_ms
                for sentence in result.sentences
                if sentence.end_time_ms is not None
            ]
            if timed_ends:
                measured_seconds = max(timed_ends) / 1000.0
        await self._record_model_call(
            model=model,
            operation=operation,
            api_kind="native_filetrans",
            status="success",
            started_at=started_at,
            started_monotonic=started_monotonic,
            metrics=metrics,
            audio_seconds=measured_seconds,
        )
        return result

    async def _transcribe_file_url_impl(
        self,
        *,
        model: str,
        file_url: str,
        language_hints: Optional[list[str]] = None,
        channel_ids: Optional[list[int]] = None,
        poll_interval: float = 2.0,
        timeout: float = 1800.0,
        metrics: Optional[_RequestMetrics] = None,
    ) -> str:
        """Submit a public media URL to native DashScope and await its transcript."""
        result = await self._transcribe_file_url_detailed_impl(
            model=model,
            file_url=file_url,
            language_hints=language_hints,
            channel_ids=channel_ids,
            poll_interval=poll_interval,
            timeout=timeout,
            metrics=metrics,
        )
        return result.text

    async def _transcribe_file_url_detailed_impl(
        self,
        *,
        model: str,
        file_url: str,
        language_hints: Optional[list[str]] = None,
        channel_ids: Optional[list[int]] = None,
        poll_interval: float = 2.0,
        timeout: float = 1800.0,
        metrics: Optional[_RequestMetrics] = None,
    ) -> TranscriptionResult:
        """Submit a public media URL and await timestamp-preserving FileTrans output."""
        parsed_url = urlparse(file_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise AliyunAPIError("文件转写需要公网可访问的 http(s) URL")
        if poll_interval <= 0 or timeout <= 0:
            raise ValueError("poll_interval 和 timeout 必须大于 0")

        parameters: dict[str, Any] = {"channel_id": channel_ids or [0]}
        if language_hints:
            parameters["language_hints"] = language_hints
        submit_data = await self._request_json(
            "POST",
            self._native_url("services/audio/asr/transcription"),
            payload={
                "model": model,
                "input": {"file_urls": [file_url]},
                "parameters": parameters,
            },
            extra_headers={"X-DashScope-Async": "enable"},
            retry=False,
            metrics=metrics,
        )
        output = submit_data.get("output")
        if not isinstance(output, dict):
            raise AliyunAPIError("百炼文件转写提交响应缺少 output")
        task_id = output.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise AliyunAPIError("百炼文件转写提交响应缺少 task_id")
        if metrics is not None:
            metrics.provider_request_id = task_id.strip()[:256]

        max_polls = max(1, math.ceil(timeout / poll_interval))
        task_url = self._native_url(f"tasks/{quote(task_id.strip(), safe='')}")
        transcription_urls: list[str] = []
        last_status = "UNKNOWN"
        for poll_index in range(max_polls):
            task_data = await self._request_json("GET", task_url, metrics=metrics)
            task_output = task_data.get("output")
            if not isinstance(task_output, dict):
                raise AliyunAPIError("百炼文件转写任务响应缺少 output")

            last_status = str(task_output.get("task_status") or "UNKNOWN").upper()
            transcription_urls = self._extract_transcription_urls(task_output)
            if last_status == "SUCCEEDED" or (last_status == "UNKNOWN" and transcription_urls):
                break
            if last_status in {"FAILED", "CANCELED", "CANCELLED"}:
                detail = str(task_output.get("message") or task_data.get("message") or "")
                detail = " ".join(detail.split())[:200]
                suffix = f"：{detail}" if detail else ""
                raise AliyunAPIError(f"百炼文件转写任务失败（{last_status}）{suffix}")
            if poll_index + 1 < max_polls:
                await self._sleep(poll_interval)
        else:
            raise AliyunAPIError(
                f"百炼文件转写等待超时（最后状态 {last_status}）"
            )

        if not transcription_urls:
            raise AliyunAPIError("百炼文件转写成功，但未返回 transcription_url")

        transcripts: list[TranscriptionResult] = []
        for result_url in transcription_urls:
            self._validate_transcription_result_url(result_url)
            result_data = await self._download_transcription_json(
                result_url,
                metrics=metrics,
            )
            transcripts.append(self._extract_transcription_result(result_data))
        return TranscriptionResult(
            text="\n\n".join(result.text for result in transcripts),
            sentences=tuple(
                sentence
                for result in transcripts
                for sentence in result.sentences
            ),
        )


aliyun_client = AliyunModelClient(
    usage_store_factory=get_model_usage_store if MODEL_USAGE_LOG_ENABLED else None
)


async def close_aliyun_client() -> None:
    await aliyun_client.close()
