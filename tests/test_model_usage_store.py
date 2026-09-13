"""Offline tests for privacy-safe model telemetry and price snapshots."""
from __future__ import annotations

import json
import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx

from app.database.model_usage_store import (
    ModelUsageStore,
    bind_model_usage_context,
    estimate_price,
    extract_usage_numbers,
)
from app.services.aliyun_client import AliyunAPIError, AliyunModelClient


class ModelUsagePriceTests(unittest.TestCase):
    def test_text_price_splits_cached_and_uncached_tokens(self) -> None:
        usage = extract_usage_numbers(
            {
                "usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 100_000,
                    "total_tokens": 1_100_000,
                    "prompt_tokens_details": {"cached_tokens": 200_000},
                    "completion_tokens_details": {"reasoning_tokens": 40_000},
                }
            }
        )
        estimate = estimate_price(
            model="qwen3.8-max-2026-08-01",
            usage=usage,
            audio_seconds=None,
            status="success",
        )

        self.assertEqual(usage.cached_input_tokens, 200_000)
        self.assertEqual(usage.reasoning_tokens, 40_000)
        self.assertEqual(estimate.currency, "CNY")
        self.assertAlmostEqual(estimate.estimated_cost or 0, 13.5)

    def test_audio_price_uses_duration_and_failed_call_is_unknown(self) -> None:
        empty_usage = extract_usage_numbers(None)
        success = estimate_price(
            model="qwen3-asr-flash",
            usage=empty_usage,
            audio_seconds=12.5,
            status="success",
        )
        failed = estimate_price(
            model="qwen-audio-3.0-asr-flash-filetrans",
            usage=empty_usage,
            audio_seconds=12.5,
            status="error",
        )

        self.assertEqual(success.currency, "CNY")
        self.assertAlmostEqual(success.estimated_cost or 0, 0.00275)
        self.assertIsNone(failed.estimated_cost)
        self.assertEqual(failed.cost_status, "failed_charge_unknown")

    def test_usage_selects_price_tier_by_input_length(self) -> None:
        usage = extract_usage_numbers(
            {"usage": {"input_tokens": 32_001, "output_tokens": 1}}
        )
        estimate = estimate_price(
            model="qwen3.7-flash",
            usage=usage,
            audio_seconds=None,
            status="success",
        )

        expected = (32_001 * 0.6 + 2.4) / 1_000_000
        self.assertAlmostEqual(estimate.estimated_cost or 0, expected)
        self.assertEqual(estimate.currency, "CNY")
        self.assertEqual(estimate.cost_status, "estimated")


class ModelUsagePersistenceTests(unittest.TestCase):
    def test_empty_summary_uses_zero_not_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ModelUsageStore(str(Path(directory) / "usage.db"))
            overall = store.summary(days=1)["overall"]

        self.assertEqual(overall["calls"], 0)
        self.assertEqual(overall["successes"], 0)
        self.assertEqual(overall["audio_seconds"], 0)

    def test_records_only_usage_metadata_with_job_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ModelUsageStore(str(Path(directory) / "usage.db"))
            with bind_model_usage_context(job_id="job-123", video_code="abc12"):
                store.record(
                    model="qwen3.7-plus",
                    operation="research",
                    api_kind="chat_completions",
                    status="success",
                    started_at="2026-08-10T00:00:00+00:00",
                    finished_at="2026-08-10T00:00:01+00:00",
                    latency_ms=1000,
                    request_count=2,
                    retry_count=1,
                    provider_request_id="provider-1",
                    response_payload={
                        "choices": [{"message": {"content": "must not persist"}}],
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 200,
                            "input_tokens_details": {"cached_tokens": 100},
                        },
                    },
                )

            rows = store.recent(days=3650, limit=10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["job_id"], "job-123")
            self.assertEqual(rows[0]["video_code"], "abc12")
            self.assertEqual(rows[0]["operation"], "research")
            self.assertEqual(rows[0]["request_count"], 2)
            self.assertEqual(rows[0]["retry_count"], 1)
            db_bytes = (Path(directory) / "usage.db").read_bytes()
            self.assertNotIn(b"must not persist", db_bytes)


class InstrumentedClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_records_usage_request_id_and_retry(self) -> None:
        requests = 0

        async def no_sleep(_delay: float) -> None:
            return None

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            if requests == 1:
                return httpx.Response(429, json={"error": {"message": "retry"}})
            return httpx.Response(
                200,
                headers={"x-request-id": "req-usage-1"},
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 100,
                        "total_tokens": 1100,
                        "prompt_tokens_details": {"cached_tokens": 250},
                    },
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            store = ModelUsageStore(str(Path(directory) / "usage.db"))
            client = AliyunModelClient(
                api_key="offline-key",
                base_url="https://model.invalid/v1",
                native_base_url="https://native.invalid/api/v1",
                timeout=1,
                max_retries=1,
                max_concurrency=1,
                transport=httpx.MockTransport(handler),
                sleep=no_sleep,
                usage_store=store,
            )
            try:
                with bind_model_usage_context(job_id="job-async", video_code="vv001"):
                    call_task = asyncio.create_task(
                        client.chat(
                            model="qwen3.7-flash",
                            operation="tagging",
                            messages=[{"role": "user", "content": "private prompt"}],
                        )
                    )
                    result = await call_task
            finally:
                await client.close()

            self.assertEqual(result, "ok")
            row = store.recent(days=1, limit=1)[0]
            self.assertEqual(row["job_id"], "job-async")
            self.assertEqual(row["provider_request_id"], "req-usage-1")
            self.assertEqual(row["request_count"], 2)
            self.assertEqual(row["retry_count"], 1)
            self.assertEqual(row["cached_input_tokens"], 250)
            self.assertEqual(row["cost_status"], "estimated")

    async def test_api_error_is_recorded_without_prompt_content(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                401,
                json={"error": {"message": "invalid credential"}},
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ModelUsageStore(str(Path(directory) / "usage.db"))
            client = AliyunModelClient(
                api_key="offline-key",
                base_url="https://model.invalid/v1",
                native_base_url="https://native.invalid/api/v1",
                timeout=1,
                max_retries=0,
                max_concurrency=1,
                transport=transport,
                usage_store=store,
            )
            try:
                with self.assertRaises(AliyunAPIError):
                    await client.chat(
                        model="qwen3.8-max",
                        operation="final_edit",
                        messages=[{"role": "user", "content": "secret user content"}],
                    )
            finally:
                await client.close()

            row = store.recent(days=1, limit=1)[0]
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["http_status"], 401)
            self.assertEqual(row["error_type"], "AliyunAPIError")
            self.assertEqual(row["cost_status"], "failed_without_usage")
            self.assertNotIn("secret user content", json.dumps(row))


if __name__ == "__main__":
    unittest.main()
