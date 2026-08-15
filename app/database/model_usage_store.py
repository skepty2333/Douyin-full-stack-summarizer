"""Privacy-safe model call telemetry and price-snapshot reporting."""
from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional

from app.config import MODEL_USAGE_DB_PATH


PRICING_VERSION = "aliyun-cn-beijing-2026-08-15"
_MILLION = Decimal("1000000")


@dataclass(frozen=True, slots=True)
class TextPriceTier:
    """CNY rates per one million tokens for one input-length tier."""

    max_input_tokens: int
    input_rate: Decimal
    cached_input_rate: Decimal
    output_rate: Decimal


# Dated China (Beijing) price snapshot. qwen3.7-plus's rolling alias uses the
# limited-time 20% rate shown on 2026-08-15; its dated snapshot retains the
# standard rate. Cached-input rates follow the provider console/docs snapshot.
_TEXT_PRICE_TIERS: dict[str, tuple[TextPriceTier, ...]] = {
    "qwen3.7-flash": (
        TextPriceTier(32_000, Decimal("0.2"), Decimal("0.04"), Decimal("0.8")),
        TextPriceTier(256_000, Decimal("0.6"), Decimal("0.12"), Decimal("2.4")),
        TextPriceTier(1_000_000, Decimal("1.2"), Decimal("0.24"), Decimal("4.8")),
    ),
    "qwen3.7-flash-2026-07-15": (
        TextPriceTier(32_000, Decimal("0.2"), Decimal("0.04"), Decimal("0.8")),
        TextPriceTier(256_000, Decimal("0.6"), Decimal("0.12"), Decimal("2.4")),
        TextPriceTier(1_000_000, Decimal("1.2"), Decimal("0.24"), Decimal("4.8")),
    ),
    "qwen3.7-plus": (
        TextPriceTier(256_000, Decimal("1.6"), Decimal("0.32"), Decimal("6.4")),
        TextPriceTier(1_000_000, Decimal("4.8"), Decimal("0.96"), Decimal("19.2")),
    ),
    "qwen3.7-plus-2026-05-26": (
        TextPriceTier(256_000, Decimal("2"), Decimal("0.4"), Decimal("8")),
        TextPriceTier(1_000_000, Decimal("6"), Decimal("1.2"), Decimal("24")),
    ),
    "qwen3.8-max": (
        TextPriceTier(1_000_000, Decimal("12"), Decimal("1.5"), Decimal("36")),
    ),
}
_AUDIO_PRICES: dict[str, Decimal] = {
    "qwen-audio-3.0-asr-flash-filetrans": Decimal("0.00022"),
    # The user requested that fallback ASR use the same duration-based rate.
    "qwen3-asr-flash": Decimal("0.00022"),
}


@dataclass(frozen=True, slots=True)
class ModelUsageContext:
    """Non-sensitive identifiers propagated through parallel pipeline tasks."""

    job_id: str = ""
    video_code: str = ""


_usage_context: ContextVar[ModelUsageContext] = ContextVar(
    "model_usage_context",
    default=ModelUsageContext(),
)


@contextmanager
def bind_model_usage_context(*, job_id: str, video_code: str) -> Iterator[None]:
    """Associate model calls with one bot job without storing user content."""

    token = _usage_context.set(
        ModelUsageContext(job_id=job_id[:128], video_code=video_code[:32])
    )
    try:
        yield
    finally:
        _usage_context.reset(token)


def current_model_usage_context() -> ModelUsageContext:
    return _usage_context.get()


@dataclass(frozen=True, slots=True)
class UsageNumbers:
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    cached_input_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    raw_json: str


@dataclass(frozen=True, slots=True)
class PriceEstimate:
    currency: Optional[str]
    estimated_cost: Optional[float]
    input_rate: Optional[float]
    cached_input_rate: Optional[float]
    output_rate: Optional[float]
    audio_rate: Optional[float]
    cost_status: str


def _nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and math.isfinite(value) and value >= 0 and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def extract_usage_numbers(payload: Optional[dict[str, Any]]) -> UsageNumbers:
    """Normalize Chat Completions and Responses API token usage shapes."""

    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = _nonnegative_int(
        usage.get("prompt_tokens", usage.get("input_tokens"))
    )
    output_tokens = _nonnegative_int(
        usage.get("completion_tokens", usage.get("output_tokens"))
    )
    total_tokens = _nonnegative_int(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    input_details = usage.get("prompt_tokens_details")
    if not isinstance(input_details, dict):
        input_details = usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    output_details = usage.get("completion_tokens_details")
    if not isinstance(output_details, dict):
        output_details = usage.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = {}

    # Store only the provider's usage object, never prompts or response text.
    raw_json = json.dumps(usage, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(raw_json.encode("utf-8")) > 8192:
        raw_json = "{}"
    return UsageNumbers(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=_nonnegative_int(input_details.get("cached_tokens")),
        reasoning_tokens=_nonnegative_int(output_details.get("reasoning_tokens")),
        raw_json=raw_json,
    )


def _match_price_key(model: str, prices: dict[str, Any]) -> Optional[str]:
    normalized = model.strip().lower()
    matches = [key for key in prices if normalized == key or normalized.startswith(key + "-")]
    return max(matches, key=len) if matches else None


def estimate_price(
    *,
    model: str,
    usage: UsageNumbers,
    audio_seconds: Optional[float],
    status: str,
) -> PriceEstimate:
    """Estimate one call with the dated China (Beijing) price snapshot."""

    audio_key = _match_price_key(model, _AUDIO_PRICES)
    if audio_key is not None:
        rate = _AUDIO_PRICES[audio_key]
        if status != "success":
            return PriceEstimate("CNY", None, None, None, None, float(rate), "failed_charge_unknown")
        if audio_seconds is None or not math.isfinite(audio_seconds) or audio_seconds < 0:
            return PriceEstimate("CNY", None, None, None, None, float(rate), "missing_audio_duration")
        cost = Decimal(str(audio_seconds)) * rate
        return PriceEstimate("CNY", float(cost), None, None, None, float(rate), "estimated")

    text_key = _match_price_key(model, _TEXT_PRICE_TIERS)
    if text_key is None:
        return PriceEstimate(None, None, None, None, None, None, "unpriced_model")
    if usage.input_tokens is None or usage.output_tokens is None:
        state = "failed_without_usage" if status != "success" else "missing_token_usage"
        return PriceEstimate("CNY", None, None, None, None, None, state)
    tier = next(
        (
            candidate
            for candidate in _TEXT_PRICE_TIERS[text_key]
            if usage.input_tokens <= candidate.max_input_tokens
        ),
        None,
    )
    if tier is None:
        return PriceEstimate("CNY", None, None, None, None, None, "outside_price_snapshot")
    cached = min(usage.cached_input_tokens or 0, usage.input_tokens)
    uncached = usage.input_tokens - cached
    cost = (
        Decimal(uncached) * tier.input_rate
        + Decimal(cached) * tier.cached_input_rate
        + Decimal(usage.output_tokens) * tier.output_rate
    ) / _MILLION
    return PriceEstimate(
        "CNY",
        float(cost),
        float(tier.input_rate),
        float(tier.cached_input_rate),
        float(tier.output_rate),
        None,
        "estimated",
    )


class ModelUsageStore:
    """Append-only SQLite store for model call telemetry."""

    def __init__(
        self,
        db_path: str = MODEL_USAGE_DB_PATH,
        *,
        read_only: bool = False,
    ):
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        self.read_only = read_only
        if not read_only:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        target = (
            f"{Path(self.db_path).as_uri()}?mode=ro"
            if self.read_only
            else self.db_path
        )
        conn = sqlite3.connect(target, timeout=10.0, uri=self.read_only)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT NOT NULL UNIQUE,
                    job_id TEXT NOT NULL DEFAULT '',
                    video_code TEXT NOT NULL DEFAULT '',
                    operation TEXT NOT NULL,
                    api_kind TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    provider_request_id TEXT NOT NULL DEFAULT '',
                    http_status INTEGER,
                    error_type TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    cached_input_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    audio_seconds REAL,
                    usage_json TEXT NOT NULL DEFAULT '{}',
                    pricing_version TEXT NOT NULL,
                    currency TEXT,
                    estimated_cost REAL,
                    input_rate_per_million REAL,
                    cached_input_rate_per_million REAL,
                    output_rate_per_million REAL,
                    audio_rate_per_second REAL,
                    cost_status TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_model_usage_started
                ON model_usage_log(started_at DESC, id DESC);

                CREATE INDEX IF NOT EXISTS idx_model_usage_job
                ON model_usage_log(job_id, started_at, id);

                CREATE INDEX IF NOT EXISTS idx_model_usage_model
                ON model_usage_log(model, operation, started_at);
                """
            )
            conn.commit()
        finally:
            conn.close()

    def record(
        self,
        *,
        model: str,
        operation: str,
        api_kind: str,
        status: str,
        started_at: str,
        finished_at: str,
        latency_ms: int,
        request_count: int,
        retry_count: int,
        provider_request_id: str = "",
        http_status: Optional[int] = None,
        error_type: str = "",
        error_message: str = "",
        response_payload: Optional[dict[str, Any]] = None,
        audio_seconds: Optional[float] = None,
    ) -> str:
        usage = extract_usage_numbers(response_payload)
        estimate = estimate_price(
            model=model,
            usage=usage,
            audio_seconds=audio_seconds,
            status=status,
        )
        context = current_model_usage_context()
        call_id = uuid.uuid4().hex
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO model_usage_log (
                    call_id, job_id, video_code, operation, api_kind, model,
                    status, started_at, finished_at, latency_ms, request_count,
                    retry_count, provider_request_id, http_status, error_type,
                    error_message, input_tokens, output_tokens, total_tokens,
                    cached_input_tokens, reasoning_tokens, audio_seconds,
                    usage_json, pricing_version, currency, estimated_cost,
                    input_rate_per_million, cached_input_rate_per_million,
                    output_rate_per_million, audio_rate_per_second, cost_status
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    call_id,
                    context.job_id,
                    context.video_code,
                    operation[:64],
                    api_kind[:32],
                    model[:128],
                    status[:16],
                    started_at,
                    finished_at,
                    max(0, int(latency_ms)),
                    max(0, int(request_count)),
                    max(0, int(retry_count)),
                    provider_request_id[:256],
                    http_status,
                    error_type[:128],
                    " ".join(error_message.split())[:500],
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    usage.cached_input_tokens,
                    usage.reasoning_tokens,
                    audio_seconds,
                    usage.raw_json,
                    PRICING_VERSION,
                    estimate.currency,
                    estimate.estimated_cost,
                    estimate.input_rate,
                    estimate.cached_input_rate,
                    estimate.output_rate,
                    estimate.audio_rate,
                    estimate.cost_status,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return call_id

    def health_check(self) -> bool:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='model_usage_log'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    @staticmethod
    def _cutoff(days: float) -> str:
        if not math.isfinite(days) or days <= 0 or days > 3660:
            raise ValueError("days 必须在 0 到 3660 之间")
        return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    def summary(self, *, days: float = 7.0) -> dict[str, Any]:
        cutoff = self._cutoff(days)
        conn = self._get_conn()
        try:
            overall = dict(
                conn.execute(
                    """
                    SELECT COUNT(*) AS calls,
                           COALESCE(SUM(CASE WHEN status='success' THEN 1 ELSE 0 END), 0) AS successes,
                           COALESCE(SUM(CASE WHEN status!='success' THEN 1 ELSE 0 END), 0) AS failures,
                           COALESCE(SUM(COALESCE(input_tokens, 0)), 0) AS input_tokens,
                           COALESCE(SUM(COALESCE(output_tokens, 0)), 0) AS output_tokens,
                           COALESCE(SUM(COALESCE(cached_input_tokens, 0)), 0) AS cached_input_tokens,
                           COALESCE(SUM(COALESCE(reasoning_tokens, 0)), 0) AS reasoning_tokens,
                           COALESCE(SUM(COALESCE(audio_seconds, 0)), 0) AS audio_seconds,
                           COALESCE(SUM(CASE WHEN estimated_cost IS NULL THEN 1 ELSE 0 END), 0) AS cost_unknown_calls
                    FROM model_usage_log WHERE started_at >= ?
                    """,
                    (cutoff,),
                ).fetchone()
            )
            cost_rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT currency, SUM(estimated_cost) AS estimated_cost
                    FROM model_usage_log
                    WHERE started_at >= ? AND estimated_cost IS NOT NULL
                    GROUP BY currency ORDER BY currency
                    """,
                    (cutoff,),
                ).fetchall()
            ]
            pricing_versions = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT pricing_version, COUNT(*) AS calls
                    FROM model_usage_log
                    WHERE started_at >= ?
                    GROUP BY pricing_version ORDER BY pricing_version
                    """,
                    (cutoff,),
                ).fetchall()
            ]
            groups = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT model, operation, COUNT(*) AS calls,
                           SUM(CASE WHEN status!='success' THEN 1 ELSE 0 END) AS failures,
                           SUM(COALESCE(input_tokens, 0)) AS input_tokens,
                           SUM(COALESCE(output_tokens, 0)) AS output_tokens,
                           SUM(COALESCE(cached_input_tokens, 0)) AS cached_input_tokens,
                           SUM(COALESCE(reasoning_tokens, 0)) AS reasoning_tokens,
                           SUM(COALESCE(audio_seconds, 0)) AS audio_seconds,
                           currency, SUM(estimated_cost) AS estimated_cost,
                           SUM(CASE WHEN estimated_cost IS NULL THEN 1 ELSE 0 END) AS cost_unknown_calls
                    FROM model_usage_log WHERE started_at >= ?
                    GROUP BY model, operation, currency
                    ORDER BY model, operation
                    """,
                    (cutoff,),
                ).fetchall()
            ]
            return {
                "since": cutoff,
                "pricing_version": PRICING_VERSION,
                "pricing_versions": pricing_versions,
                "overall": overall,
                "costs": cost_rows,
                "groups": groups,
            }
        finally:
            conn.close()

    def recent(
        self,
        *,
        days: float = 7.0,
        limit: int = 100,
        job_id: str = "",
    ) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
            raise ValueError("limit 必须在 1 到 5000 之间")
        cutoff = self._cutoff(days)
        where = "started_at >= ?"
        params: list[Any] = [cutoff]
        if job_id:
            where += " AND job_id = ?"
            params.append(job_id)
        params.append(limit)
        conn = self._get_conn()
        try:
            return [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT id, call_id, job_id, video_code, operation, api_kind,
                           model, status, started_at, finished_at, latency_ms,
                           request_count, retry_count, provider_request_id,
                           http_status, error_type, error_message, input_tokens,
                           output_tokens, total_tokens, cached_input_tokens,
                           reasoning_tokens, audio_seconds, pricing_version,
                           currency, estimated_cost, cost_status
                    FROM model_usage_log WHERE {where}
                    ORDER BY started_at DESC, id DESC LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
            ]
        finally:
            conn.close()


@lru_cache(maxsize=1)
def get_model_usage_store() -> ModelUsageStore:
    return ModelUsageStore(MODEL_USAGE_DB_PATH)
