#!/usr/bin/env python3
"""Read-only report for the model usage observation log."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database.model_usage_store import ModelUsageStore  # noqa: E402
from app.config import MODEL_USAGE_DB_PATH  # noqa: E402


def _integer(value: Any) -> int:
    return int(value or 0)


def _number(value: Any, digits: int = 6) -> str:
    return f"{float(value or 0):.{digits}f}"


def _print_summary(report: dict[str, Any], *, days: float) -> None:
    overall = report["overall"]
    print(f"模型调用观测报告（最近 {days:g} 天）")
    print(f"当前估价规则: {report['pricing_version']}")
    versions = report.get("pricing_versions", [])
    if versions:
        version_text = "；".join(
            f"{row['pricing_version']} ({_integer(row['calls'])} 次)" for row in versions
        )
        print(f"区间内记录版本: {version_text}")
    print(
        "调用: "
        f"{_integer(overall['calls'])} 次，"
        f"成功 {_integer(overall['successes'])}，"
        f"失败/取消 {_integer(overall['failures'])}，"
        f"费用未知 {_integer(overall['cost_unknown_calls'])}"
    )
    print(
        "用量: "
        f"输入 {_integer(overall['input_tokens'])} tokens，"
        f"缓存 {_integer(overall['cached_input_tokens'])}，"
        f"输出 {_integer(overall['output_tokens'])}，"
        f"推理 {_integer(overall['reasoning_tokens'])}，"
        f"音频 {_number(overall['audio_seconds'], 2)} 秒"
    )
    cost_parts = [
        f"{row['currency']} {_number(row['estimated_cost'], 6)}"
        for row in report["costs"]
    ]
    print(f"估算费用: {'；'.join(cost_parts) if cost_parts else '暂无可估算费用'}")
    print()
    print("按模型与环节:")
    for row in report["groups"]:
        currency = row.get("currency") or "-"
        cost = (
            "未知"
            if row.get("estimated_cost") is None
            else f"{currency} {_number(row['estimated_cost'], 6)}"
        )
        print(
            f"- {row['model']} / {row['operation']}: "
            f"{_integer(row['calls'])} 次，失败 {_integer(row['failures'])}，"
            f"输入 {_integer(row['input_tokens'])}，缓存 {_integer(row['cached_input_tokens'])}，"
            f"输出 {_integer(row['output_tokens'])}，推理 {_integer(row['reasoning_tokens'])}，"
            f"音频 {_number(row['audio_seconds'], 2)} 秒，费用 {cost}，"
            f"费用未知 {_integer(row['cost_unknown_calls'])} 次"
        )


def _print_details(rows: list[dict[str, Any]]) -> None:
    print()
    print(f"最近调用明细（{len(rows)} 条）:")
    for row in rows:
        cost = (
            "未知"
            if row.get("estimated_cost") is None
            else f"{row.get('currency')} {_number(row['estimated_cost'], 8)}"
        )
        error = ""
        if row["status"] != "success":
            error = f" error={row['error_type']}:{row['error_message']}"
        print(
            f"- {row['started_at']} job={row['job_id'] or '-'} video={row['video_code'] or '-'} "
            f"{row['model']}/{row['operation']} status={row['status']} "
            f"latency={row['latency_ms']}ms requests={row['request_count']} retries={row['retry_count']} "
            f"in={row['input_tokens']} cached={row['cached_input_tokens']} "
            f"out={row['output_tokens']} reasoning={row['reasoning_tokens']} "
            f"audio_s={row['audio_seconds']} cost={cost} cost_status={row['cost_status']}"
            f" pricing_version={row['pricing_version']}"
            f"{error}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="查看模型调用 token、时长与估算费用")
    parser.add_argument("--days", type=float, default=7.0, help="统计最近 N 天，默认 7")
    parser.add_argument("--details", action="store_true", help="同时打印逐次调用明细")
    parser.add_argument("--limit", type=int, default=100, help="明细条数，默认 100，最大 5000")
    parser.add_argument("--job-id", default="", help="明细只查看指定任务 ID")
    parser.add_argument("--db", default=MODEL_USAGE_DB_PATH, help="SQLite 路径")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()

    try:
        store = ModelUsageStore(args.db, read_only=True)
        report = store.summary(days=args.days)
        rows = (
            store.recent(days=args.days, limit=args.limit, job_id=args.job_id)
            if args.details or args.json
            else []
        )
    except sqlite3.Error as exc:
        print(f"无法读取模型调用日志: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"summary": report, "details": rows}, ensure_ascii=False, indent=2))
    else:
        _print_summary(report, days=args.days)
        if args.details:
            _print_details(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
