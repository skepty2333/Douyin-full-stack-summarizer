#!/usr/bin/env python3
"""Fill ``knowledge.published_at`` for notes ingested before it was recorded.

Each note needs one Douyin page/detail request (no video download). Requests
are spaced with a random delay to stay polite; failures are logged and left
empty so a rerun only retries what is still missing.

    venv/bin/python scripts/backfill_publish_dates.py            # all missing
    venv/bin/python scripts/backfill_publish_dates.py --limit 20
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import KNOWLEDGE_DB_PATH  # noqa: E402
from app.database.knowledge_store import KnowledgeStore  # noqa: E402
from app.services.douyin_parser import resolve_metadata  # noqa: E402


async def run(args: argparse.Namespace) -> int:
    store = KnowledgeStore(args.db)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, video_code, source_url, video_id FROM knowledge WHERE published_at = '' ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    if args.limit:
        rows = rows[: args.limit]
    print(f"缺少发布时间的笔记: {len(rows)}")
    ok = failed = mismatched = 0
    started = time.monotonic()
    for position, row in enumerate(rows, 1):
        try:
            info = await resolve_metadata(row["source_url"])
        except Exception as exc:  # network / parse / anti-bot: leave empty, continue
            failed += 1
            print(f"  [{row['video_code']}] 失败: {type(exc).__name__}")
        else:
            if info["video_id"] and row["video_id"] and info["video_id"] != row["video_id"]:
                mismatched += 1
                print(f"  [{row['video_code']}] 视频 ID 不一致，跳过")
            elif info["published_at"]:
                store.update_metadata(int(row["id"]), published_at=info["published_at"])
                ok += 1
            else:
                failed += 1
                print(f"  [{row['video_code']}] 详情中没有发布时间")
        if position % 20 == 0:
            print(f"  进度 {position}/{len(rows)} · 成功 {ok} · 失败 {failed}（{time.monotonic() - started:.0f} 秒）")
        if position < len(rows):
            await asyncio.sleep(random.uniform(args.min_delay, args.max_delay))
    print(f"完成: 成功 {ok} · 失败 {failed} · ID 不一致 {mismatched} · 耗时 {time.monotonic() - started:.0f} 秒")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=KNOWLEDGE_DB_PATH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-delay", type=float, default=2.0)
    parser.add_argument("--max-delay", type=float, default=5.0)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
