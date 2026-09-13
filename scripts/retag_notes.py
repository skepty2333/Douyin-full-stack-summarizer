#!/usr/bin/env python3
"""Classify notes (domain, temporality) and re-tag them against the vocabulary.

By default only notes without vocabulary links are processed, so reruns are
cheap. ``--all`` re-classifies everything (for example after the vocabulary
was reorganised). Each note costs one flash-tier call.

    venv/bin/python scripts/retag_notes.py               # untagged notes only
    venv/bin/python scripts/retag_notes.py --all         # every note
    venv/bin/python scripts/retag_notes.py --limit 5 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import AI_MAX_CONCURRENCY, KNOWLEDGE_DB_PATH  # noqa: E402
from app.database.knowledge_store import KnowledgeStore  # noqa: E402
from app.database.vocabulary import VocabularyStore  # noqa: E402
from app.services.note_tagging import apply_classification, classify_note  # noqa: E402


async def run(args: argparse.Namespace) -> int:
    from app.services.aliyun_client import close_aliyun_client

    store = KnowledgeStore(args.db)
    vocab = VocabularyStore(args.db)
    if args.all:
        import sqlite3

        conn = sqlite3.connect(args.db)
        try:
            ids = [int(r[0]) for r in conn.execute("SELECT id FROM knowledge ORDER BY id")]
        finally:
            conn.close()
    else:
        ids = vocab.untagged_note_ids()
    if args.limit:
        ids = ids[: args.limit]
    print(f"待处理笔记: {len(ids)}")
    if not ids:
        return 0

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    domains: collections.Counter = collections.Counter()
    temporalities: collections.Counter = collections.Counter()
    created: list[str] = []
    failed: list[int] = []
    done = 0
    started = time.monotonic()

    async def handle(note_id: int) -> None:
        nonlocal done
        async with semaphore:
            note = store.get_by_id(note_id)
            if note is None:
                return
            result = await classify_note(
                note["summary_markdown"], note["title"], note["author"], vocabulary=vocab
            )
            if result is None:
                failed.append(note_id)
                return
            if args.dry_run:
                print(
                    f"  [{note['video_code']}] {note['title'][:32]} → {result.domain}/{result.temporality} · "
                    + ", ".join(f"{t.name}({t.kind[0]})" for t in result.tags)
                )
            else:
                applied = apply_classification(note_id, result, store=store, vocabulary=vocab)
                created.extend(applied.created)
            domains[result.domain] += 1
            temporalities[result.temporality] += 1
            done += 1
            if done % 25 == 0:
                print(f"  已处理 {done}/{len(ids)}（{time.monotonic() - started:.0f} 秒）")

    try:
        await asyncio.gather(*(handle(note_id) for note_id in ids))
    finally:
        await close_aliyun_client()

    print(f"完成: 成功 {done} · 失败 {len(failed)} · 耗时 {time.monotonic() - started:.0f} 秒")
    print(f"domain: {dict(domains)}")
    print(f"temporality: {dict(temporalities)}")
    if created:
        counts = collections.Counter(created)
        print(f"新建词表条目 {len(counts)} 个: " + ", ".join(f"{n}×{c}" if c > 1 else n for n, c in counts.most_common()))
    if failed:
        print(f"失败笔记 ID: {failed}")
    print(vocab.stats())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=KNOWLEDGE_DB_PATH)
    parser.add_argument("--all", action="store_true", help="重新分类所有笔记，而不只是未打标签的")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条")
    parser.add_argument("--concurrency", type=int, default=AI_MAX_CONCURRENCY)
    parser.add_argument("--dry-run", action="store_true", help="只打印分类结果，不写入")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
