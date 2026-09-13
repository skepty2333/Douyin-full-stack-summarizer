#!/usr/bin/env python3
"""Build or refresh the section index (chunks + vectors) for all notes.

The index is derived data: running this script is always safe and idempotent.
Sections whose text has not changed keep their existing vectors, so a rerun
after new notes arrive only pays for the new sections.

    venv/bin/python scripts/build_note_index.py            # chunk missing notes, embed pending
    venv/bin/python scripts/build_note_index.py --rebuild  # re-chunk everything first
    venv/bin/python scripts/build_note_index.py --no-embed # chunk only, no API calls
    venv/bin/python scripts/build_note_index.py --stats    # report only
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, KNOWLEDGE_DB_PATH  # noqa: E402
from app.database.note_index import NoteIndex  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_note_index")


def _print_stats(index: NoteIndex) -> None:
    stats = index.stats()
    print(
        f"索引状态: 笔记 {stats['indexed_notes']} · 章节 {stats['chunks']} · "
        f"向量 {stats['vectors']} · 待向量化 {stats['pending_embeddings']} · "
        f"模型 {stats['model']}@{stats['dimensions']}"
    )


async def _embed_all(index: NoteIndex, limit: int | None) -> int:
    total = 0
    started = time.monotonic()
    while True:
        step = await index.embed_pending(limit=100 if limit is None else min(100, limit - total))
        total += step
        if step:
            logger.info("已向量化 %s 段（%.0f 秒）", total, time.monotonic() - started)
        if step == 0 or (limit is not None and total >= limit):
            return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=KNOWLEDGE_DB_PATH, help="知识库 SQLite 路径")
    parser.add_argument("--rebuild", action="store_true", help="重新切分所有笔记（向量按文本哈希复用）")
    parser.add_argument("--no-embed", action="store_true", help="只切分章节，不调用向量接口")
    parser.add_argument("--limit", type=int, default=None, help="本次最多向量化的章节数")
    parser.add_argument("--stats", action="store_true", help="只打印索引状态")
    args = parser.parse_args()

    embed_fn = None
    if not args.no_embed and not args.stats:
        from app.services.aliyun_client import aliyun_client

        async def embed_fn(texts):
            return await aliyun_client.embed(
                model=EMBEDDING_MODEL,
                texts=texts,
                dimensions=EMBEDDING_DIMENSIONS,
                operation="embedding_index",
            )

    index = NoteIndex(args.db, embed_fn=embed_fn)
    if args.stats:
        _print_stats(index)
        return 0

    report = index.index_all(only_missing=not args.rebuild)
    print(f"切分完成: 笔记 {report.notes_indexed} · 章节 {report.chunks_written} · 待向量化 {report.embeddings_pending}")

    if embed_fn is not None and report.embeddings_pending:
        async def run() -> int:
            from app.services.aliyun_client import close_aliyun_client

            try:
                return await _embed_all(index, args.limit)
            finally:
                await close_aliyun_client()

        embedded = asyncio.run(run())
        print(f"向量化完成: {embedded} 段")
    _print_stats(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
