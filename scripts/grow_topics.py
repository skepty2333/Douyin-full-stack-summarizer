#!/usr/bin/env python3
"""Let topics grow: birth dense vocabulary entries, split oversized ones,
compile new or dirty pages. Safe to rerun; nothing is compiled twice unless
it is dirty.

    venv/bin/python scripts/grow_topics.py --dry-run     # show what would be born / split / compiled
    venv/bin/python scripts/grow_topics.py               # grow
    venv/bin/python scripts/grow_topics.py --compile 主题名   # force one page
    venv/bin/python scripts/grow_topics.py --list        # current landscape
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, KNOWLEDGE_DB_PATH, TOPIC_MAX_NOTES  # noqa: E402
from app.database.knowledge_store import KnowledgeStore  # noqa: E402
from app.database.note_index import NoteIndex  # noqa: E402
from app.database.topics import TopicStore  # noqa: E402
from app.database.vocabulary import VocabularyStore  # noqa: E402


def _print_landscape(topics: TopicStore) -> None:
    rows = topics.list_topics(statuses=("candidate", "active", "archived"))
    if not rows:
        print("（还没有主题）")
        return
    by_id = {t.id: t for t in rows}
    print(f"{'主题':<18}{'状态':<10}{'笔记':>5}{'作者':>5}{'版本':>5}{'脏':>4}  备注")
    for t in sorted(rows, key=lambda t: (t.parent_id or t.id, t.parent_id is not None, -t.note_count)):
        parent = by_id.get(t.parent_id) if t.parent_id else None
        note = "枢纽" if t.is_hub else ""
        if parent:
            note = f"← {parent.name}"
        name = ("  " if parent else "") + t.name
        print(f"{name:<18}{t.status:<10}{t.note_count:>5}{t.author_count:>5}{t.latest_version:>5}{t.dirty:>4}  {note}")
    for s in topics.merge_suggestions():
        print(f"建议合并: {s.topic_a.name} ↔ {s.topic_b.name}（成员重叠 {s.overlap:.0%}）")


async def run(args: argparse.Namespace) -> int:
    from app.services.aliyun_client import aliyun_client, close_aliyun_client
    from app.services.topic_compiler import TopicCompiler

    store = KnowledgeStore(args.db)
    vocab = VocabularyStore(args.db)
    topics = TopicStore(args.db, vocabulary=vocab)

    async def embed(texts):
        return await aliyun_client.embed(model=EMBEDDING_MODEL, texts=texts, dimensions=EMBEDDING_DIMENSIONS, operation="embedding_query")

    index = NoteIndex(args.db, embed_fn=embed, alias_groups_fn=vocab.alias_groups)
    compiler = TopicCompiler(store=store, index=index, topics=topics)
    try:
        if args.list:
            _print_landscape(topics)
            return 0
        if args.dry_run:
            eligible = topics.eligible_entries()
            print(f"将出生的主题 {len(eligible)} 个:")
            for e in eligible:
                flag = "（超过上限，出生后会拆分）" if e.note_count > TOPIC_MAX_NOTES else ""
                print(f"  {e.canonical:<16} {e.note_count:>3} 条 / {e.author_count:>2} 位作者 [{e.kind}]{flag}")
            pending = [t for t in topics.list_topics() if t.latest_version == 0 or t.dirty]
            print(f"待编译的已有主题 {len(pending)} 个: " + ", ".join(t.name for t in pending))
            return 0
        if args.compile:
            topic = topics.resolve(args.compile)
            if topic is None:
                entry = vocab.resolve(args.compile)
                if entry is None:
                    print(f"未知主题: {args.compile}")
                    return 1
                topic = topics.born(entry, reason="手动编译")
            if topics.needs_split(topic):
                result = await compiler.split(topic)
                print(f"已拆分 {topic.name} → {', '.join(c.name for c in result.children)}")
                for child in result.children:
                    if child.latest_version == 0:
                        r = await compiler.compile(child)
                        print(f"  已编译 {child.name} v{r.version.version}")
                topic = topics.get(topic.id) or topic
            result = await compiler.compile(topic)
            print(f"已编译 {result.topic.name} v{result.version.version}: 成员 {result.member_notes} · 外部 {result.outside_notes} · 未引用结论 {result.uncited_lines}")
            print(result.version.markdown[:1500])
            return 0
        report = await compiler.grow()
        print(f"出生: {report['born']}")
        for name, children in report["split"]:
            print(f"拆分: {name} → {children}")
        for name, version in report["compiled"]:
            print(f"编译: {name} v{version}")
        for name, why in report["failed"]:
            print(f"失败: {name} ({why})")
        print()
        _print_landscape(topics)
        return 0
    finally:
        await close_aliyun_client()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=KNOWLEDGE_DB_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--compile", default=None, help="强制编译一个主题（名称或别名）")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
