#!/usr/bin/env python3
"""Build the seed vocabulary from the library's existing free-form tags.

Reads every tag in ``knowledge.tags`` with its frequency, asks the final-stage
model to cluster them into canonical entries (topics / entities / content
types) with aliases, and loads the result into the vocabulary tables with
``source='seed'``. Re-running is safe: existing entries only gain aliases.

    venv/bin/python scripts/seed_vocabulary.py                # build and load
    venv/bin/python scripts/seed_vocabulary.py --dry-run      # print the proposal only
    venv/bin/python scripts/seed_vocabulary.py --from FILE    # load a saved proposal
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ALIYUN_FINAL_MODEL, KNOWLEDGE_DB_PATH  # noqa: E402
from app.database.vocabulary import KINDS, VocabularyStore  # noqa: E402

SYSTEM_PROMPT = """你是个人视频知识库的词表编辑。下面是这个库现有的全部自由标签及其出现次数（约 250 条笔记，约一半是 AI/Agent/编程，约 15% 是交易与量化，其余是学习方法、健身、职业等生活内容）。

请把它们整理成一份受控词表，输出 JSON：
{"entries": [{"canonical": "规范名", "kind": "topic|entity|content_type", "aliases": ["别名1", "别名2"]}]}

要求：
1. 规范名的颗粒度以"值得为它写一页综述"为准：既不要像"AI""技术"这样大到无法综述，也不要细到只对应一条视频。例如 Agent 应拆成 Agent 架构、Agent 记忆、多智能体协作、Agent 安全 等；量化交易、价格行为、风险管理、交易心理 各自独立。
2. kind：topic = 主题、概念、方法领域；entity = 具体产品、框架、模型、公司、人物、指标（如 OpenClaw、Claude Code、Codex、Hermes、MCP、RAG 可作为 entity 或 topic，按最自然的方式）；content_type = 内容形式（教程、评测、新闻、访谈、面试题、开源项目）。
3. aliases 只收录明确同义或拼写变体（如 智能体/AI Agent/AI智能体 → Agent），不要把上下位概念当别名；别名可以直接引用给定标签。
4. 泛化空话（效率提升、信息差、干货分享、自我提升、技术教程之外的"XX技巧"）一律不收录，也不要作为别名。
5. 高频标签必须被覆盖（作为规范名或别名）；低频标签只在明显属于某个条目时才收录，其余忽略。
6. 总量控制在 60 到 130 个条目。规范名用中文或通用英文名，简短（2 到 12 个字），不带标点。

只输出 JSON。"""


def collect_tags(db_path: str) -> collections.Counter:
    import sqlite3

    conn = sqlite3.connect(db_path)
    counter: collections.Counter = collections.Counter()
    try:
        for (tags,) in conn.execute("SELECT tags FROM knowledge"):
            for tag in re.split(r"[,，]", tags or ""):
                tag = tag.strip().lstrip("#").strip()
                if tag:
                    counter[tag] += 1
    finally:
        conn.close()
    return counter


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


async def propose(counter: collections.Counter, model: str) -> dict:
    from app.services.aliyun_client import aliyun_client, close_aliyun_client

    listing = "\n".join(f"{tag}\t{count}" for tag, count in counter.most_common())
    try:
        raw = await aliyun_client.chat(
            model=model,
            operation="seed_vocabulary",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"标签\t次数\n{listing}"},
            ],
            max_tokens=16000,
            temperature=0.2,
            enable_thinking=True,
            thinking_budget=4096,
        )
    finally:
        await close_aliyun_client()
    return _extract_json(raw)


def load(proposal: dict, db_path: str) -> tuple[int, int]:
    vocab = VocabularyStore(db_path)
    created = skipped = 0
    for item in proposal.get("entries") or []:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical", "")).strip()
        kind = str(item.get("kind", "topic")).strip().lower()
        if kind not in KINDS or kind == "vacuous" or len(canonical) < 2:
            skipped += 1
            continue
        aliases = [str(a).strip() for a in (item.get("aliases") or []) if str(a).strip()]
        before = vocab.resolve(canonical)
        vocab.upsert_entry(canonical, kind=kind, aliases=aliases, source="seed")
        if before is None:
            created += 1
    return created, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=KNOWLEDGE_DB_PATH)
    parser.add_argument("--model", default=ALIYUN_FINAL_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="只打印提案，不写入")
    parser.add_argument("--save", default=None, help="把提案 JSON 保存到文件")
    parser.add_argument("--from", dest="from_file", default=None, help="从已保存的提案文件加载，不调用模型")
    args = parser.parse_args()

    if args.from_file:
        proposal = json.loads(Path(args.from_file).read_text(encoding="utf-8"))
    else:
        counter = collect_tags(args.db)
        print(f"现有标签: {len(counter)} 个不同标签，{sum(counter.values())} 次使用")
        proposal = asyncio.run(propose(counter, args.model))
        if args.save:
            Path(args.save).write_text(json.dumps(proposal, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"提案已保存: {args.save}")

    entries = proposal.get("entries") or []
    by_kind = collections.Counter(str(e.get("kind")) for e in entries if isinstance(e, dict))
    print(f"提案条目: {len(entries)}  按类型: {dict(by_kind)}")
    for e in entries:
        if isinstance(e, dict):
            aliases = e.get("aliases") or []
            print(f"  [{e.get('kind')}] {e.get('canonical')}" + (f"  ← {' / '.join(map(str, aliases[:8]))}" if aliases else ""))
    if args.dry_run:
        return 0
    created, skipped = load(proposal, args.db)
    stats = VocabularyStore(args.db).stats()
    print(f"已载入: 新建 {created} · 跳过 {skipped} · 当前活跃条目 {stats['active_entries']} · 别名 {stats['aliases']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
