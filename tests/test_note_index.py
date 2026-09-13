"""Offline tests for the section index and hybrid retrieval."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-key"}, clear=False):
    with patch("dotenv.load_dotenv", return_value=False):
        from app.database.knowledge_store import KnowledgeEntry, KnowledgeStore
        from app.database.note_index import NoteIndex, split_terms


VOCAB = ["记忆", "架构", "止损", "量化", "学习", "咖啡", "openclaw", "serena"]


async def toy_embed(texts):
    """Bag-of-words over a tiny vocabulary: deterministic 'semantics' for tests."""
    vectors = []
    for text in texts:
        lowered = text.lower()
        vectors.append([0.01 + lowered.count(word) for word in VOCAB])
    return vectors


class FailingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, texts):
        self.calls += 1
        raise RuntimeError("embedding unavailable")


MEMORY_NOTE = """# Agent 记忆架构

> 关于长期记忆的设计。

## 为什么需要外部记忆

长任务的 Agent 不应把全部状态放进 prompt，稳定状态应写入外部持久化存储，这是记忆架构的核心。

## 记忆的分层

短期记忆放上下文，长期记忆放数据库，工作记忆放文件；记忆分层是常见的架构方案。
"""

TRADING_NOTE = """# 止损的数学真相

## 没有止损就无法计算

止损决定了单笔风险，量化交易的仓位公式必须以止损为分母，这是风险管理的基础。

## 常见误区

很多人把止损当成认输，其实止损只是量化风险的工具。
"""

COFFEE_NOTE = """# 咖啡与心血管

## 每天几杯

研究显示每天两到三杯咖啡对心血管的获益最大，超过之后收益递减。
"""

SERENA_NOTE = """# 节省 token 的编码技能

## Serena 工具

Serena 提供语义检索能力，让编码 Agent 少读无关文件，从而节省 token；它的记忆功能也值得一看。
"""


class NoteIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.db_path = str(root / "knowledge.db")
        self.store = KnowledgeStore(self.db_path, asset_root=str(root / "assets"))
        self.index = NoteIndex(self.db_path, embed_fn=toy_embed, model="toy", dimensions=len(VOCAB), batch_size=3)
        self.ids = {}
        for code, title, markdown in (
            ("mem01", "Agent 记忆架构讲解", MEMORY_NOTE),
            ("trd01", "止损的数学真相", TRADING_NOTE),
            ("cof01", "咖啡怎么喝最健康", COFFEE_NOTE),
            ("ser01", "AI写代码总白耗token", SERENA_NOTE),
        ):
            self.ids[code] = self.store.save(
                KnowledgeEntry(
                    video_id=f"vid-{code}",
                    title=title,
                    author="offline",
                    source_url=f"https://example.test/{code}",
                    summary_markdown=markdown,
                    tags="测试" if code != "ser01" else "AI编程,Serena工具",
                    video_code=code,
                )
            )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _index_all(self) -> None:
        self.index.index_all(only_missing=False)
        asyncio.run(self.index.embed_pending())

    def test_index_all_chunks_every_note_and_embeds_in_batches(self) -> None:
        report = self.index.index_all()
        self.assertEqual(report.notes_indexed, 4)
        self.assertGreater(report.chunks_written, 4)
        self.assertEqual(report.embeddings_pending, report.chunks_written)
        embedded = asyncio.run(self.index.embed_pending())
        self.assertEqual(embedded, report.chunks_written)
        stats = self.index.stats()
        self.assertEqual(stats["pending_embeddings"], 0)
        self.assertEqual(stats["vectors"], report.chunks_written)

    def test_reindex_reuses_vectors_for_unchanged_sections(self) -> None:
        self._index_all()
        before = self.index.stats()["vectors"]
        report = self.index.index_note(self.ids["mem01"])
        self.assertEqual(report.embeddings_pending, 0)
        self.assertEqual(self.index.stats()["vectors"], before)

    def test_deleting_a_note_removes_its_chunks(self) -> None:
        self._index_all()
        self.assertTrue(self.store.delete(self.ids["cof01"]))
        conn = sqlite3.connect(self.db_path)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM note_chunks WHERE knowledge_id = ?", (self.ids["cof01"],)
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 0)
        result = asyncio.run(self.index.search("咖啡"))
        self.assertNotIn("cof01", [note.video_code for note in result.notes])

    def test_semantic_channel_ranks_related_note_first(self) -> None:
        self._index_all()
        result = asyncio.run(self.index.search("智能体的记忆分层怎么设计", limit=3))
        self.assertTrue(result.semantic_available)
        self.assertEqual(result.notes[0].video_code, "mem01")
        self.assertIn("记忆", result.notes[0].best_heading)
        self.assertIn("语义", result.notes[0].channels)

    def test_keyword_channel_keeps_product_names_exact(self) -> None:
        self._index_all()
        result = asyncio.run(self.index.search("serena", limit=3))
        self.assertEqual(result.notes[0].video_code, "ser01")
        self.assertIn("关键词", result.notes[0].channels)

    def test_require_all_terms_filters_at_note_level(self) -> None:
        self._index_all()
        loose = asyncio.run(self.index.search("止损 记忆", limit=5))
        self.assertGreaterEqual(len(loose.notes), 2)
        strict = asyncio.run(self.index.search("止损 记忆", limit=5, require_all_terms=True))
        self.assertEqual(strict.notes, [])
        strict_hit = asyncio.run(self.index.search("Serena token", limit=5, require_all_terms=True))
        self.assertEqual([note.video_code for note in strict_hit.notes], ["ser01"])

    def test_each_note_appears_once_with_match_count(self) -> None:
        self._index_all()
        result = asyncio.run(self.index.search("记忆", limit=10))
        codes = [note.video_code for note in result.notes]
        self.assertEqual(len(codes), len(set(codes)))
        memory = next(note for note in result.notes if note.video_code == "mem01")
        self.assertGreaterEqual(memory.matched_chunks, 2)

    def test_search_degrades_to_keywords_when_embedder_fails(self) -> None:
        self._index_all()
        failing = FailingEmbedder()
        degraded = NoteIndex(self.db_path, embed_fn=failing, model="toy", dimensions=len(VOCAB))
        result = asyncio.run(degraded.search("止损", limit=3))
        self.assertFalse(result.semantic_available)
        self.assertEqual(result.notes[0].video_code, "trd01")
        self.assertEqual(failing.calls, 1)

    def test_search_without_embedder_is_keyword_only(self) -> None:
        self.index.index_all()
        plain = NoteIndex(self.db_path, embed_fn=None, model="toy", dimensions=len(VOCAB))
        result = asyncio.run(plain.search("咖啡", limit=3))
        self.assertFalse(result.semantic_available)
        self.assertEqual(result.notes[0].video_code, "cof01")

    def test_new_note_is_visible_after_cache_generation_changes(self) -> None:
        self._index_all()
        asyncio.run(self.index.search("记忆"))  # warm cache
        new_id = self.store.save(
            KnowledgeEntry(
                video_id="vid-new",
                title="量化交易入门",
                author="offline",
                source_url="https://example.test/new",
                summary_markdown="# 量化交易入门\n\n## 因子\n\n量化研究从因子开始，量化回测决定策略是否可用。",
                tags="量化",
                video_code="qnt01",
            )
        )
        asyncio.run(self.index.index_note_and_embed(new_id))
        result = asyncio.run(self.index.search("量化 因子", limit=3))
        self.assertEqual(result.notes[0].video_code, "qnt01")

    def test_collect_respects_budget_per_note_cap_and_dedupes_near_duplicates(self) -> None:
        self._index_all()
        # A near-duplicate section of the memory note in another note.
        dup_id = self.store.save(
            KnowledgeEntry(
                video_id="vid-dup",
                title="记忆架构转述",
                author="offline",
                source_url="https://example.test/dup",
                summary_markdown="# 记忆架构转述\n\n## 分层\n\n" + "记忆记忆记忆架构架构架构" * 3,
                tags="",
                video_code="dup01",
            )
        )
        asyncio.run(self.index.index_note_and_embed(dup_id))
        result = asyncio.run(self.index.collect("记忆 架构", max_chars=400, max_per_note=1))
        self.assertLessEqual(result.total_chars, 400)
        per_note = {}
        for section in result.sections:
            per_note[section.chunk.knowledge_id] = per_note.get(section.chunk.knowledge_id, 0) + 1
        self.assertTrue(all(count <= 1 for count in per_note.values()))
        self.assertEqual(result.note_count, len(per_note))

        wide = asyncio.run(self.index.collect("记忆 架构", max_chars=5000, max_per_note=3))
        codes = {section.chunk.video_code for section in wide.sections}
        # mem01's two memory sections both survive; the duplicate note is a
        # near-copy of them in vector space and is suppressed.
        self.assertIn("mem01", codes)
        self.assertNotIn("dup01", codes)

    def test_weak_semantic_neighbours_are_not_padded_in(self) -> None:
        self._index_all()
        # "serena" is only mentioned in one note; the other notes' vectors are
        # nearly orthogonal to the query and must not fill the remaining slots.
        result = asyncio.run(self.index.search("serena", limit=10))
        self.assertEqual([note.video_code for note in result.notes], ["ser01"])

    def test_empty_query_returns_nothing(self) -> None:
        self._index_all()
        self.assertEqual(asyncio.run(self.index.search("   ")).notes, [])
        self.assertEqual(asyncio.run(self.index.collect("")).sections, [])


class SplitTermsTests(unittest.TestCase):
    def test_terms_are_lowercased_deduplicated_and_split_on_cjk_punctuation(self) -> None:
        self.assertEqual(split_terms(" Agent，记忆、agent ;止损 "), ["agent", "记忆", "止损"])


if __name__ == "__main__":
    unittest.main()
