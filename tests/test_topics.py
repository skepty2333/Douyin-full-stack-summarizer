"""Offline tests for topic birth, compilation, splitting and curation."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-key"}, clear=False):
    with patch("dotenv.load_dotenv", return_value=False):
        from app.database.knowledge_store import KnowledgeEntry, KnowledgeStore
        from app.database.note_index import NoteIndex
        from app.database.topics import TopicStore
        from app.database.vocabulary import VocabularyStore
        from app.services import topic_compiler
        from app.services.topic_compiler import TopicCompiler, count_uncited_lines


VOCAB = ["记忆", "止损", "咖啡", "架构", "量化", "学习", "健身", "面试"]


async def toy_embed(texts):
    return [[0.01 + text.lower().count(word) for word in VOCAB] for text in texts]


class FakeChat:
    """Returns a canned page or split proposal and records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, *, system, user, max_tokens, operation, thinking_budget):
        self.calls.append({"system": system, "user": user, "operation": operation})
        if operation == "topic_split":
            codes = [line.split("\t")[0].strip("- ") for line in user.splitlines() if line.startswith("- ") and "\t" in line]
            half = max(4, len(codes) // 2)
            return json.dumps(
                {
                    "children": [
                        {"name": "记忆分层", "existing": False, "codes": codes[:half]},
                        {"name": "记忆持久化", "existing": False, "codes": codes[half - 2 :] if len(codes) - half + 2 >= 4 else codes[:half]},
                    ]
                },
                ensure_ascii=False,
            )
        first_code = next((c for c in ("mem01", "mem02") if f"[{c}]" in user), "mem01")
        previous = "【上一版页面" in user
        return (
            "# Agent记忆\n\n> 关于 Agent 记忆的设计。\n\n"
            "## 核心结论与主流做法\n\n"
            f"- 稳定状态应写入外部存储 [{first_code}]\n"
            "- 没有来源的一句话\n"
            + ("- 上一版保留的结论 [mem02]\n" if previous else "")
            + "\n## 不同来源的分歧\n\n暂无\n\n## 可能已过时\n\n暂无\n\n## 待验证\n\n暂无\n\n"
            "## 来源笔记\n\n- 模型自己写的来源，应被替换\n"
        )


class TopicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.db_path = str(root / "knowledge.db")
        self.store = KnowledgeStore(self.db_path, asset_root=str(root / "assets"))
        self.vocab = VocabularyStore(self.db_path)
        self.index = NoteIndex(self.db_path, embed_fn=toy_embed, model="toy", dimensions=len(VOCAB))
        self.topics = TopicStore(self.db_path, vocabulary=self.vocab)
        self.chat = FakeChat()
        self.compiler = TopicCompiler(store=self.store, index=self.index, topics=self.topics, chat_fn=self.chat, model="fake")
        self.memory = self.vocab.upsert_entry("Agent记忆", kind="topic", aliases=["记忆系统"], source="seed")
        self.tutorial = self.vocab.upsert_entry("技术教程", kind="content_type", source="seed")
        self.ids: dict[str, int] = {}

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _add_note(self, code: str, author: str, *, published: str = "2026-03-01T00:00:00+00:00", temporality: str = "stable", entries=None) -> int:
        note_id = self.store.save(
            KnowledgeEntry(
                video_id=f"vid-{code}",
                title=f"笔记 {code}",
                author=author,
                source_url=f"https://example.test/{code}",
                summary_markdown=f"# 笔记 {code}\n\n## 记忆分层\n\n长期记忆放数据库，短期记忆放上下文，记忆架构 {code}。\n\n## 其他\n\n无关内容。",
                tags="Agent记忆",
                video_code=code,
                published_at=published,
                domain="ai",
                temporality=temporality,
            )
        )
        self.vocab.set_note_tags(note_id, [e.id for e in (entries or [self.memory])])
        self.ids[code] = note_id
        return note_id

    def _index(self) -> None:
        self.index.index_all()
        asyncio.run(self.index.embed_pending())

    def test_birth_requires_density_and_eligible_kind(self) -> None:
        for i in range(5):
            self._add_note(f"mem0{i}", f"作者{i}")
        self.assertEqual(self.topics.eligible_entries(), [])  # weight 5 < 6
        self._add_note("mem05", "作者0")  # weight 6, 5 authors
        self.assertEqual([e.canonical for e in self.topics.eligible_entries()], ["Agent记忆"])
        # A content type never becomes a topic even when dense.
        for i in range(7):
            self._add_note(f"tut0{i}", f"作者{i}", entries=[self.tutorial])
        self.assertEqual([e.canonical for e in self.topics.eligible_entries()], ["Agent记忆"])
        topic = self.topics.born(self.topics.eligible_entries()[0])
        self.assertEqual((topic.status, topic.note_count, topic.latest_version), ("candidate", 6, 0))
        self.assertEqual(self.topics.eligible_entries(), [])
        self.assertEqual(self.topics.resolve("记忆系统").id, topic.id)

    def test_secondary_tag_positions_count_half(self) -> None:
        filler = self.vocab.upsert_entry("填充", kind="topic", source="seed")
        other = self.vocab.upsert_entry("填充二", kind="topic", source="seed")
        # 4 primary (position 0) + 4 secondary (position 2) = weight 6.
        for i in range(4):
            self._add_note(f"pri0{i}", f"作者{i}")
        for i in range(4):
            self._add_note(f"sec0{i}", f"作者{i + 4}", entries=[filler, other, self.memory])
        entry = self.vocab.resolve("Agent记忆")
        self.assertEqual((entry.note_count, entry.weight), (8, 6.0))
        self.assertEqual([e.canonical for e in self.topics.eligible_entries()], ["Agent记忆"])

    def test_compile_cites_sources_and_appends_authoritative_source_list(self) -> None:
        for i in range(6):
            self._add_note(f"mem0{i}", f"作者{i}")
        self._index()
        topic = self.topics.born(self.topics.eligible_entries()[0])
        result = asyncio.run(self.compiler.compile(topic))
        self.assertEqual(result.version.version, 1)
        self.assertEqual(result.topic.status, "active")
        self.assertEqual(result.member_notes, 6)
        self.assertEqual(result.uncited_lines, 1)  # "没有来源的一句话"
        page = result.version.markdown
        self.assertIn("## 来源笔记", page)
        self.assertNotIn("模型自己写的来源", page)
        for i in range(6):
            self.assertIn(f"`mem0{i}`", page)
        self.assertIn("发布 2026-03-01", page)
        self.assertEqual(sorted(result.version.source_note_ids), sorted(self.ids.values()))
        prompt = self.chat.calls[0]["user"]
        self.assertIn("[mem00]", prompt)
        self.assertIn("稳定", prompt)  # temporality label in the material header
        self.assertNotIn("无关内容", prompt)  # only relevant sections reach the compiler

    def test_recompile_feeds_previous_version_and_bumps_version(self) -> None:
        for i in range(6):
            self._add_note(f"mem0{i}", f"作者{i}")
        self._index()
        topic = self.topics.born(self.topics.eligible_entries()[0])
        asyncio.run(self.compiler.compile(topic))
        new_note = self._add_note("mem06", "作者6")
        self.assertEqual(self.topics.note_tagged(new_note), [])  # dirty 1 < threshold 3
        self.assertEqual(self.topics.get(topic.id).dirty, 1)
        result = asyncio.run(self.compiler.compile(self.topics.get(topic.id)))
        self.assertEqual(result.version.version, 2)
        self.assertIn("【上一版页面", self.chat.calls[-1]["user"])
        self.assertIn("上一版保留的结论", result.version.markdown)
        self.assertEqual(self.topics.get(topic.id).dirty, 0)
        self.assertIsNotNone(self.topics.version(topic.id, 1))

    def test_note_tagged_reports_topics_due_after_threshold(self) -> None:
        for i in range(6):
            self._add_note(f"mem0{i}", f"作者{i}")
        topic = self.topics.born(self.topics.eligible_entries()[0])
        due = []
        for i in range(6, 9):
            due = self.topics.note_tagged(self._add_note(f"mem0{i}", f"作者{i}"))
        self.assertEqual([t.id for t in due], [topic.id])

    def test_split_makes_hub_with_children_and_hub_page_lists_them(self) -> None:
        with patch.object(topic_compiler, "TOPIC_RECOMPILE_DIRTY", 3):
            for i in range(12):
                self._add_note(f"mem{i:02d}", f"作者{i}")
            self._index()
            topic = self.topics.born(self.topics.eligible_entries()[0])
            with patch("app.database.topics.TOPIC_MAX_NOTES", 10):
                self.assertTrue(self.topics.needs_split(self.topics.get(topic.id)))
                split = asyncio.run(self.compiler.split(self.topics.get(topic.id)))
            self.assertTrue(split.parent.is_hub)
            self.assertEqual({c.name for c in split.children}, {"记忆分层", "记忆持久化"})
            self.assertTrue(all(c.parent_id == topic.id for c in split.children))
            self.assertEqual(set(split.created_entries), {"记忆分层", "记忆持久化"})
            # Notes keep their parent tag and gain the child tag.
            first = split.children[0]
            member = self.topics.member_note_ids(first)[0]
            names = {e.canonical for e in self.vocab.note_entries(member)}
            self.assertIn("Agent记忆", names)
            self.assertIn(first.name, names)
            # The hub page indexes the children instead of synthesising.
            for child in split.children:
                asyncio.run(self.compiler.compile(child))
            hub = asyncio.run(self.compiler.compile(self.topics.get(topic.id)))
            self.assertEqual(hub.version.compiled_by, "hub")
            self.assertIn("## 子主题", hub.version.markdown)
            self.assertIn("**记忆分层**", hub.version.markdown)
            self.assertIn("关于 Agent 记忆的设计。", hub.version.markdown)  # child definition quoted

    def test_grow_births_compiles_and_reports(self) -> None:
        for i in range(6):
            self._add_note(f"mem0{i}", f"作者{i}")
        self._index()
        report = asyncio.run(self.compiler.grow())
        self.assertEqual(report["born"], ["Agent记忆"])
        self.assertEqual(report["compiled"], [("Agent记忆", 1)])
        self.assertEqual(report["failed"], [])
        again = asyncio.run(self.compiler.grow())
        self.assertEqual((again["born"], again["compiled"]), ([], []))

    def test_veto_and_merge_suggestions(self) -> None:
        for i in range(6):
            self._add_note(f"mem0{i}", f"作者{i}")
        other = self.vocab.upsert_entry("记忆机制", kind="topic", source="seed")
        for code in list(self.ids)[:5]:
            note_id = self.ids[code]
            self.vocab.set_note_tags(note_id, [self.memory.id, other.id])
        self._add_note("mem06", "作者6", entries=[other])
        a, b = (self.topics.born(e) for e in self.topics.eligible_entries())
        for t in (a, b):
            self.topics.set_status(t.id, "active")
        suggestions = self.topics.merge_suggestions(min_overlap=0.5)
        self.assertEqual(len(suggestions), 1)
        self.assertAlmostEqual(suggestions[0].overlap, 5 / 7)
        self.topics.set_status(b.id, "archived")
        self.assertEqual([t.name for t in self.topics.list_topics()], ["Agent记忆"])
        self.assertEqual(self.topics.merge_suggestions(), [])

    def test_count_uncited_lines(self) -> None:
        page = (
            "# T\n\n## 核心结论与主流做法\n\n**一、分组标签不算结论**\n\n- **小标题：**\n"
            "- 有来源 [ab123]\n- 没来源\n- 假来源 [zz999]\n\n## 分歧\n\n- 这里不算\n"
        )
        self.assertEqual(count_uncited_lines(page, {"ab123"}), 2)


if __name__ == "__main__":
    unittest.main()
