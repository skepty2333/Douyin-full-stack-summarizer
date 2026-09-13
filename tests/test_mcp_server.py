"""Offline tests for text-first MCP note and on-demand image retrieval."""
from __future__ import annotations

import asyncio
import base64
import hashlib
from pathlib import Path
import tempfile
import unittest

import mcp_server
from app.database.knowledge_store import KnowledgeAsset, KnowledgeEntry, KnowledgeStore
from app.database.note_index import NoteIndex
from app.database.topics import TopicStore
from app.database.vocabulary import VocabularyStore
from app.services.topic_compiler import TopicCompiler


VOCAB = ["记忆", "止损", "咖啡"]


async def toy_embed(texts):
    return [[0.01 + text.lower().count(word) for word in VOCAB] for text in texts]


class MCPImageToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = KnowledgeStore(
            str(root / "knowledge.db"),
            asset_root=str(root / "assets"),
        )
        self.original_store = mcp_server.store
        mcp_server.store = self.store

        self.payload = b"\xff\xd8mcp-reviewed-jpeg\xff\xd9"
        digest = hashlib.sha256(self.payload).hexdigest()
        relative_path = f"blobs/{digest[:2]}/{digest}.jpg"
        destination = self.store.asset_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.payload)
        asset = KnowledgeAsset(
            asset_key="V0001",
            relative_path=relative_path,
            mime_type="image/jpeg",
            timestamp_ms=12_000,
            caption="界面显示任务进入进行中状态",
            kind="state_change",
            confidence="high",
            width=320,
            height=180,
            byte_size=len(self.payload),
            sha256=digest,
            display_order=0,
            quality_score=9.0,
        )
        self.note_id = self.store.save(
            KnowledgeEntry(
                video_id="video-mcp01",
                title="MCP 多模态测试",
                author="offline",
                source_url="https://example.test/mcp01",
                summary_markdown=(
                    "正文\n\n![视频画面 00:12：界面显示任务进入进行中状态]"
                    "(knowledge-asset://mcp01/V0001)"
                ),
                video_code="mcp01",
            ),
            assets=[asset],
        )

    def tearDown(self) -> None:
        mcp_server.store = self.original_store
        self.temp_dir.cleanup()

    def test_list_tool_exposes_logical_reference_not_server_path(self) -> None:
        result = asyncio.run(
            mcp_server.list_note_images(
                mcp_server.ListNoteImagesInput(note_id=self.note_id)
            )
        )

        self.assertIn("knowledge-asset://mcp01/V0001", result)
        self.assertIn("界面显示任务进入进行中状态", result)
        self.assertNotIn(str(self.store.asset_root), result)
        self.assertNotIn("blobs/", result)

    def test_get_image_tool_returns_verified_jpeg_content(self) -> None:
        image = asyncio.run(
            mcp_server.get_note_image(
                mcp_server.GetNoteImageInput(video_code="mcp01", asset_id="V0001")
            )
        )
        content = image.to_image_content()

        self.assertEqual(content.mimeType, "image/jpeg")
        self.assertEqual(base64.b64decode(content.data), self.payload)


class MCPSearchToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        db_path = str(root / "knowledge.db")
        self.store = KnowledgeStore(db_path, asset_root=str(root / "assets"))
        self.vocab = VocabularyStore(db_path)
        self.index = NoteIndex(
            db_path,
            embed_fn=toy_embed,
            alias_groups_fn=self.vocab.alias_groups,
            model="toy",
            dimensions=len(VOCAB),
        )
        self.topics = TopicStore(db_path, vocabulary=self.vocab)

        async def fake_chat(*, system, user, max_tokens, operation, thinking_budget):
            return "# 页面\n\n> 定义。\n\n## 核心结论与主流做法\n\n- 结论 [mem01]\n\n## 不同来源的分歧\n\n暂无\n\n## 可能已过时\n\n暂无\n\n## 待验证\n\n暂无\n"

        self.compiler = TopicCompiler(store=self.store, index=self.index, topics=self.topics, chat_fn=fake_chat, model="fake")
        self.original = (
            mcp_server.store, mcp_server.index, mcp_server.vocabulary, mcp_server.topics, mcp_server.topic_compiler,
        )
        mcp_server.store, mcp_server.index, mcp_server.vocabulary = self.store, self.index, self.vocab
        mcp_server.topics, mcp_server.topic_compiler = self.topics, self.compiler
        notes = (
            (
                "mem01",
                "Agent 记忆架构",
                "# Agent 记忆架构\n\n## 记忆分层\n\n长期记忆放数据库，短期记忆放上下文，记忆分层很重要。"
                + "这一段刻意写得很长，用来验证搜索结果只返回片段而不是整段正文。" * 4
                + "结尾句：全文结束标记。",
            ),
            ("trd01", "止损的数学真相", "# 止损\n\n## 仓位公式\n\n止损决定单笔风险，仓位公式以止损为分母。"),
            ("cof01", "咖啡与心血管", "# 咖啡\n\n## 每天几杯\n\n两到三杯咖啡获益最大。"),
        )
        self.note_ids = {}
        metadata = {
            "mem01": ("ai", "version_sensitive", "2026-03-01T00:00:00+00:00"),
            "trd01": ("trading", "stable", "2025-12-02T13:21:57+00:00"),
            "cof01": ("life", "time_bound", ""),
        }
        for code, title, markdown in notes:
            domain, temporality, published = metadata[code]
            self.note_ids[code] = self.store.save(
                KnowledgeEntry(
                    video_id=f"vid-{code}",
                    title=title,
                    author="offline",
                    source_url=f"https://example.test/{code}",
                    summary_markdown=markdown,
                    tags="测试",
                    video_code=code,
                    published_at=published,
                    domain=domain,
                    temporality=temporality,
                )
            )
        agent = self.vocab.upsert_entry("Agent", kind="topic", aliases=["智能体"], source="seed")
        self.vocab.set_note_tags(self.note_ids["mem01"], [agent.id])

    def tearDown(self) -> None:
        (
            mcp_server.store, mcp_server.index, mcp_server.vocabulary, mcp_server.topics, mcp_server.topic_compiler,
        ) = self.original
        self.temp_dir.cleanup()

    def _build_index(self) -> None:
        self.index.index_all()
        asyncio.run(self.index.embed_pending())

    def test_search_notes_returns_compact_ranked_list(self) -> None:
        self._build_index()
        text = asyncio.run(mcp_server.search_notes(mcp_server.SearchInput(query="记忆 怎么分层", limit=2)))
        self.assertIn("语义 + 关键词", text)
        first = text.split("\n")[2]
        self.assertTrue(first.startswith("1. `mem01`"), first)
        self.assertIn("▸ 记忆分层", text)
        self.assertIn("collect_sections", text)
        self.assertNotIn("结尾句：全文结束标记。", text)  # snippet only, never the full section

    def test_precise_search_requires_every_term(self) -> None:
        self._build_index()
        miss = asyncio.run(
            mcp_server.search_notes_precise(mcp_server.PreciseSearchInput(query="止损 咖啡"))
        )
        self.assertIn("未找到同时包含", miss)
        hit = asyncio.run(
            mcp_server.search_notes_precise(mcp_server.PreciseSearchInput(query="止损 仓位"))
        )
        self.assertIn("`trd01`", hit)
        self.assertNotIn("`cof01`", hit)

    def test_collect_sections_groups_bodies_by_note(self) -> None:
        self._build_index()
        text = asyncio.run(
            mcp_server.collect_sections(mcp_server.CollectSectionsInput(query="止损 仓位", max_chars=2000))
        )
        self.assertIn("### `trd01` 止损的数学真相", text)
        self.assertIn("#### 仓位公式", text)
        self.assertIn("止损决定单笔风险，仓位公式以止损为分母。", text)
        self.assertIn("get_note_by_code", text)

    def test_search_falls_back_to_legacy_matching_when_index_is_empty(self) -> None:
        text = asyncio.run(mcp_server.search_notes(mcp_server.SearchInput(query="咖啡")))
        self.assertIn("章节索引未建立", text)
        self.assertIn("`cof01`", text)
        collect = asyncio.run(
            mcp_server.collect_sections(mcp_server.CollectSectionsInput(query="咖啡"))
        )
        self.assertIn("章节索引未建立", collect)

    def test_results_show_publish_date_and_temporality(self) -> None:
        self._build_index()
        text = asyncio.run(mcp_server.search_notes(mcp_server.SearchInput(query="止损 仓位", limit=3)))
        self.assertIn("`trd01`", text)
        self.assertIn("发布 2025-12-02", text)
        mem = asyncio.run(mcp_server.search_notes(mcp_server.SearchInput(query="记忆", limit=3)))
        self.assertIn("版本敏感", mem)
        coffee = asyncio.run(mcp_server.search_notes(mcp_server.SearchInput(query="咖啡", limit=3)))
        self.assertIn("入库 ", coffee)  # no publish date known
        self.assertIn("时效性", coffee)

    def test_domain_filter_narrows_search(self) -> None:
        self._build_index()
        text = asyncio.run(
            mcp_server.search_notes(mcp_server.SearchInput(query="止损 记忆", limit=5, domain="trading"))
        )
        self.assertIn("仅 交易 领域", text)
        self.assertIn("`trd01`", text)
        self.assertNotIn("`mem01`", text)

    def test_list_by_tag_resolves_aliases_through_vocabulary(self) -> None:
        text = asyncio.run(mcp_server.list_by_tag(mcp_server.TagFilterInput(tag="智能体")))
        self.assertIn("Agent（topic，别名 智能体，共 1 条）", text)
        self.assertIn("`mem01`", text)
        self.assertIn("发布 2026-03-01", text)
        fallback = asyncio.run(mcp_server.list_by_tag(mcp_server.TagFilterInput(tag="测试")))
        self.assertIn("不在词表中", fallback)

    def test_get_note_by_code_shows_metadata(self) -> None:
        text = asyncio.run(mcp_server.get_note_by_code("trd01"))
        self.assertIn("**发布时间**: 2025-12-02", text)
        self.assertIn("**领域 / 时效**: 交易", text)

    def test_topic_tools_compile_read_list_and_veto(self) -> None:
        self._build_index()
        empty = asyncio.run(mcp_server.list_topics())
        self.assertIn("还没有主题", empty)
        not_yet = asyncio.run(mcp_server.read_topic(mcp_server.ReadTopicInput(name="智能体")))
        self.assertIn("还不是主题", not_yet)
        compiled = asyncio.run(mcp_server.compile_topic(mcp_server.TopicNameInput(name="智能体")))
        self.assertIn("已编译 **Agent** v1", compiled)
        self.assertIn("## 来源笔记", compiled)
        self.assertIn("`mem01`", compiled)
        page = asyncio.run(mcp_server.read_topic(mcp_server.ReadTopicInput(name="Agent")))
        self.assertIn("<!-- 主题 Agent · v1", page)
        listing = asyncio.run(mcp_server.list_topics())
        self.assertIn("**Agent** — 1 条 / 1 位作者 · 有效 · v1", listing)
        search = asyncio.run(mcp_server.search_notes(mcp_server.SearchInput(query="记忆", limit=3)))
        self.assertNotIn("相关主题页", search)  # only one hit carries the tag (needs >= 2)
        vetoed = asyncio.run(mcp_server.veto_topic(mcp_server.TopicNameInput(name="Agent")))
        self.assertIn("已否决", vetoed)
        self.assertIn("还没有主题", asyncio.run(mcp_server.list_topics()))
        renamed = asyncio.run(mcp_server.rename_topic(mcp_server.RenameTopicInput(name="Agent", new_name="智能体系统")))
        self.assertIn("Agent → 智能体系统", renamed)
        self.assertIsNotNone(self.vocab.resolve("agent"))

    def test_stats_reports_index_state(self) -> None:
        self._build_index()
        text = asyncio.run(mcp_server.knowledge_stats())
        self.assertIn("**总笔记数**: 3", text)
        self.assertIn("章节索引", text)
        self.assertIn("待向量化 0", text)
        self.assertIn("**词表**: 1 个规范条目", text)
        self.assertIn("**主题**: 有效 0", text)


if __name__ == "__main__":
    unittest.main()
