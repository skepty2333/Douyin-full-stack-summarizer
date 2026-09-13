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
        self.index = NoteIndex(db_path, embed_fn=toy_embed, model="toy", dimensions=len(VOCAB))
        self.original = (mcp_server.store, mcp_server.index)
        mcp_server.store, mcp_server.index = self.store, self.index
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
        for code, title, markdown in notes:
            self.store.save(
                KnowledgeEntry(
                    video_id=f"vid-{code}",
                    title=title,
                    author="offline",
                    source_url=f"https://example.test/{code}",
                    summary_markdown=markdown,
                    tags="测试",
                    video_code=code,
                )
            )

    def tearDown(self) -> None:
        mcp_server.store, mcp_server.index = self.original
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

    def test_stats_reports_index_state(self) -> None:
        self._build_index()
        text = asyncio.run(mcp_server.knowledge_stats())
        self.assertIn("**总笔记数**: 3", text)
        self.assertIn("章节索引", text)
        self.assertIn("待向量化 0", text)


if __name__ == "__main__":
    unittest.main()
