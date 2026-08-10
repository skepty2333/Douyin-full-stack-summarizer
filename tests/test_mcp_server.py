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


if __name__ == "__main__":
    unittest.main()
