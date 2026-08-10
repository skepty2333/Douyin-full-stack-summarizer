"""Temporary-database tests for the local knowledge store."""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch


# This module may be run on its own, so independently prevent app.config from
# consulting the project .env while importing the database implementation.
with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-key"}, clear=False):
    with patch("dotenv.load_dotenv", return_value=False):
        from app.database.knowledge_store import KnowledgeAsset, KnowledgeEntry, KnowledgeStore


class KnowledgeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = KnowledgeStore(os.path.join(self.temp_dir.name, "knowledge.sqlite3"))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def entry(
        *,
        video_code: str,
        title: str = "Architecture Notes",
        summary: str = "Retries and idempotency improve service reliability.",
        tags: str = "architecture,stability",
    ) -> KnowledgeEntry:
        return KnowledgeEntry(
            video_id=f"video-{video_code}",
            title=title,
            author="offline-author",
            source_url=f"https://example.test/{video_code}",
            summary_markdown=summary,
            tags=tags,
            user_requirement="offline test",
            duration_seconds=12.5,
            video_code=video_code,
        )

    def asset(self, *, key: str = "V0001", caption: str = "任务看板显示四列流程") -> tuple[KnowledgeAsset, bytes]:
        payload = b"\xff\xd8offline-reviewed-jpeg\xff\xd9"
        digest = hashlib.sha256(payload).hexdigest()
        relative_path = f"blobs/{digest[:2]}/{digest}.jpg"
        destination = Path(self.store.asset_root) / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        os.chmod(destination, 0o600)
        return (
            KnowledgeAsset(
                asset_key=key,
                relative_path=relative_path,
                mime_type="image/jpeg",
                timestamp_ms=65_000,
                caption=caption,
                kind="ui_structure",
                confidence="high",
                width=320,
                height=180,
                byte_size=len(payload),
                sha256=digest,
                display_order=0,
                quality_score=8.5,
            ),
            payload,
        )

    def test_insert_and_read_back(self) -> None:
        entry_id = self.store.save(self.entry(video_code="A0001"))

        saved = self.store.get_by_id(entry_id)
        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertEqual(saved["video_code"], "A0001")
        self.assertEqual(saved["title"], "Architecture Notes")
        self.assertEqual(saved["duration_seconds"], 12.5)
        self.assertTrue(saved["created_at"])
        self.assertTrue(saved["timestamp"])
        self.assertTrue(self.store.video_code_exists("A0001"))

    def test_explicit_overwrite_updates_same_row(self) -> None:
        original_id = self.store.save(self.entry(video_code="B0001", title="Old title"))

        replacement_id = self.store.save(
            self.entry(
                video_code="B0001",
                title="New title",
                summary="Replacement content",
                tags="replacement",
            ),
            allow_overwrite=True,
        )

        self.assertEqual(replacement_id, original_id)
        self.assertEqual(self.store.stats()["total_entries"], 1)
        saved = self.store.get_by_video_code("B0001")
        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertEqual(saved["title"], "New title")
        self.assertEqual(saved["summary_markdown"], "Replacement content")
        self.assertEqual(saved["tags"], "replacement")

    def test_default_collision_raises_without_overwriting(self) -> None:
        original_id = self.store.save(self.entry(video_code="C0001", title="Keep me"))

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save(self.entry(video_code="C0001", title="Do not save"))

        self.assertEqual(self.store.stats()["total_entries"], 1)
        saved = self.store.get_by_video_code("C0001")
        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertEqual(saved["id"], original_id)
        self.assertEqual(saved["title"], "Keep me")

    def test_loose_and_precise_search_and_health_check(self) -> None:
        architecture_id = self.store.save(self.entry(video_code="D0001"))
        self.store.save(
            self.entry(
                video_code="D0002",
                title="Cooking Notes",
                summary="A simple noodle recipe.",
                tags="food,cooking",
            )
        )

        tag_results = self.store.search("stability")
        body_results = self.store.search("idempotency")
        precise_results = self.store.search_precise("stability idempotency")
        no_precise_match = self.store.search_precise("stability recipe")

        self.assertEqual([row["id"] for row in tag_results], [architecture_id])
        self.assertEqual([row["id"] for row in body_results], [architecture_id])
        self.assertEqual([row["id"] for row in precise_results], [architecture_id])
        self.assertEqual(no_precise_match, [])
        self.assertTrue(self.store.health_check())

    def test_multimodal_markdown_and_asset_round_trip(self) -> None:
        asset, payload = self.asset(caption="任务看板显示四列工作流")
        entry = self.entry(
            video_code="E0001",
            summary=(
                "正文\n\n![视频画面 01:05：任务看板显示四列工作流]"
                "(knowledge-asset://E0001/V0001)"
            ),
        )

        entry_id = self.store.save(entry, assets=[asset])

        listed = self.store.list_assets(entry_id)
        self.assertEqual([item["asset_key"] for item in listed], ["V0001"])
        self.assertEqual(listed[0]["caption"], "任务看板显示四列工作流")
        self.assertNotIn("relative_path", listed[0])
        self.assertNotIn("sha256", listed[0])
        self.assertEqual(self.store.read_asset_by_video_code("E0001", "V0001"), payload)
        self.assertEqual([row["id"] for row in self.store.search("四列工作流")], [entry_id])

    def test_explicit_asset_replacement_defers_blob_removal_to_safe_gc(self) -> None:
        asset, _payload = self.asset()
        blob_path = Path(self.store.asset_root) / asset.relative_path
        entry_id = self.store.save(self.entry(video_code="F0001"), assets=[asset])

        replacement_id = self.store.save(
            self.entry(video_code="F0001", summary="纯文字替代内容"),
            allow_overwrite=True,
            assets=[],
        )

        self.assertEqual(replacement_id, entry_id)
        self.assertEqual(self.store.list_assets(entry_id), [])
        self.assertTrue(blob_path.exists())
        os.utime(blob_path, (0, 0))
        self.assertEqual(
            self.store.prune_orphan_assets(min_age_seconds=0),
            1,
        )
        self.assertFalse(blob_path.exists())

    def test_overwrite_without_asset_argument_preserves_existing_manifest(self) -> None:
        asset, payload = self.asset()
        entry_id = self.store.save(self.entry(video_code="G0001"), assets=[asset])

        self.store.save(
            self.entry(video_code="G0001", title="只更新正文"),
            allow_overwrite=True,
        )

        self.assertEqual(len(self.store.list_assets(entry_id)), 1)
        self.assertEqual(self.store.read_asset_by_video_code("G0001", "V0001"), payload)
        os.utime(Path(self.store.asset_root) / asset.relative_path, (0, 0))
        self.assertEqual(self.store.prune_orphan_assets(min_age_seconds=0), 0)

    def test_forged_database_path_is_never_read(self) -> None:
        asset, _payload = self.asset()
        entry_id = self.store.save(self.entry(video_code="H0001"), assets=[asset])
        conn = self.store._get_conn()
        try:
            conn.execute(
                "UPDATE knowledge_assets SET relative_path = ? WHERE knowledge_id = ?",
                ("../../etc/passwd", entry_id),
            )
            conn.commit()
        finally:
            conn.close()

        with self.assertRaises(ValueError):
            self.store.read_asset_by_video_code("H0001", "V0001")


if __name__ == "__main__":
    unittest.main()
