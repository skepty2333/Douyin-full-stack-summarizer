"""Offline tests for the controlled vocabulary and note tag links."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-key"}, clear=False):
    with patch("dotenv.load_dotenv", return_value=False):
        from app.database.knowledge_store import KnowledgeEntry, KnowledgeStore
        from app.database.vocabulary import VocabularyStore, normalize_name


class VocabularyStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.db_path = str(root / "knowledge.db")
        self.store = KnowledgeStore(self.db_path, asset_root=str(root / "assets"))
        self.vocab = VocabularyStore(self.db_path)
        self.note_ids = []
        for code, author in (("n0001", "甲"), ("n0002", "乙"), ("n0003", "乙")):
            self.note_ids.append(
                self.store.save(
                    KnowledgeEntry(
                        video_id=f"vid-{code}",
                        title=f"笔记 {code}",
                        author=author,
                        source_url=f"https://example.test/{code}",
                        summary_markdown="# 标题\n\n## 正文\n\n内容。",
                        video_code=code,
                    )
                )
            )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_normalize_ignores_case_whitespace_and_hash(self) -> None:
        self.assertEqual(normalize_name(" AI Agent "), "aiagent")
        self.assertEqual(normalize_name("#智能体"), "智能体")

    def test_upsert_creates_entry_and_aliases_resolve_to_it(self) -> None:
        entry = self.vocab.upsert_entry("Agent", kind="topic", aliases=["智能体", "AI Agent"], source="seed")
        self.assertEqual(entry.kind, "topic")
        self.assertEqual(set(entry.aliases), {"智能体", "AI Agent"})
        for name in ("agent", "智能体", "ai agent", "AIAgent", "#智能体"):
            resolved = self.vocab.resolve(name)
            self.assertIsNotNone(resolved, name)
            self.assertEqual(resolved.id, entry.id, name)
        self.assertIsNone(self.vocab.resolve("不存在"))

    def test_upsert_existing_name_adds_aliases_but_never_steals_them(self) -> None:
        agent = self.vocab.upsert_entry("Agent", kind="topic", aliases=["智能体"])
        rag = self.vocab.upsert_entry("RAG", kind="topic", aliases=["检索增强生成"])
        again = self.vocab.upsert_entry("智能体", kind="entity", aliases=["AI Agent", "检索增强生成"])
        self.assertEqual(again.id, agent.id)  # resolved through the alias, no duplicate
        self.assertEqual(again.kind, "topic")  # the existing kind wins
        self.assertIn("AI Agent", again.aliases)
        self.assertEqual(self.vocab.resolve("检索增强生成").id, rag.id)
        self.assertEqual(len(self.vocab.list_entries()), 2)

    def test_note_links_and_counts(self) -> None:
        agent = self.vocab.upsert_entry("Agent", kind="topic")
        rag = self.vocab.upsert_entry("RAG", kind="topic")
        self.vocab.set_note_tags(self.note_ids[0], [agent.id, rag.id])
        self.vocab.set_note_tags(self.note_ids[1], [agent.id])
        self.vocab.set_note_tags(self.note_ids[2], [agent.id])
        entries = {entry.canonical: entry for entry in self.vocab.list_entries()}
        self.assertEqual((entries["Agent"].note_count, entries["Agent"].author_count), (3, 2))
        self.assertEqual((entries["RAG"].note_count, entries["RAG"].author_count), (1, 1))
        self.assertEqual([e.canonical for e in self.vocab.note_entries(self.note_ids[0])], ["Agent", "RAG"])
        self.assertEqual(self.vocab.untagged_note_ids(), [])
        # Replacing links drops the old ones.
        self.vocab.set_note_tags(self.note_ids[0], [rag.id])
        self.assertEqual([e.canonical for e in self.vocab.note_entries(self.note_ids[0])], ["RAG"])

    def test_deleting_a_note_removes_its_links(self) -> None:
        agent = self.vocab.upsert_entry("Agent", kind="topic")
        self.vocab.set_note_tags(self.note_ids[0], [agent.id])
        self.store.delete(self.note_ids[0])
        self.assertEqual(self.vocab.get(agent.id).note_count, 0)

    def test_merge_moves_aliases_and_links_and_keeps_old_name_resolvable(self) -> None:
        a = self.vocab.upsert_entry("智能体系统", kind="topic", aliases=["多智能体"])
        b = self.vocab.upsert_entry("Agent", kind="topic")
        self.vocab.set_note_tags(self.note_ids[0], [a.id])
        self.vocab.set_note_tags(self.note_ids[1], [a.id, b.id])
        merged = self.vocab.merge(a.id, b.id)
        self.assertEqual(merged.id, b.id)
        self.assertEqual(merged.note_count, 2)
        self.assertIn("智能体系统", merged.aliases)
        self.assertIn("多智能体", merged.aliases)
        self.assertEqual(self.vocab.resolve("智能体系统").id, b.id)
        self.assertEqual([e.canonical for e in self.vocab.list_entries()], ["Agent"])
        self.assertEqual(self.vocab.get(a.id).status, "merged")

    def test_rename_keeps_previous_name_as_alias(self) -> None:
        entry = self.vocab.upsert_entry("MCP协议", kind="entity")
        renamed = self.vocab.rename(entry.id, "MCP")
        self.assertEqual(renamed.canonical, "MCP")
        self.assertEqual(self.vocab.resolve("mcp协议").id, entry.id)
        other = self.vocab.upsert_entry("RAG", kind="topic")
        with self.assertRaises(ValueError):
            self.vocab.rename(other.id, "MCP")

    def test_alias_groups_and_prompt_listing(self) -> None:
        self.vocab.upsert_entry("Agent", kind="topic", aliases=["智能体"])
        self.vocab.upsert_entry("干货", kind="vacuous")
        groups = self.vocab.alias_groups()
        self.assertEqual(set(groups["智能体"]), {"Agent", "智能体"})
        listing = self.vocab.prompt_listing()
        self.assertIn("- Agent [topic]（别名：智能体）", listing)
        self.assertNotIn("干货", listing)

    def test_invalid_kind_and_source_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.vocab.upsert_entry("X", kind="nope")
        with self.assertRaises(ValueError):
            self.vocab.upsert_entry("X", source="nope")


if __name__ == "__main__":
    unittest.main()
