"""Offline tests for note classification parsing and vocabulary application."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-key"}, clear=False):
    with patch("dotenv.load_dotenv", return_value=False):
        from app.database.knowledge_store import KnowledgeEntry, KnowledgeStore
        from app.database.vocabulary import VocabularyStore
        from app.services import note_tagging
        from app.services.note_tagging import (
            apply_classification,
            classify_note,
            parse_classification,
            resolve_tags,
        )


class ParseClassificationTests(unittest.TestCase):
    def test_valid_json_is_parsed(self) -> None:
        raw = '{"domain": "ai", "temporality": "version_sensitive", "tags": [{"name": "OpenClaw", "kind": "entity"}, {"name": "Agent", "kind": "topic"}]}'
        result = parse_classification(raw)
        self.assertEqual((result.domain, result.temporality), ("ai", "version_sensitive"))
        self.assertEqual([(t.name, t.kind) for t in result.tags], [("OpenClaw", "entity"), ("Agent", "topic")])

    def test_misspelled_temporality_key_and_code_fence_are_tolerated(self) -> None:
        raw = '```json\n{"domain": "trading", "temporability": "stable", "tags": ["止损"]}\n```'
        result = parse_classification(raw)
        self.assertEqual(result.temporality, "stable")
        self.assertEqual(result.tags[0].kind, "topic")

    def test_invalid_domain_or_empty_tags_raise(self) -> None:
        with self.assertRaises(ValueError):
            parse_classification('{"domain": "sports", "temporality": "stable", "tags": ["x"]}')
        with self.assertRaises(ValueError):
            parse_classification('{"domain": "ai", "temporality": "stable", "tags": ["效率提升技巧"]}')

    def test_vacuous_names_duplicates_and_overflow_are_dropped(self) -> None:
        tags = [{"name": "干货分享", "kind": "topic"}, {"name": "RAG", "kind": "topic"}, {"name": "rag", "kind": "topic"}]
        tags += [{"name": f"主题{i}", "kind": "topic"} for i in range(10)]
        raw = '{"domain": "ai", "temporality": "stable", "tags": %s}' % __import__("json").dumps(tags, ensure_ascii=False)
        result = parse_classification(raw)
        names = [t.name for t in result.tags]
        self.assertNotIn("干货分享", names)
        self.assertEqual(names.count("RAG"), 1)
        self.assertLessEqual(len(names), note_tagging.MAX_TAGS)


class ApplyClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        db_path = str(root / "knowledge.db")
        self.store = KnowledgeStore(db_path, asset_root=str(root / "assets"))
        self.vocab = VocabularyStore(db_path)
        self.vocab.upsert_entry("Agent", kind="topic", aliases=["智能体"], source="seed")
        self.note_id = self.store.save(
            KnowledgeEntry(
                video_id="vid-1",
                title="测试",
                author="offline",
                source_url="https://example.test/1",
                summary_markdown="# 标题\n\n## 正文\n\n内容。",
                tags="旧标签",
                video_code="tag01",
            )
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_resolve_maps_aliases_and_creates_only_unknown_names(self) -> None:
        classification = parse_classification(
            '{"domain": "ai", "temporality": "stable", "tags": [{"name": "智能体", "kind": "entity"}, {"name": "Hermes", "kind": "entity"}, {"name": "agent", "kind": "topic"}]}'
        )
        applied = resolve_tags(classification, vocabulary=self.vocab)
        self.assertEqual(applied.canonical_names, ("Agent", "Hermes"))
        self.assertEqual(applied.created, ("Hermes",))
        self.assertEqual(self.vocab.resolve("hermes").kind, "entity")

    def test_apply_updates_note_columns_and_links(self) -> None:
        classification = parse_classification(
            '{"domain": "ai", "temporality": "version_sensitive", "tags": [{"name": "智能体", "kind": "topic"}]}'
        )
        apply_classification(self.note_id, classification, store=self.store, vocabulary=self.vocab)
        note = self.store.get_by_id(self.note_id)
        self.assertEqual((note["domain"], note["temporality"], note["tags"]), ("ai", "version_sensitive", "Agent"))
        self.assertEqual([e.canonical for e in self.vocab.note_entries(self.note_id)], ["Agent"])

    def test_classify_note_returns_none_on_model_failure(self) -> None:
        with patch.object(note_tagging.aliyun_client, "chat", AsyncMock(side_effect=RuntimeError("down"))):
            result = asyncio.run(classify_note("正文", "标题", "作者", vocabulary=self.vocab))
        self.assertIsNone(result)

    def test_classify_note_injects_vocabulary_into_prompt(self) -> None:
        fake = AsyncMock(return_value='{"domain": "ai", "temporality": "stable", "tags": [{"name": "Agent", "kind": "topic"}]}')
        with patch.object(note_tagging.aliyun_client, "chat", fake):
            result = asyncio.run(classify_note("正文", "标题", "作者", vocabulary=self.vocab))
        self.assertIsNotNone(result)
        user_message = fake.call_args.kwargs["messages"][1]["content"]
        self.assertIn("- Agent [topic]（别名：智能体）", user_message)
        self.assertEqual(fake.call_args.kwargs["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
