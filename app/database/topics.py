"""Topics that grow out of the vocabulary, with compiled page versions.

A topic is a vocabulary entry that has gathered enough notes from enough
authors to be worth a synthesis page. Nothing here is hand-curated: entries
are *born* when they cross the density threshold, pages are compiled from
the notes' sections, new member notes mark the topic dirty, oversized topics
are split into children under a hub, and the user only keeps veto, rename
and merge. Every compiled page is kept as a version; notes remain the
evidence layer and are never modified.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from app.config import (
    KNOWLEDGE_DB_PATH,
    TOPIC_MAX_NOTES,
    TOPIC_MIN_AUTHORS,
    TOPIC_MIN_NOTES,
    TOPIC_RECOMPILE_DIRTY,
)
from app.database.vocabulary import VocabEntry, VocabularyStore

STATUSES = ("candidate", "active", "archived")
ELIGIBLE_KINDS = ("topic", "entity")


@dataclass(frozen=True)
class Topic:
    id: int
    vocabulary_id: int
    name: str
    kind: str
    aliases: tuple[str, ...]
    status: str
    is_hub: bool
    parent_id: Optional[int]
    dirty: int
    note_count: int
    author_count: int
    weight: float
    latest_version: int
    compiled_at: str
    born_at: str
    reason: str


@dataclass(frozen=True)
class TopicVersion:
    topic_id: int
    version: int
    markdown: str
    valid_as_of: str
    compiled_by: str
    source_note_ids: tuple[int, ...]
    notes_count: int
    input_chars: int
    created_at: str


@dataclass(frozen=True)
class MergeSuggestion:
    topic_a: Topic
    topic_b: Topic
    overlap: float


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TopicStore:
    def __init__(self, db_path: str = KNOWLEDGE_DB_PATH, *, vocabulary: Optional[VocabularyStore] = None):
        self.db_path = db_path
        self.vocabulary = vocabulary or VocabularyStore(db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vocabulary_id INTEGER NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'candidate',
                    is_hub INTEGER NOT NULL DEFAULT 0,
                    parent_id INTEGER,
                    dirty INTEGER NOT NULL DEFAULT 0,
                    reason TEXT NOT NULL DEFAULT '',
                    born_at TEXT NOT NULL DEFAULT '',
                    compiled_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (vocabulary_id) REFERENCES vocabulary(id) ON DELETE CASCADE,
                    FOREIGN KEY (parent_id) REFERENCES topics(id) ON DELETE SET NULL
                );
                CREATE TABLE IF NOT EXISTS topic_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    markdown TEXT NOT NULL,
                    valid_as_of TEXT NOT NULL DEFAULT '',
                    compiled_by TEXT NOT NULL DEFAULT '',
                    source_note_ids TEXT NOT NULL DEFAULT '[]',
                    notes_count INTEGER NOT NULL DEFAULT 0,
                    input_chars INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE,
                    UNIQUE (topic_id, version)
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ read
    def _build(self, row: sqlite3.Row, entry: VocabEntry, latest: int) -> Topic:
        return Topic(
            id=int(row["id"]),
            vocabulary_id=entry.id,
            name=entry.canonical,
            kind=entry.kind,
            aliases=entry.aliases,
            status=str(row["status"]),
            is_hub=bool(row["is_hub"]),
            parent_id=int(row["parent_id"]) if row["parent_id"] is not None else None,
            dirty=int(row["dirty"]),
            note_count=entry.note_count,
            author_count=entry.author_count,
            weight=entry.weight,
            latest_version=latest,
            compiled_at=str(row["compiled_at"] or ""),
            born_at=str(row["born_at"] or ""),
            reason=str(row["reason"] or ""),
        )

    def _latest_versions(self, conn: sqlite3.Connection) -> dict[int, int]:
        return {
            int(r["topic_id"]): int(r["v"])
            for r in conn.execute("SELECT topic_id, MAX(version) AS v FROM topic_versions GROUP BY topic_id")
        }

    def list_topics(self, *, statuses: Sequence[str] = ("candidate", "active")) -> list[Topic]:
        entries = {entry.id: entry for entry in self.vocabulary.list_entries(active_only=False)}
        conn = self._get_conn()
        try:
            latest = self._latest_versions(conn)
            rows = conn.execute("SELECT * FROM topics ORDER BY id").fetchall()
        finally:
            conn.close()
        topics = []
        for row in rows:
            if row["status"] not in statuses:
                continue
            entry = entries.get(int(row["vocabulary_id"]))
            if entry is None:
                continue
            topics.append(self._build(row, entry, latest.get(int(row["id"]), 0)))
        return topics

    def get(self, topic_id: int) -> Optional[Topic]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
            latest = self._latest_versions(conn).get(topic_id, 0) if row else 0
        finally:
            conn.close()
        if row is None:
            return None
        entry = self.vocabulary.get(int(row["vocabulary_id"]))
        if entry is None:
            return None
        return self._build(row, entry, latest)

    def get_by_vocabulary(self, vocabulary_id: int) -> Optional[Topic]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT id FROM topics WHERE vocabulary_id = ?", (vocabulary_id,)).fetchone()
        finally:
            conn.close()
        return self.get(int(row["id"])) if row else None

    def resolve(self, name: str) -> Optional[Topic]:
        """Topic by canonical name or any alias (follows vocabulary merges)."""
        entry = self.vocabulary.resolve(name)
        if entry is None:
            return None
        return self.get_by_vocabulary(entry.id)

    def children(self, topic_id: int) -> list[Topic]:
        return [t for t in self.list_topics(statuses=STATUSES) if t.parent_id == topic_id]

    def version(self, topic_id: int, version: Optional[int] = None) -> Optional[TopicVersion]:
        conn = self._get_conn()
        try:
            if version is None:
                row = conn.execute(
                    "SELECT * FROM topic_versions WHERE topic_id = ? ORDER BY version DESC LIMIT 1",
                    (topic_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM topic_versions WHERE topic_id = ? AND version = ?",
                    (topic_id, version),
                ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return TopicVersion(
            topic_id=int(row["topic_id"]),
            version=int(row["version"]),
            markdown=str(row["markdown"]),
            valid_as_of=str(row["valid_as_of"]),
            compiled_by=str(row["compiled_by"]),
            source_note_ids=tuple(int(i) for i in json.loads(row["source_note_ids"] or "[]")),
            notes_count=int(row["notes_count"]),
            input_chars=int(row["input_chars"]),
            created_at=str(row["created_at"]),
        )

    def member_note_ids(self, topic: Topic) -> list[int]:
        return self.vocabulary.notes_for_entry(topic.vocabulary_id)

    # --------------------------------------------------------------- growth
    def eligible_entries(self) -> list[VocabEntry]:
        """Vocabulary entries dense enough to be born, that have no topic yet."""
        existing = {
            t.vocabulary_id for t in self.list_topics(statuses=STATUSES)
        }
        return [
            entry
            for entry in self.vocabulary.list_entries()
            if entry.kind in ELIGIBLE_KINDS
            and entry.id not in existing
            and entry.weight >= TOPIC_MIN_NOTES
            and entry.author_count >= TOPIC_MIN_AUTHORS
        ]

    def born(self, entry: VocabEntry, *, reason: str = "", parent_id: Optional[int] = None) -> Topic:
        now = _now()
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO topics
                   (vocabulary_id, status, parent_id, dirty, reason, born_at, updated_at)
                   VALUES (?, 'candidate', ?, 0, ?, ?, ?)""",
                (entry.id, parent_id, reason or f"{entry.note_count} 条笔记（主次加权 {entry.weight:g}）/ {entry.author_count} 位作者", now, now),
            )
            conn.commit()
        finally:
            conn.close()
        topic = self.get_by_vocabulary(entry.id)
        assert topic is not None
        return topic

    def note_tagged(self, knowledge_id: int) -> list[Topic]:
        """Mark topics touched by a newly tagged note dirty; return those due for recompile."""
        due: list[Topic] = []
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT t.id FROM topics t
                   JOIN note_tags n ON n.vocabulary_id = t.vocabulary_id
                   WHERE n.knowledge_id = ? AND t.status IN ('candidate', 'active')""",
                (knowledge_id,),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE topics SET dirty = dirty + 1, updated_at = ? WHERE id = ?",
                    (_now(), int(row["id"])),
                )
            conn.commit()
        finally:
            conn.close()
        for row in rows:
            topic = self.get(int(row["id"]))
            if topic is not None and topic.dirty >= TOPIC_RECOMPILE_DIRTY:
                due.append(topic)
        return due

    def mark_dirty(self, topic_id: int, amount: int = 1) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE topics SET dirty = dirty + ?, updated_at = ? WHERE id = ?",
                (max(1, int(amount)), _now(), topic_id),
            )
            conn.commit()
        finally:
            conn.close()

    def needs_split(self, topic: Topic) -> bool:
        return not topic.is_hub and topic.note_count > TOPIC_MAX_NOTES

    def mark_hub(self, topic_id: int) -> None:
        conn = self._get_conn()
        try:
            conn.execute("UPDATE topics SET is_hub = 1, updated_at = ? WHERE id = ?", (_now(), topic_id))
            conn.commit()
        finally:
            conn.close()

    def set_parent(self, topic_id: int, parent_id: Optional[int]) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE topics SET parent_id = ?, updated_at = ? WHERE id = ?", (parent_id, _now(), topic_id)
            )
            conn.commit()
        finally:
            conn.close()

    def save_version(
        self,
        topic_id: int,
        markdown: str,
        *,
        compiled_by: str,
        source_note_ids: Sequence[int],
        input_chars: int,
        activate: bool = True,
    ) -> TopicVersion:
        now = _now()
        conn = self._get_conn()
        try:
            latest = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM topic_versions WHERE topic_id = ?", (topic_id,)
            ).fetchone()["v"]
            version = int(latest) + 1
            ids = sorted({int(i) for i in source_note_ids})
            conn.execute(
                """INSERT INTO topic_versions
                   (topic_id, version, markdown, valid_as_of, compiled_by, source_note_ids,
                    notes_count, input_chars, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (topic_id, version, markdown, now[:10], compiled_by, json.dumps(ids), len(ids), input_chars, now),
            )
            status_sql = ", status = 'active'" if activate else ""
            conn.execute(
                f"UPDATE topics SET dirty = 0, compiled_at = ?, updated_at = ?{status_sql} WHERE id = ?",
                (now, now, topic_id),
            )
            conn.commit()
        finally:
            conn.close()
        saved = self.version(topic_id, version)
        assert saved is not None
        return saved

    # ------------------------------------------------------------ curation
    def set_status(self, topic_id: int, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"未知状态: {status}")
        conn = self._get_conn()
        try:
            conn.execute("UPDATE topics SET status = ?, updated_at = ? WHERE id = ?", (status, _now(), topic_id))
            conn.commit()
        finally:
            conn.close()

    def merge_suggestions(self, *, min_overlap: float = 0.5) -> list[MergeSuggestion]:
        """Pairs of active topics whose member sets overlap heavily (Jaccard)."""
        topics = [t for t in self.list_topics(statuses=("active",)) if not t.is_hub]
        members = {t.id: set(self.member_note_ids(t)) for t in topics}
        suggestions: list[MergeSuggestion] = []
        for i, a in enumerate(topics):
            for b in topics[i + 1 :]:
                union = members[a.id] | members[b.id]
                if not union:
                    continue
                overlap = len(members[a.id] & members[b.id]) / len(union)
                if overlap >= min_overlap:
                    suggestions.append(MergeSuggestion(topic_a=a, topic_b=b, overlap=overlap))
        suggestions.sort(key=lambda s: -s.overlap)
        return suggestions

    def stats(self) -> dict:
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT
                     SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active,
                     SUM(CASE WHEN status = 'candidate' THEN 1 ELSE 0 END) AS candidate,
                     SUM(CASE WHEN status = 'archived' THEN 1 ELSE 0 END) AS archived,
                     SUM(CASE WHEN dirty > 0 THEN 1 ELSE 0 END) AS dirty,
                     (SELECT COUNT(*) FROM topic_versions) AS versions
                   FROM topics"""
            ).fetchone()
            return {
                "active": int(row["active"] or 0),
                "candidate": int(row["candidate"] or 0),
                "archived": int(row["archived"] or 0),
                "dirty": int(row["dirty"] or 0),
                "versions": int(row["versions"] or 0),
            }
        finally:
            conn.close()
