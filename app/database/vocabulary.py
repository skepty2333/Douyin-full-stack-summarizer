"""Controlled tag vocabulary: canonical entries, aliases, and note links.

Free-form tags fragment fast (the library reached 1,775 distinct tags for
253 notes, 87% used once, with 40 spellings of "Agent"). This module keeps a
small set of canonical entries instead. Every entry has a ``kind`` because
only topics and entities can grow into topic pages later; content types
(教程, 评测) and vacuous words (效率提升) are still allowed as tags but never
become topics.

Names are matched case-insensitively with whitespace removed, so "AI Agent",
"ai agent" and "AIAgent" resolve to the same entry.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

from app.config import KNOWLEDGE_DB_PATH

KINDS = ("topic", "entity", "content_type", "vacuous")
STATUSES = ("active", "merged", "archived")
SOURCES = ("seed", "ai", "user")
MAX_NAME_CHARS = 40


@dataclass(frozen=True)
class VocabEntry:
    id: int
    canonical: str
    kind: str
    aliases: tuple[str, ...]
    status: str = "active"
    merged_into: Optional[int] = None
    source: str = "ai"
    note_count: int = 0
    author_count: int = 0


def normalize_name(name: str) -> str:
    """Matching key: trimmed, whitespace removed, lower-cased, no leading #."""
    cleaned = re.sub(r"\s+", "", str(name or "")).strip().lstrip("#")
    return cleaned.lower()


def clean_display_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip().lstrip("#").strip()
    return cleaned[:MAX_NAME_CHARS]


class VocabularyStore:
    def __init__(self, db_path: str = KNOWLEDGE_DB_PATH):
        self.db_path = db_path
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
                CREATE TABLE IF NOT EXISTS vocabulary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL DEFAULT 'topic',
                    status TEXT NOT NULL DEFAULT 'active',
                    merged_into INTEGER,
                    source TEXT NOT NULL DEFAULT 'ai',
                    created_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS vocabulary_aliases (
                    norm TEXT PRIMARY KEY,
                    alias TEXT NOT NULL,
                    vocabulary_id INTEGER NOT NULL,
                    FOREIGN KEY (vocabulary_id) REFERENCES vocabulary(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_vocabulary_aliases_entry
                ON vocabulary_aliases(vocabulary_id);
                CREATE TABLE IF NOT EXISTS note_tags (
                    knowledge_id INTEGER NOT NULL,
                    vocabulary_id INTEGER NOT NULL,
                    source TEXT NOT NULL DEFAULT 'ai',
                    created_at TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (knowledge_id, vocabulary_id),
                    FOREIGN KEY (knowledge_id) REFERENCES knowledge(id) ON DELETE CASCADE,
                    FOREIGN KEY (vocabulary_id) REFERENCES vocabulary(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_note_tags_vocab ON note_tags(vocabulary_id);
                """
            )
            conn.commit()
        finally:
            conn.close()

    # ----------------------------------------------------------------- read
    @staticmethod
    def _entry_from_row(row: sqlite3.Row, aliases: Sequence[str], counts: tuple[int, int]) -> VocabEntry:
        return VocabEntry(
            id=int(row["id"]),
            canonical=str(row["canonical"]),
            kind=str(row["kind"]),
            aliases=tuple(aliases),
            status=str(row["status"]),
            merged_into=int(row["merged_into"]) if row["merged_into"] is not None else None,
            source=str(row["source"]),
            note_count=counts[0],
            author_count=counts[1],
        )

    def _aliases_by_entry(self, conn: sqlite3.Connection) -> dict[int, list[str]]:
        grouped: dict[int, list[str]] = {}
        for row in conn.execute(
            "SELECT vocabulary_id, alias FROM vocabulary_aliases ORDER BY rowid"
        ):
            grouped.setdefault(int(row["vocabulary_id"]), []).append(str(row["alias"]))
        return grouped

    def _counts_by_entry(self, conn: sqlite3.Connection) -> dict[int, tuple[int, int]]:
        counts: dict[int, tuple[int, int]] = {}
        for row in conn.execute(
            """SELECT t.vocabulary_id AS v, COUNT(*) AS notes, COUNT(DISTINCT k.author) AS authors
               FROM note_tags t JOIN knowledge k ON k.id = t.knowledge_id
               GROUP BY t.vocabulary_id"""
        ):
            counts[int(row["v"])] = (int(row["notes"]), int(row["authors"]))
        return counts

    def list_entries(self, *, active_only: bool = True) -> list[VocabEntry]:
        conn = self._get_conn()
        try:
            aliases = self._aliases_by_entry(conn)
            counts = self._counts_by_entry(conn)
            sql = "SELECT * FROM vocabulary"
            if active_only:
                sql += " WHERE status = 'active'"
            sql += " ORDER BY canonical"
            entries = []
            for row in conn.execute(sql):
                entry_id = int(row["id"])
                own = [a for a in aliases.get(entry_id, []) if normalize_name(a) != normalize_name(row["canonical"])]
                entries.append(self._entry_from_row(row, own, counts.get(entry_id, (0, 0))))
            return entries
        finally:
            conn.close()

    def get(self, entry_id: int) -> Optional[VocabEntry]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM vocabulary WHERE id = ?", (entry_id,)).fetchone()
            if row is None:
                return None
            aliases = [
                str(r["alias"])
                for r in conn.execute(
                    "SELECT alias FROM vocabulary_aliases WHERE vocabulary_id = ? ORDER BY rowid",
                    (entry_id,),
                )
                if normalize_name(r["alias"]) != normalize_name(row["canonical"])
            ]
            counts = self._counts_by_entry(conn).get(entry_id, (0, 0))
            return self._entry_from_row(row, aliases, counts)
        finally:
            conn.close()

    def resolve(self, name: str) -> Optional[VocabEntry]:
        """Find the active entry a canonical name or alias points to."""
        key = normalize_name(name)
        if not key:
            return None
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT vocabulary_id FROM vocabulary_aliases WHERE norm = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        entry = self.get(int(row["vocabulary_id"]))
        # Follow merges so callers always land on the surviving entry.
        seen = set()
        while entry is not None and entry.status == "merged" and entry.merged_into and entry.id not in seen:
            seen.add(entry.id)
            entry = self.get(entry.merged_into)
        return entry

    def alias_groups(self) -> dict[str, tuple[str, ...]]:
        """normalized name -> every spelling in its group (for query expansion)."""
        groups: dict[str, tuple[str, ...]] = {}
        for entry in self.list_entries():
            names = tuple(dict.fromkeys((entry.canonical, *entry.aliases)))
            for name in names:
                groups[normalize_name(name)] = names
        return groups

    def prompt_listing(self, *, kinds: Iterable[str] = ("topic", "entity", "content_type")) -> str:
        """Compact vocabulary text for the tagging prompt."""
        wanted = set(kinds)
        lines = []
        for entry in self.list_entries():
            if entry.kind not in wanted:
                continue
            alias_text = f"（别名：{'/'.join(entry.aliases[:6])}）" if entry.aliases else ""
            lines.append(f"- {entry.canonical} [{entry.kind}]{alias_text}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- write
    def upsert_entry(
        self,
        canonical: str,
        *,
        kind: str = "topic",
        aliases: Sequence[str] = (),
        source: str = "ai",
    ) -> VocabEntry:
        """Create the entry or attach new aliases to the one that already owns the name."""
        display = clean_display_name(canonical)
        if not display:
            raise ValueError("规范名不能为空")
        if kind not in KINDS:
            raise ValueError(f"未知 kind: {kind}")
        if source not in SOURCES:
            raise ValueError(f"未知 source: {source}")
        existing = self.resolve(display)
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            if existing is None:
                cursor = conn.execute(
                    """INSERT INTO vocabulary (canonical, kind, status, source, created_at, updated_at)
                       VALUES (?, ?, 'active', ?, ?, ?)""",
                    (display, kind, source, now, now),
                )
                entry_id = int(cursor.lastrowid)
                conn.execute(
                    "INSERT OR IGNORE INTO vocabulary_aliases (norm, alias, vocabulary_id) VALUES (?, ?, ?)",
                    (normalize_name(display), display, entry_id),
                )
            else:
                entry_id = existing.id
            for alias in aliases:
                alias_display = clean_display_name(alias)
                key = normalize_name(alias_display)
                if not key:
                    continue
                # An alias already owned by another entry is left alone: merging
                # is a deliberate action, not a side effect of tagging.
                conn.execute(
                    "INSERT OR IGNORE INTO vocabulary_aliases (norm, alias, vocabulary_id) VALUES (?, ?, ?)",
                    (key, alias_display, entry_id),
                )
            conn.execute("UPDATE vocabulary SET updated_at = ? WHERE id = ?", (now, entry_id))
            conn.commit()
        finally:
            conn.close()
        entry = self.get(entry_id)
        assert entry is not None
        return entry

    def rename(self, entry_id: int, new_canonical: str) -> VocabEntry:
        display = clean_display_name(new_canonical)
        if not display:
            raise ValueError("规范名不能为空")
        owner = self.resolve(display)
        if owner is not None and owner.id != entry_id:
            raise ValueError(f"名称已属于条目 {owner.canonical}")
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT canonical FROM vocabulary WHERE id = ?", (entry_id,)).fetchone()
            if row is None:
                raise KeyError(f"词表条目 {entry_id} 不存在")
            old = str(row["canonical"])
            conn.execute(
                "UPDATE vocabulary SET canonical = ?, updated_at = ? WHERE id = ?",
                (display, now, entry_id),
            )
            # The old name stays resolvable as an alias.
            conn.execute(
                "INSERT OR IGNORE INTO vocabulary_aliases (norm, alias, vocabulary_id) VALUES (?, ?, ?)",
                (normalize_name(old), old, entry_id),
            )
            conn.execute(
                "INSERT OR IGNORE INTO vocabulary_aliases (norm, alias, vocabulary_id) VALUES (?, ?, ?)",
                (normalize_name(display), display, entry_id),
            )
            conn.commit()
        finally:
            conn.close()
        entry = self.get(entry_id)
        assert entry is not None
        return entry

    def merge(self, from_id: int, into_id: int) -> VocabEntry:
        """Fold one entry into another: aliases and note links move, the old id stays resolvable."""
        if from_id == into_id:
            raise ValueError("不能把条目合并到自身")
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            for entry_id in (from_id, into_id):
                if conn.execute("SELECT 1 FROM vocabulary WHERE id = ?", (entry_id,)).fetchone() is None:
                    raise KeyError(f"词表条目 {entry_id} 不存在")
            conn.execute(
                "UPDATE vocabulary_aliases SET vocabulary_id = ? WHERE vocabulary_id = ?",
                (into_id, from_id),
            )
            conn.execute(
                """INSERT OR IGNORE INTO note_tags (knowledge_id, vocabulary_id, source, created_at)
                   SELECT knowledge_id, ?, source, created_at FROM note_tags WHERE vocabulary_id = ?""",
                (into_id, from_id),
            )
            conn.execute("DELETE FROM note_tags WHERE vocabulary_id = ?", (from_id,))
            conn.execute(
                "UPDATE vocabulary SET status = 'merged', merged_into = ?, updated_at = ? WHERE id = ?",
                (into_id, now, from_id),
            )
            conn.execute("UPDATE vocabulary SET updated_at = ? WHERE id = ?", (now, into_id))
            conn.commit()
        finally:
            conn.close()
        entry = self.get(into_id)
        assert entry is not None
        return entry

    def set_kind(self, entry_id: int, kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"未知 kind: {kind}")
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE vocabulary SET kind = ?, updated_at = ? WHERE id = ?",
                (kind, datetime.now(timezone.utc).isoformat(), entry_id),
            )
            conn.commit()
        finally:
            conn.close()

    def archive(self, entry_id: int) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE vocabulary SET status = 'archived', updated_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), entry_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------ note links
    def set_note_tags(self, knowledge_id: int, entry_ids: Iterable[int], *, source: str = "ai") -> int:
        """Replace a note's tag links; returns how many links were written."""
        if source not in SOURCES:
            raise ValueError(f"未知 source: {source}")
        ids = list(dict.fromkeys(int(i) for i in entry_ids))
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM note_tags WHERE knowledge_id = ?", (knowledge_id,))
            conn.executemany(
                "INSERT OR IGNORE INTO note_tags (knowledge_id, vocabulary_id, source, created_at) VALUES (?, ?, ?, ?)",
                [(knowledge_id, entry_id, source, now) for entry_id in ids],
            )
            conn.commit()
            return len(ids)
        finally:
            conn.close()

    def note_entries(self, knowledge_id: int) -> list[VocabEntry]:
        conn = self._get_conn()
        try:
            ids = [
                int(row["vocabulary_id"])
                for row in conn.execute(
                    "SELECT vocabulary_id FROM note_tags WHERE knowledge_id = ? ORDER BY rowid",
                    (knowledge_id,),
                )
            ]
        finally:
            conn.close()
        entries = [self.get(entry_id) for entry_id in ids]
        return [entry for entry in entries if entry is not None]

    def notes_for_entry(self, entry_id: int) -> list[int]:
        conn = self._get_conn()
        try:
            return [
                int(row["knowledge_id"])
                for row in conn.execute(
                    "SELECT knowledge_id FROM note_tags WHERE vocabulary_id = ? ORDER BY knowledge_id",
                    (entry_id,),
                )
            ]
        finally:
            conn.close()

    def untagged_note_ids(self) -> list[int]:
        conn = self._get_conn()
        try:
            return [
                int(row["id"])
                for row in conn.execute(
                    """SELECT k.id FROM knowledge k
                       WHERE NOT EXISTS (SELECT 1 FROM note_tags t WHERE t.knowledge_id = k.id)
                       ORDER BY k.id"""
                )
            ]
        finally:
            conn.close()

    def stats(self) -> dict:
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT
                     (SELECT COUNT(*) FROM vocabulary WHERE status = 'active') AS active,
                     (SELECT COUNT(*) FROM vocabulary) AS total,
                     (SELECT COUNT(*) FROM vocabulary_aliases) AS aliases,
                     (SELECT COUNT(DISTINCT knowledge_id) FROM note_tags) AS tagged_notes,
                     (SELECT COUNT(*) FROM knowledge) AS notes"""
            ).fetchone()
            return {
                "active_entries": int(row["active"]),
                "total_entries": int(row["total"]),
                "aliases": int(row["aliases"]),
                "tagged_notes": int(row["tagged_notes"]),
                "notes": int(row["notes"]),
            }
        finally:
            conn.close()
