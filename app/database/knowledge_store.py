"""知识库存储模块 (SQLite + FTS5)"""
import os
import json
import sqlite3
import logging
import re
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, List
from app.config import KNOWLEDGE_DB_PATH

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """一条知识记录"""
    id: Optional[int] = None
    video_id: str = ""
    title: str = ""
    author: str = ""
    source_url: str = ""
    summary_markdown: str = ""
    tags: str = ""                  # 逗号分隔
    user_requirement: str = ""      # 用户的原始要求
    created_at: str = ""            # ISO 格式
    duration_seconds: float = 0.0
    video_code: str = ""            # 5位随机码 (uid)


class KnowledgeStore:
    """知识库管理器"""

    def __init__(self, db_path: str = KNOWLEDGE_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """初始化表结构和全文索引"""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    author TEXT NOT NULL DEFAULT '',
                    source_url TEXT NOT NULL DEFAULT '',
                    summary_markdown TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '',
                    user_requirement TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT '',
                    duration_seconds REAL NOT NULL DEFAULT 0.0,
                    video_code TEXT UNIQUE
                );

                -- FTS5 全文搜索虚拟表 (中文分词用 unicode61)
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    title,
                    author,
                    summary_markdown,
                    tags,
                    content='knowledge',
                    content_rowid='id',
                    tokenize='unicode61'
                );

                -- 自动同步触发器
                CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
                    INSERT INTO knowledge_fts(rowid, title, author, summary_markdown, tags)
                    VALUES (new.id, new.title, new.author, new.summary_markdown, new.tags);
                END;

                CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
                    INSERT INTO knowledge_fts(knowledge_fts, rowid, title, author, summary_markdown, tags)
                    VALUES ('delete', old.id, old.title, old.author, old.summary_markdown, old.tags);
                END;

                CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
                    INSERT INTO knowledge_fts(knowledge_fts, rowid, title, author, summary_markdown, tags)
                    VALUES ('delete', old.id, old.title, old.author, old.summary_markdown, old.tags);
                    INSERT INTO knowledge_fts(rowid, title, author, summary_markdown, tags)
                    VALUES (new.id, new.title, new.author, new.summary_markdown, new.tags);
                END;
            """)
            conn.commit()
            
            # Migration: Add video_code column if not exists
            try:
                cursor = conn.execute("PRAGMA table_info(knowledge)")
                columns = [input_row[1] for input_row in cursor.fetchall()]
                if "video_code" not in columns:
                    conn.execute("ALTER TABLE knowledge ADD COLUMN video_code TEXT")
                    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_video_code ON knowledge(video_code)")
                    conn.commit()
            except Exception:
                pass

            logger.info(f"知识库初始化完成: {self.db_path}")
        finally:
            conn.close()

    def save(self, entry: KnowledgeEntry) -> int:
        """保存知识记录"""
        if not entry.created_at:
            entry.created_at = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO knowledge
                   (video_id, title, author, source_url, summary_markdown,
                    tags, user_requirement, created_at, duration_seconds, video_code)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(video_id) DO UPDATE SET
                       title=excluded.title,
                       author=excluded.author,
                       summary_markdown=excluded.summary_markdown,
                       tags=excluded.tags,
                       user_requirement=excluded.user_requirement,
                       created_at=excluded.created_at,
                       video_code=excluded.video_code
                """,
                (
                    entry.video_id, entry.title, entry.author,
                    entry.source_url, entry.summary_markdown,
                    entry.tags, entry.user_requirement,
                    entry.created_at, entry.duration_seconds,
                    entry.video_code
                ),
            )
            conn.commit()
            entry_id = cursor.lastrowid
            logger.info(f"知识已保存: [{entry_id}] {entry.title}")
            return entry_id
        finally:
            conn.close()

    def search(self, query: str, limit: int = 10) -> List[dict]:
        """全文搜索 (尝试 FTS5, 失败回退到 LIKE)"""
        conn = self._get_conn()
        try:
            # FTS5 搜索
            rows = conn.execute(
                """SELECT k.id, k.video_id, k.title, k.author, k.tags,
                          k.source_url, k.created_at, k.duration_seconds, k.video_code,
                          snippet(knowledge_fts, 2, '**', '**', '...', 40) AS snippet
                   FROM knowledge_fts fts
                   JOIN knowledge k ON k.id = fts.rowid
                   WHERE knowledge_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            # 回退到 LIKE
            like = f"%{query}%"
            rows = conn.execute(
                """SELECT id, video_id, title, author, tags,
                          source_url, created_at, duration_seconds, video_code,
                          substr(summary_markdown, 1, 200) AS snippet
                   FROM knowledge
                   WHERE title LIKE ? OR author LIKE ?
                      OR summary_markdown LIKE ? OR tags LIKE ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (like, like, like, like, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_by_id(self, entry_id: int) -> Optional[dict]:
        """通过 ID 获取完整记录"""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM knowledge WHERE id = ?", (entry_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_by_video_id(self, video_id: str) -> Optional[dict]:
        """通过视频ID获取"""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM knowledge WHERE video_id = ?", (video_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_by_video_code(self, video_code: str) -> Optional[dict]:
        """通过视频码获取"""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM knowledge WHERE video_code = ?", (video_code,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_recent(self, limit: int = 20, offset: int = 0) -> List[dict]:
        """列出最近的记录"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT id, video_id, title, author, tags,
                          source_url, created_at, duration_seconds, video_code
                   FROM knowledge
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def list_by_tag(self, tag: str, limit: int = 20) -> List[dict]:
        """按标签筛选"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT id, video_id, title, author, tags,
                          source_url, created_at, duration_seconds, video_code
                   FROM knowledge
                   WHERE tags LIKE ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (f"%{tag}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete(self, entry_id: int) -> bool:
        """删除记录"""
        conn = self._get_conn()
        try:
            cursor = conn.execute("DELETE FROM knowledge WHERE id = ?", (entry_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def stats(self) -> dict:
        """数据库统计"""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) as total, MAX(created_at) as latest FROM knowledge").fetchone()
            return {
                "total_entries": row["total"],
                "latest_entry": row["latest"],
                "db_path": self.db_path,
            }
        finally:
            conn.close()


def extract_tags_from_markdown(markdown: str) -> str:
    """从 Markdown 中提取加粗的关键词作为标签"""
    bold_terms = re.findall(r'\*\*([^*]+)\*\*', markdown)
    # 取前15个, 去重, 去过短的
    seen = set()
    tags = []
    for term in bold_terms:
        t = term.strip()
        if len(t) >= 2 and t not in seen:
            seen.add(t)
            tags.append(t)
        if len(tags) >= 15:
            break
    return ",".join(tags)
