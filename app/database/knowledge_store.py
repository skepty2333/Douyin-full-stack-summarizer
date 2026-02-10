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
    timestamp: str = ""             # 北京时间


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
        """初始化表结构和全文索引 (强制重置以支持 Schema 变更)"""
        conn = self._get_conn()
        try:
            # 强制重置
            conn.executescript("""
                DROP TABLE IF EXISTS knowledge;
                DROP TABLE IF EXISTS knowledge_fts;
                CREATE TABLE knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,         -- 不再唯一，允许同一视频多条记录
                    title TEXT NOT NULL DEFAULT '',
                    author TEXT NOT NULL DEFAULT '',
                    source_url TEXT NOT NULL DEFAULT '',
                    summary_markdown TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '',
                    user_requirement TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT '',
                    duration_seconds REAL NOT NULL DEFAULT 0.0,
                    video_code TEXT UNIQUE,
                    timestamp TEXT NOT NULL DEFAULT '' -- 北京时间戳
                );

                -- FTS5 全文搜索虚拟表 (中文分词用 unicode61)
                CREATE VIRTUAL TABLE knowledge_fts USING fts5(
                    title,
                    author,
                    summary_markdown,
                    tags,
                    content='knowledge',
                    content_rowid='id',
                    tokenize='unicode61'
                );

                -- 自动同步触发器
                CREATE TRIGGER knowledge_ai AFTER INSERT ON knowledge BEGIN
                    INSERT INTO knowledge_fts(rowid, title, author, summary_markdown, tags)
                    VALUES (new.id, new.title, new.author, new.summary_markdown, new.tags);
                END;

                CREATE TRIGGER knowledge_ad AFTER DELETE ON knowledge BEGIN
                    INSERT INTO knowledge_fts(knowledge_fts, rowid, title, author, summary_markdown, tags)
                    VALUES ('delete', old.id, old.title, old.author, old.summary_markdown, old.tags);
                END;

                CREATE TRIGGER knowledge_au AFTER UPDATE ON knowledge BEGIN
                    INSERT INTO knowledge_fts(knowledge_fts, rowid, title, author, summary_markdown, tags)
                    VALUES ('delete', old.id, old.title, old.author, old.summary_markdown, old.tags);
                    INSERT INTO knowledge_fts(rowid, title, author, summary_markdown, tags)
                    VALUES (new.id, new.title, new.author, new.summary_markdown, new.tags);
                END;
            """)
            conn.commit()
            logger.info(f"知识库初始化完成 (重置): {self.db_path}")
        finally:
            conn.close()

    def save(self, entry: KnowledgeEntry) -> int:
        """保存知识记录"""
        if not entry.created_at:
            entry.created_at = datetime.now(timezone.utc).isoformat()
        
        # 强制更新时间戳为北京时间 (简单起见，这里直接生成字符串)
        # 注意: 实际应该用 pytz 或 zoneinfo，但为了减少依赖，这里简单处理 +8
        from datetime import timedelta
        beijing_time = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        entry.timestamp = beijing_time

        conn = self._get_conn()
        try:
            # 这里的逻辑通过 video_code 唯一性来判断是否覆盖 (Overwrite)
            # 如果是 "新增" (New)，video_code 应该是新的，所以是 INSERT
            # 如果是 "覆盖" (Overwrite)，video_code 应该是旧的，所以是 UPDATE (ON CONFLICT)
            
            cursor = conn.execute(
                """INSERT INTO knowledge
                   (video_id, title, author, source_url, summary_markdown,
                    tags, user_requirement, created_at, duration_seconds, video_code, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(video_code) DO UPDATE SET
                       title=excluded.title,
                       author=excluded.author,
                       summary_markdown=excluded.summary_markdown,
                       tags=excluded.tags,
                       user_requirement=excluded.user_requirement,
                       created_at=excluded.created_at,
                       timestamp=excluded.timestamp
                """,
                (
                    entry.video_id, entry.title, entry.author,
                    entry.source_url, entry.summary_markdown,
                    entry.tags, entry.user_requirement,
                    entry.created_at, entry.duration_seconds,
                    entry.video_code, entry.timestamp
                ),
            )
            conn.commit()
            entry_id = cursor.lastrowid
            logger.info(f"知识已保存: [{entry_id}] {entry.title}")
            return entry_id
        finally:
            conn.close()

    def get_by_title_and_author(self, title: str, author: str) -> List[dict]:
        """通过标题和作者查找重复视频"""
        conn = self._get_conn()
        try:
            # 简单的精确匹配，实际可能需要模糊匹配？用户要求"双重合"，假设是精确匹配
            rows = conn.execute(
                "SELECT * FROM knowledge WHERE title = ? AND author = ? ORDER BY created_at DESC", 
                (title, author)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def search(self, query: str, limit: int = 10) -> List[dict]:
        """全文搜索 (尝试 FTS5, 失败回退到 LIKE)"""
        conn = self._get_conn()
        try:
            # FTS5 搜索
            rows = conn.execute(
                """SELECT k.id, k.video_id, k.title, k.author, k.tags,
                          k.source_url, k.created_at, k.duration_seconds, k.video_code, k.timestamp,
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
                          source_url, created_at, duration_seconds, video_code, timestamp,
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
        """通过视频ID获取 (可能返回多条，这里只返回最新一条)"""
        conn = self._get_conn()
        try:
            # 修改为按时间倒序取最新
            row = conn.execute("SELECT * FROM knowledge WHERE video_id = ? ORDER BY created_at DESC LIMIT 1", (video_id,)).fetchone()
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

    def delete_by_video_code(self, video_code: str) -> bool:
        """通过视频码删除记录"""
        conn = self._get_conn()
        try:
            cursor = conn.execute("DELETE FROM knowledge WHERE video_code = ?", (video_code,))
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
