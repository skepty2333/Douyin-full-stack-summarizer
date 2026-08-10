"""知识库存储模块 (SQLite + FTS5)"""
import os
import hashlib
import json
import sqlite3
import logging
import math
import re
import stat
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Sequence
from app.config import KNOWLEDGE_ASSETS_DIR, KNOWLEDGE_DB_PATH

logger = logging.getLogger(__name__)

ASSET_KEY_RE = re.compile(r"^V\d{4}$")
ASSET_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ASSET_RELATIVE_PATH_RE = re.compile(r"^blobs/[0-9a-f]{2}/[0-9a-f]{64}\.jpg$")
MAX_ASSETS_PER_NOTE = 8
MAX_ASSET_BYTES = 2 * 1024 * 1024


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


@dataclass(frozen=True)
class KnowledgeAsset:
    """Metadata for one persistent, reviewed image attachment."""

    asset_key: str
    relative_path: str
    mime_type: str
    timestamp_ms: int
    caption: str
    kind: str
    confidence: str
    width: int
    height: int
    byte_size: int
    sha256: str
    display_order: int
    quality_score: float = 0.0


class KnowledgeStore:
    """知识库管理器"""

    def __init__(
        self,
        db_path: str = KNOWLEDGE_DB_PATH,
        asset_root: Optional[str] = None,
    ):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        if asset_root is None:
            asset_root = (
                KNOWLEDGE_ASSETS_DIR
                if os.path.abspath(db_path) == os.path.abspath(KNOWLEDGE_DB_PATH)
                else os.path.join(os.path.dirname(os.path.abspath(db_path)), "knowledge_assets")
            )
        self.asset_root = self._prepare_asset_root(asset_root)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _prepare_asset_root(asset_root: str) -> Path:
        root = Path(asset_root)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("KNOWLEDGE_ASSETS_DIR 必须是无跳转段的绝对路径")
        if not root.exists():
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink():
            raise ValueError("KNOWLEDGE_ASSETS_DIR 不能是符号链接")
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("KNOWLEDGE_ASSETS_DIR 不是目录")
        return resolved

    @staticmethod
    def _validate_assets(
        assets: Sequence[KnowledgeAsset],
    ) -> tuple[KnowledgeAsset, ...]:
        if isinstance(assets, (str, bytes, bytearray)):
            raise TypeError("assets 必须是图片元数据序列")
        bounded = tuple(assets)
        if len(bounded) > MAX_ASSETS_PER_NOTE:
            raise ValueError("单条笔记图片数量超过上限")
        seen_keys: set[str] = set()
        seen_orders: set[int] = set()
        for asset in bounded:
            if not isinstance(asset, KnowledgeAsset):
                raise TypeError("assets 包含无效对象")
            if not ASSET_KEY_RE.fullmatch(asset.asset_key) or asset.asset_key in seen_keys:
                raise ValueError("图片 ID 无效或重复")
            if not ASSET_SHA256_RE.fullmatch(asset.sha256):
                raise ValueError("图片摘要无效")
            expected_path = f"blobs/{asset.sha256[:2]}/{asset.sha256}.jpg"
            if (
                asset.relative_path != expected_path
                or not ASSET_RELATIVE_PATH_RE.fullmatch(asset.relative_path)
            ):
                raise ValueError("图片相对路径无效")
            if asset.mime_type != "image/jpeg":
                raise ValueError("只允许 JPEG 图片")
            if not (0 < asset.byte_size <= MAX_ASSET_BYTES):
                raise ValueError("图片大小无效")
            if not (0 < asset.width <= 1600 and 0 < asset.height <= 1600):
                raise ValueError("图片尺寸无效")
            if not (0 <= asset.timestamp_ms <= 6 * 60 * 60 * 1000):
                raise ValueError("图片时间点无效")
            if not asset.caption.strip() or len(asset.caption) > 1000:
                raise ValueError("图片说明无效")
            if not (0 <= asset.display_order < MAX_ASSETS_PER_NOTE):
                raise ValueError("图片顺序无效")
            if asset.display_order in seen_orders:
                raise ValueError("图片顺序重复")
            if not math.isfinite(float(asset.quality_score)):
                raise ValueError("图片质量分数无效")
            seen_keys.add(asset.asset_key)
            seen_orders.add(asset.display_order)
        return bounded

    def _resolve_asset_path(self, relative_path: str, *, must_exist: bool) -> Path:
        if not ASSET_RELATIVE_PATH_RE.fullmatch(relative_path):
            raise ValueError("图片相对路径无效")
        candidate = self.asset_root.joinpath(*Path(relative_path).parts)
        if candidate.is_symlink():
            raise ValueError("图片不能是符号链接")
        resolved = candidate.resolve(strict=must_exist)
        if os.path.commonpath((str(self.asset_root), str(resolved))) != str(self.asset_root):
            raise ValueError("图片路径超出资产目录")
        return resolved

    def _read_verified_asset(self, metadata: dict) -> bytes:
        relative_path = str(metadata.get("relative_path", ""))
        digest = str(metadata.get("sha256", ""))
        if not ASSET_SHA256_RE.fullmatch(digest):
            raise ValueError("图片摘要无效")
        path = self._resolve_asset_path(relative_path, must_exist=True)
        info = path.stat()
        expected_size = int(metadata.get("byte_size", 0))
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size != expected_size
            or not (0 < info.st_size <= MAX_ASSET_BYTES)
        ):
            raise ValueError("图片文件大小无效")
        payload = path.read_bytes()
        if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise ValueError("图片不是完整 JPEG")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("图片内容校验失败")
        return payload

    def prune_orphan_assets(
        self,
        *,
        min_age_seconds: float = 24 * 60 * 60,
        max_files: int = 10_000,
    ) -> int:
        """Delete old unreferenced blobs during startup, before jobs are accepted.

        Immediate deletion after an overwrite is deliberately avoided: another
        concurrent job may have already reused the same content-addressed blob
        but not committed its manifest yet.  A grace period plus startup-only
        collection prevents that race while bounding long-term disk growth.
        """

        if (
            isinstance(min_age_seconds, bool)
            or not isinstance(min_age_seconds, (int, float))
            or not math.isfinite(float(min_age_seconds))
            or min_age_seconds < 0
        ):
            raise ValueError("孤儿图片保留时间无效")
        if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files <= 0:
            raise ValueError("孤儿图片扫描上限无效")

        conn = self._get_conn()
        try:
            referenced = {
                str(row["relative_path"])
                for row in conn.execute(
                    "SELECT DISTINCT relative_path FROM knowledge_assets"
                ).fetchall()
            }
        finally:
            conn.close()

        blobs_root = self.asset_root / "blobs"
        if not blobs_root.exists() or blobs_root.is_symlink() or not blobs_root.is_dir():
            return 0
        cutoff = time.time() - float(min_age_seconds)
        removed = 0
        scanned = 0
        for prefix_dir in sorted(blobs_root.iterdir()):
            if scanned >= max_files:
                break
            if (
                prefix_dir.is_symlink()
                or not prefix_dir.is_dir()
                or not re.fullmatch(r"[0-9a-f]{2}", prefix_dir.name)
            ):
                continue
            for path in sorted(prefix_dir.iterdir()):
                if scanned >= max_files:
                    break
                scanned += 1
                relative_path = f"blobs/{prefix_dir.name}/{path.name}"
                if relative_path in referenced or not ASSET_RELATIVE_PATH_RE.fullmatch(relative_path):
                    continue
                try:
                    info = path.lstat()
                    if (
                        stat.S_ISREG(info.st_mode)
                        and not path.is_symlink()
                        and info.st_mtime <= cutoff
                    ):
                        path.unlink()
                        removed += 1
                except OSError as exc:
                    logger.warning("清理孤儿知识图片失败: %s", exc)
        if removed:
            logger.info("已清理 %s 个过期孤儿知识图片", removed)
        return removed

    def _init_db(self):
        """初始化表结构和全文索引 (非破坏性: 仅在不存在时创建)"""
        conn = self._get_conn()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            # 使用 IF NOT EXISTS 避免覆盖现有数据
            # 触发器采用先删后建策略，确保逻辑更新
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge (
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

                CREATE TABLE IF NOT EXISTS knowledge_assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    knowledge_id INTEGER NOT NULL,
                    asset_key TEXT NOT NULL,
                    asset_type TEXT NOT NULL DEFAULT 'video_frame',
                    mime_type TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    caption TEXT NOT NULL DEFAULT '',
                    annotation_kind TEXT NOT NULL DEFAULT '',
                    confidence TEXT NOT NULL DEFAULT '',
                    quality_score REAL NOT NULL DEFAULT 0.0,
                    display_order INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (knowledge_id) REFERENCES knowledge(id) ON DELETE CASCADE,
                    UNIQUE (knowledge_id, asset_key)
                );

                CREATE INDEX IF NOT EXISTS idx_knowledge_assets_note_order
                ON knowledge_assets(knowledge_id, display_order, id);

                CREATE INDEX IF NOT EXISTS idx_knowledge_assets_sha256
                ON knowledge_assets(sha256);
            """)
            conn.commit()
            logger.info(f"知识库初始化完成 (持久化模式): {self.db_path}")
        finally:
            conn.close()

    def save(
        self,
        entry: KnowledgeEntry,
        allow_overwrite: bool = False,
        assets: Optional[Sequence[KnowledgeAsset]] = None,
    ) -> int:
        """Save one note and, when supplied, atomically replace its asset manifest."""
        validated_assets = self._validate_assets(assets) if assets is not None else None
        if not entry.created_at:
            entry.created_at = datetime.now(timezone.utc).isoformat()
        
        # 强制更新时间戳为北京时间 (简单起见，这里直接生成字符串)
        # 注意: 实际应该用 pytz 或 zoneinfo，但为了减少依赖，这里简单处理 +8
        from datetime import timedelta
        beijing_time = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        entry.timestamp = beijing_time

        conn = self._get_conn()
        try:
            values = (
                entry.video_id, entry.title, entry.author,
                entry.source_url, entry.summary_markdown,
                entry.tags, entry.user_requirement,
                entry.created_at, entry.duration_seconds,
                entry.video_code, entry.timestamp,
            )
            base_sql = """INSERT INTO knowledge
                (video_id, title, author, source_url, summary_markdown,
                 tags, user_requirement, created_at, duration_seconds, video_code, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            if allow_overwrite:
                base_sql += """
                    ON CONFLICT(video_code) DO UPDATE SET
                        video_id=excluded.video_id,
                        title=excluded.title,
                        author=excluded.author,
                        source_url=excluded.source_url,
                        summary_markdown=excluded.summary_markdown,
                        tags=excluded.tags,
                        user_requirement=excluded.user_requirement,
                        created_at=excluded.created_at,
                        duration_seconds=excluded.duration_seconds,
                        timestamp=excluded.timestamp
                """
            cursor = conn.execute(base_sql, values)
            if allow_overwrite:
                row = conn.execute(
                    "SELECT id FROM knowledge WHERE video_code = ?", (entry.video_code,)
                ).fetchone()
                entry_id = int(row["id"])
            else:
                entry_id = int(cursor.lastrowid)
            if validated_assets is not None:
                conn.execute(
                    "DELETE FROM knowledge_assets WHERE knowledge_id = ?",
                    (entry_id,),
                )
                conn.executemany(
                    """INSERT INTO knowledge_assets
                       (knowledge_id, asset_key, asset_type, mime_type, relative_path,
                        sha256, byte_size, width, height, timestamp_ms, caption,
                        annotation_kind, confidence, quality_score, display_order, created_at)
                       VALUES (?, ?, 'video_frame', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            entry_id,
                            asset.asset_key,
                            asset.mime_type,
                            asset.relative_path,
                            asset.sha256,
                            asset.byte_size,
                            asset.width,
                            asset.height,
                            asset.timestamp_ms,
                            asset.caption,
                            asset.kind,
                            asset.confidence,
                            asset.quality_score,
                            asset.display_order,
                            entry.created_at,
                        )
                        for asset in validated_assets
                    ],
                )
            conn.commit()
            logger.info(f"知识已保存: [{entry_id}] {entry.title}")
            return entry_id
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def list_assets(self, entry_id: int) -> List[dict]:
        """Return ordered, path-free image metadata for one note."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT asset_key, asset_type, mime_type, timestamp_ms, caption,
                          annotation_kind, confidence, width, height, byte_size,
                          display_order
                   FROM knowledge_assets
                   WHERE knowledge_id = ?
                   ORDER BY display_order, id""",
                (entry_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def read_asset_by_video_code(self, video_code: str, asset_key: str) -> bytes:
        """Resolve one logical asset reference and return verified JPEG bytes."""
        if not isinstance(video_code, str) or not re.fullmatch(r"[A-Za-z0-9]{1,32}", video_code):
            raise ValueError("视频码无效")
        if not isinstance(asset_key, str) or not ASSET_KEY_RE.fullmatch(asset_key):
            raise ValueError("图片 ID 无效")
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT a.* FROM knowledge_assets a
                   JOIN knowledge k ON k.id = a.knowledge_id
                   WHERE k.video_code = ? AND a.asset_key = ?""",
                (video_code, asset_key),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError("图片不存在")
        return self._read_verified_asset(dict(row))

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
        """宽松全文搜索：标签优先 + 多关键词 OR 匹配"""
        conn = self._get_conn()
        try:
            results = []
            seen_ids = set()

            # 拆分关键词
            keywords = [k.strip() for k in query.replace(",", " ").replace("，", " ").split() if k.strip()]
            if not keywords:
                keywords = [query.strip()]

            # 策略1：标签精确匹配（优先级最高）
            for kw in keywords:
                rows = conn.execute(
                    """SELECT id, video_id, title, author, tags,
                              source_url, created_at, duration_seconds, video_code, timestamp,
                              substr(summary_markdown, 1, 200) AS snippet
                       FROM knowledge
                       WHERE tags LIKE ?
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (f"%{kw}%", limit),
                ).fetchall()
                for r in rows:
                    d = dict(r)
                    if d["id"] not in seen_ids:
                        seen_ids.add(d["id"])
                        results.append(d)

            # 策略2：FTS5 全文搜索
            if len(results) < limit:
                try:
                    fts_query = " OR ".join(keywords)
                    rows = conn.execute(
                        """SELECT k.id, k.video_id, k.title, k.author, k.tags,
                                  k.source_url, k.created_at, k.duration_seconds, k.video_code, k.timestamp,
                                  snippet(knowledge_fts, 2, '**', '**', '...', 40) AS snippet
                           FROM knowledge_fts fts
                           JOIN knowledge k ON k.id = fts.rowid
                           WHERE knowledge_fts MATCH ?
                           ORDER BY rank
                           LIMIT ?""",
                        (fts_query, limit),
                    ).fetchall()
                    for r in rows:
                        d = dict(r)
                        if d["id"] not in seen_ids:
                            seen_ids.add(d["id"])
                            results.append(d)
                except Exception as exc:
                    logger.warning("FTS5 搜索失败，使用 LIKE 兜底: %s", exc)

            # 策略3：LIKE 兜底（标题 + 正文 + 标签）
            if len(results) < limit:
                for kw in keywords:
                    like = f"%{kw}%"
                    rows = conn.execute(
                        """SELECT id, video_id, title, author, tags,
                                  source_url, created_at, duration_seconds, video_code, timestamp,
                                  substr(summary_markdown, 1, 200) AS snippet
                           FROM knowledge
                           WHERE title LIKE ? OR summary_markdown LIKE ? OR tags LIKE ?
                           ORDER BY created_at DESC
                           LIMIT ?""",
                        (like, like, like, limit),
                    ).fetchall()
                    for r in rows:
                        d = dict(r)
                        if d["id"] not in seen_ids:
                            seen_ids.add(d["id"])
                            results.append(d)

            return results[:limit]
        finally:
            conn.close()

    def search_precise(self, query: str, limit: int = 20) -> List[dict]:
        """精确搜索：所有关键词必须同时命中（AND 逻辑）"""
        conn = self._get_conn()
        try:
            keywords = [k.strip() for k in query.replace(",", " ").replace("，", " ").split() if k.strip()]
            if not keywords:
                return []

            # 构建 AND 条件：每个关键词都必须出现在 tags/title/summary 中
            where_clauses = []
            params = []
            for kw in keywords:
                like = f"%{kw}%"
                where_clauses.append("(tags LIKE ? OR title LIKE ? OR summary_markdown LIKE ?)")
                params.extend([like, like, like])

            sql = f"""SELECT id, video_id, title, author, tags,
                             source_url, created_at, duration_seconds, video_code, timestamp,
                             substr(summary_markdown, 1, 200) AS snippet
                      FROM knowledge
                      WHERE {' AND '.join(where_clauses)}
                      ORDER BY created_at DESC
                      LIMIT ?"""
            params.append(limit)

            rows = conn.execute(sql, params).fetchall()
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

    def video_code_exists(self, video_code: str) -> bool:
        """Check whether a public short code is already in use."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM knowledge WHERE video_code = ? LIMIT 1", (video_code,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def health_check(self) -> bool:
        """Perform a local, non-mutating database readiness check."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT 1 AS ok").fetchone()
            return bool(row and row["ok"] == 1)
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
