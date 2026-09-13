"""Section-level index and hybrid retrieval for knowledge notes.

Storage lives next to the notes in the same SQLite file and is entirely
derived: ``note_chunks`` can be rebuilt from ``knowledge`` at any time and
``chunk_embeddings`` is keyed by the hash of the embedded text, so unchanged
sections are never re-embedded and orphaned vectors are harmless.

Retrieval fuses two channels with reciprocal rank fusion:

* a semantic channel (cosine over all chunk vectors held in memory), and
* a keyword channel (verbatim, case-insensitive term matches in the chunk,
  its heading path, and the note's title and tags), which keeps product and
  tool names exact where embeddings are fuzzy.

Both channels degrade independently: without an embedder the keyword channel
still answers, and a chunk that has no vector yet is still keyword-searchable.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional, Sequence

import numpy as np

from app.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    KNOWLEDGE_DB_PATH,
)
from app.services.note_chunker import chunk_note, snippet

logger = logging.getLogger(__name__)

EmbedFn = Callable[[Sequence[str]], Awaitable[list[list[float]]]]
AliasGroupsFn = Callable[[], dict[str, tuple[str, ...]]]

RRF_K = 60
VECTOR_CANDIDATES = 200
# Below this cosine the semantic channel is returning nearest neighbours, not
# matches: on text-embedding-v4 unrelated sections of this corpus sit around
# 0.30-0.38 while on-topic ones start near 0.47. Keyword hits are unaffected.
MIN_SEMANTIC_COSINE = 0.40
NEAR_DUPLICATE_COSINE = 0.92
MAX_QUERY_CHARS = 200
_TERM_SPLIT_RE = re.compile(r"[\s,，、;；]+")


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: int
    knowledge_id: int
    chunk_index: int
    heading_path: str
    text: str
    embed_text_hash: str
    video_code: str
    title: str
    author: str
    tags: str
    created_at: str
    timestamp: str
    published_at: str = ""
    domain: str = ""
    temporality: str = ""


@dataclass(frozen=True)
class ChunkHit:
    chunk: ChunkRow
    score: float
    channels: tuple[str, ...]
    cosine: Optional[float] = None


@dataclass(frozen=True)
class NoteHit:
    knowledge_id: int
    video_code: str
    title: str
    author: str
    tags: str
    created_at: str
    timestamp: str
    score: float
    best_heading: str
    best_snippet: str
    matched_chunks: int
    channels: tuple[str, ...]
    best_cosine: Optional[float] = None
    published_at: str = ""
    domain: str = ""
    temporality: str = ""


@dataclass(frozen=True)
class SearchResult:
    notes: list[NoteHit]
    total_chunk_candidates: int
    semantic_available: bool


@dataclass(frozen=True)
class CollectedSection:
    chunk: ChunkRow
    score: float


@dataclass(frozen=True)
class CollectResult:
    sections: list[CollectedSection]
    total_chars: int
    note_count: int
    semantic_available: bool


@dataclass(frozen=True)
class IndexReport:
    notes_indexed: int
    chunks_written: int
    embeddings_pending: int


@dataclass
class _Cache:
    generation: tuple[int, int, int]
    rows: list[ChunkRow]
    matrix: Optional[np.ndarray]
    vector_rows: list[int] = field(default_factory=list)  # row index per matrix row
    row_to_vector: dict[int, int] = field(default_factory=dict)
    positions: dict[int, int] = field(default_factory=dict)  # chunk_id -> row index


def _normalize(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("向量无效")
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("零向量无法归一化")
    return array / norm


def split_terms(query: str) -> list[str]:
    """Whitespace / punctuation separated terms, de-duplicated, lower-cased."""
    seen: list[str] = []
    for term in _TERM_SPLIT_RE.split(query.strip()):
        term = term.strip().lower()
        if term and term not in seen:
            seen.append(term)
    return seen


class NoteIndex:
    """Derived chunk + vector index over ``knowledge`` with hybrid search."""

    def __init__(
        self,
        db_path: str = KNOWLEDGE_DB_PATH,
        *,
        embed_fn: Optional[EmbedFn] = None,
        alias_groups_fn: Optional[AliasGroupsFn] = None,
        model: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIMENSIONS,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        self.db_path = db_path
        self._embed_fn = embed_fn
        # Optional vocabulary hook: a query term that is a known alias also
        # matches its canonical name and sibling aliases (智能体 ↔ Agent).
        self._alias_groups_fn = alias_groups_fn
        self.model = model
        self.dimensions = int(dimensions)
        self.batch_size = max(1, int(batch_size))
        self._cache: Optional[_Cache] = None
        self._query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._init_db()

    # ------------------------------------------------------------------ setup
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
                CREATE TABLE IF NOT EXISTS note_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    knowledge_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    heading_path TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL,
                    embed_text TEXT NOT NULL,
                    embed_text_hash TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (knowledge_id) REFERENCES knowledge(id) ON DELETE CASCADE,
                    UNIQUE (knowledge_id, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS idx_note_chunks_hash ON note_chunks(embed_text_hash);

                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    embed_text_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (embed_text_hash, model, dimensions)
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    @property
    def semantic_available(self) -> bool:
        return self._embed_fn is not None

    # --------------------------------------------------------------- indexing
    def index_note(self, knowledge_id: int) -> IndexReport:
        """Re-chunk one note inside a single transaction; vectors come later."""
        conn = self._get_conn()
        try:
            note = conn.execute(
                "SELECT id, title, summary_markdown FROM knowledge WHERE id = ?",
                (knowledge_id,),
            ).fetchone()
            if note is None:
                raise KeyError(f"笔记 {knowledge_id} 不存在")
            chunks = chunk_note(note["title"], note["summary_markdown"])
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("DELETE FROM note_chunks WHERE knowledge_id = ?", (knowledge_id,))
            conn.executemany(
                """INSERT INTO note_chunks
                   (knowledge_id, chunk_index, heading_path, text, embed_text,
                    embed_text_hash, char_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        knowledge_id,
                        chunk.index,
                        chunk.heading_path,
                        chunk.text,
                        chunk.embed_text,
                        chunk.embed_text_hash,
                        chunk.char_count,
                        now,
                    )
                    for chunk in chunks
                ],
            )
            conn.commit()
            pending = self._count_pending(conn, knowledge_id)
            return IndexReport(notes_indexed=1, chunks_written=len(chunks), embeddings_pending=pending)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def index_all(self, *, only_missing: bool = True) -> IndexReport:
        """Chunk every note (or only notes without chunks yet)."""
        conn = self._get_conn()
        try:
            if only_missing:
                ids = [
                    int(row["id"])
                    for row in conn.execute(
                        """SELECT k.id FROM knowledge k
                           WHERE NOT EXISTS (SELECT 1 FROM note_chunks c WHERE c.knowledge_id = k.id)
                           ORDER BY k.id"""
                    )
                ]
            else:
                ids = [int(row["id"]) for row in conn.execute("SELECT id FROM knowledge ORDER BY id")]
        finally:
            conn.close()
        chunks_written = 0
        for knowledge_id in ids:
            chunks_written += self.index_note(knowledge_id).chunks_written
        return IndexReport(
            notes_indexed=len(ids),
            chunks_written=chunks_written,
            embeddings_pending=self.count_pending(),
        )

    def _count_pending(self, conn: sqlite3.Connection, knowledge_id: Optional[int] = None) -> int:
        sql = """SELECT COUNT(DISTINCT c.embed_text_hash) AS n FROM note_chunks c
                 WHERE NOT EXISTS (
                     SELECT 1 FROM chunk_embeddings e
                     WHERE e.embed_text_hash = c.embed_text_hash
                       AND e.model = ? AND e.dimensions = ?)"""
        params: list = [self.model, self.dimensions]
        if knowledge_id is not None:
            sql += " AND c.knowledge_id = ?"
            params.append(knowledge_id)
        return int(conn.execute(sql, params).fetchone()["n"])

    def count_pending(self) -> int:
        conn = self._get_conn()
        try:
            return self._count_pending(conn)
        finally:
            conn.close()

    def pending_texts(self, limit: int) -> list[tuple[str, str]]:
        """Distinct (hash, embed_text) pairs still lacking a vector."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT c.embed_text_hash AS h, MIN(c.embed_text) AS t FROM note_chunks c
                   WHERE NOT EXISTS (
                       SELECT 1 FROM chunk_embeddings e
                       WHERE e.embed_text_hash = c.embed_text_hash
                         AND e.model = ? AND e.dimensions = ?)
                   GROUP BY c.embed_text_hash
                   ORDER BY MIN(c.id)
                   LIMIT ?""",
                (self.model, self.dimensions, int(limit)),
            ).fetchall()
            return [(str(row["h"]), str(row["t"])) for row in rows]
        finally:
            conn.close()

    def store_embeddings(self, items: Sequence[tuple[str, Sequence[float]]]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        payload = []
        for digest, vector in items:
            normalized = _normalize(vector)
            if normalized.size != self.dimensions:
                raise ValueError(f"向量维度 {normalized.size} 与配置 {self.dimensions} 不符")
            payload.append((digest, self.model, self.dimensions, normalized.tobytes(), now))
        if not payload:
            return 0
        conn = self._get_conn()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO chunk_embeddings
                   (embed_text_hash, model, dimensions, vector, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                payload,
            )
            conn.commit()
            return len(payload)
        finally:
            conn.close()

    async def embed_pending(self, *, limit: Optional[int] = None) -> int:
        """Embed texts without vectors, in provider-sized batches; returns count."""
        if self._embed_fn is None:
            return 0
        done = 0
        remaining = limit
        while remaining is None or remaining > 0:
            take = self.batch_size if remaining is None else min(self.batch_size, remaining)
            batch = await asyncio.to_thread(self.pending_texts, take)
            if not batch:
                break
            vectors = await self._embed_fn([text for _, text in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("embedding 返回条数与请求不符")
            await asyncio.to_thread(
                self.store_embeddings,
                [(digest, vector) for (digest, _), vector in zip(batch, vectors)],
            )
            done += len(batch)
            if remaining is not None:
                remaining -= len(batch)
        return done

    async def index_note_and_embed(self, knowledge_id: int) -> IndexReport:
        report = await asyncio.to_thread(self.index_note, knowledge_id)
        embedded = await self.embed_pending()
        return IndexReport(
            notes_indexed=report.notes_indexed,
            chunks_written=report.chunks_written,
            embeddings_pending=max(0, report.embeddings_pending - embedded),
        )

    # -------------------------------------------------------------- loading
    def _generation(self, conn: sqlite3.Connection) -> tuple[int, int, int]:
        row = conn.execute(
            """SELECT (SELECT COUNT(*) FROM note_chunks) AS c,
                      (SELECT COALESCE(MAX(id), 0) FROM note_chunks) AS m,
                      (SELECT COUNT(*) FROM chunk_embeddings) AS e"""
        ).fetchone()
        return (int(row["c"]), int(row["m"]), int(row["e"]))

    def _load(self) -> _Cache:
        conn = self._get_conn()
        try:
            generation = self._generation(conn)
            cached = self._cache
            if cached is not None and cached.generation == generation:
                return cached
            rows = [
                ChunkRow(
                    chunk_id=int(r["id"]),
                    knowledge_id=int(r["knowledge_id"]),
                    chunk_index=int(r["chunk_index"]),
                    heading_path=str(r["heading_path"]),
                    text=str(r["text"]),
                    embed_text_hash=str(r["embed_text_hash"]),
                    video_code=str(r["video_code"] or ""),
                    title=str(r["title"] or ""),
                    author=str(r["author"] or ""),
                    tags=str(r["tags"] or ""),
                    created_at=str(r["created_at"] or ""),
                    timestamp=str(r["timestamp"] or ""),
                    published_at=str(r["published_at"] or ""),
                    domain=str(r["domain"] or ""),
                    temporality=str(r["temporality"] or ""),
                )
                for r in conn.execute(
                    """SELECT c.id, c.knowledge_id, c.chunk_index, c.heading_path, c.text,
                              c.embed_text_hash, k.video_code, k.title, k.author, k.tags,
                              k.created_at, k.timestamp, k.published_at, k.domain, k.temporality
                       FROM note_chunks c JOIN knowledge k ON k.id = c.knowledge_id
                       ORDER BY c.knowledge_id, c.chunk_index"""
                )
            ]
            vectors: dict[str, np.ndarray] = {}
            for r in conn.execute(
                "SELECT embed_text_hash, vector FROM chunk_embeddings WHERE model = ? AND dimensions = ?",
                (self.model, self.dimensions),
            ):
                array = np.frombuffer(bytes(r["vector"]), dtype=np.float32)
                if array.size == self.dimensions:
                    vectors[str(r["embed_text_hash"])] = array
        finally:
            conn.close()
        vector_rows = [i for i, row in enumerate(rows) if row.embed_text_hash in vectors]
        matrix = (
            np.vstack([vectors[rows[i].embed_text_hash] for i in vector_rows])
            if vector_rows
            else None
        )
        cache = _Cache(
            generation=generation,
            rows=rows,
            matrix=matrix,
            vector_rows=vector_rows,
            row_to_vector={row_index: pos for pos, row_index in enumerate(vector_rows)},
            positions={row.chunk_id: index for index, row in enumerate(rows)},
        )
        self._cache = cache
        return cache

    def invalidate(self) -> None:
        self._cache = None

    # ------------------------------------------------------------ retrieval
    async def _query_vector(self, query: str) -> Optional[np.ndarray]:
        if self._embed_fn is None:
            return None
        key = query.strip()
        cached = self._query_cache.get(key)
        if cached is not None:
            self._query_cache.move_to_end(key)
            return cached
        try:
            vectors = await self._embed_fn([key])
            vector = _normalize(vectors[0])
        except Exception as exc:
            logger.warning("查询向量化失败，降级为关键词检索: %s", type(exc).__name__)
            return None
        if vector.size != self.dimensions:
            logger.warning("查询向量维度不符，降级为关键词检索")
            return None
        self._query_cache[key] = vector
        if len(self._query_cache) > 64:
            self._query_cache.popitem(last=False)
        return vector

    def _expand_terms(self, terms: list[str]) -> dict[str, tuple[str, ...]]:
        """term -> every spelling to look for (itself plus its vocabulary group)."""
        groups: dict[str, tuple[str, ...]] = {}
        if self._alias_groups_fn is not None:
            try:
                groups = self._alias_groups_fn()
            except Exception:
                logger.exception("读取别名表失败，关键词通道不做同义扩展")
                groups = {}
        expanded: dict[str, tuple[str, ...]] = {}
        for term in terms:
            key = re.sub(r"\s+", "", term).lower()
            variants = [term] + [name.lower() for name in groups.get(key, ())]
            expanded[term] = tuple(dict.fromkeys(v for v in variants if v))
        return expanded

    def _keyword_scores(
        self, rows: list[ChunkRow], terms: list[str], *, require_all: bool
    ) -> dict[int, float]:
        """Row index -> keyword score; ``require_all`` filters at note level.

        Each term is weighted by its rarity (log-scaled inverse document
        frequency over sections), so a distinctive product name outranks a
        common word like 方法 that happens to appear in hundreds of sections.
        """
        if not terms:
            return {}
        haystacks = [f"{row.heading_path}\n{row.text}".lower() for row in rows]
        total = max(1, len(rows))
        variants = self._expand_terms(terms)
        weights = {}
        for term in terms:
            frequency = sum(
                1 for haystack in haystacks if any(v in haystack for v in variants[term])
            )
            weights[term] = math.log(1.0 + total / (1.0 + frequency))
        note_terms: dict[int, set[str]] = {}
        chunk_terms: dict[int, tuple[float, set[str]]] = {}
        for index, row in enumerate(rows):
            haystack = haystacks[index]
            # Title and tag matches describe the whole note; they are credited to
            # its opening section only, so body sections rank on their own text.
            note_level = row.chunk_index == 0
            title = row.title.lower() if note_level else ""
            tags = row.tags.lower() if note_level else ""
            score = 0.0
            found: set[str] = set()
            for term in terms:
                weight = weights[term]
                spellings = variants[term]
                hit = False
                if any(v in haystack for v in spellings):
                    score += weight
                    hit = True
                if title and any(v in title for v in spellings):
                    score += 0.5 * weight
                    hit = True
                if tags and any(v in tags for v in spellings):
                    score += 0.5 * weight
                    hit = True
                if hit:
                    found.add(term)
            if found:
                chunk_terms[index] = (score, found)
                note_terms.setdefault(row.knowledge_id, set()).update(found)
        if require_all:
            wanted = set(terms)
            allowed = {note for note, found in note_terms.items() if wanted <= found}
            return {
                index: score
                for index, (score, _) in chunk_terms.items()
                if rows[index].knowledge_id in allowed
            }
        return {index: score for index, (score, _) in chunk_terms.items()}

    async def _fused_chunks(
        self, query: str, *, require_all_terms: bool, domain: Optional[str] = None
    ) -> tuple[list[ChunkHit], _Cache, bool, Optional[np.ndarray]]:
        query = (query or "").strip()[:MAX_QUERY_CHARS]
        cache = await asyncio.to_thread(self._load)
        rows = cache.rows
        if not query or not rows:
            return [], cache, self.semantic_available, None
        domain = (domain or "").strip().lower() or None
        in_domain = (
            {index for index, row in enumerate(rows) if row.domain == domain}
            if domain is not None
            else None
        )

        terms = split_terms(query)
        keyword = self._keyword_scores(rows, terms, require_all=require_all_terms)
        if in_domain is not None:
            keyword = {index: score for index, score in keyword.items() if index in in_domain}
        keyword_rank = sorted(
            keyword.items(), key=lambda item: (-item[1], rows[item[0]].knowledge_id, rows[item[0]].chunk_index)
        )

        query_vector = await self._query_vector(query)
        semantic_ok = query_vector is not None and cache.matrix is not None
        vector_rank: list[tuple[int, float]] = []
        if semantic_ok:
            sims = cache.matrix @ query_vector
            top = min(VECTOR_CANDIDATES, sims.size)
            order = np.argpartition(-sims, top - 1)[:top]
            order = order[np.argsort(-sims[order])]
            allowed_rows = set(keyword) if require_all_terms else None
            for pos in order:
                cosine = float(sims[pos])
                if cosine < MIN_SEMANTIC_COSINE:
                    break  # sorted descending: everything after is weaker
                row_index = cache.vector_rows[int(pos)]
                if allowed_rows is not None and row_index not in allowed_rows:
                    continue
                if in_domain is not None and row_index not in in_domain:
                    continue
                vector_rank.append((row_index, cosine))

        fused: dict[int, float] = {}
        channels: dict[int, set[str]] = {}
        cosines: dict[int, float] = {}
        for rank, (row_index, _) in enumerate(keyword_rank):
            fused[row_index] = fused.get(row_index, 0.0) + 1.0 / (RRF_K + rank + 1)
            channels.setdefault(row_index, set()).add("关键词")
        for rank, (row_index, cosine) in enumerate(vector_rank):
            fused[row_index] = fused.get(row_index, 0.0) + 1.0 / (RRF_K + rank + 1)
            channels.setdefault(row_index, set()).add("语义")
            cosines[row_index] = cosine
        hits = [
            ChunkHit(
                chunk=rows[index],
                score=score,
                channels=tuple(sorted(channels[index])),
                cosine=cosines.get(index),
            )
            for index, score in sorted(fused.items(), key=lambda item: -item[1])
        ]
        return hits, cache, semantic_ok, query_vector

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        require_all_terms: bool = False,
        domain: Optional[str] = None,
    ) -> SearchResult:
        """Rank notes by their best chunks; each note appears once."""
        hits, _, semantic_ok, _ = await self._fused_chunks(
            query, require_all_terms=require_all_terms, domain=domain
        )
        grouped: "OrderedDict[int, list[ChunkHit]]" = OrderedDict()
        for hit in hits:
            grouped.setdefault(hit.chunk.knowledge_id, []).append(hit)
        # RRF scores are nearly flat across ranks, so any summed aggregate would let
        # a note with several mediocre sections outrank the single best section.
        # The best section decides; the number of matching sections only breaks ties.
        notes: list[NoteHit] = []
        for knowledge_id, chunk_hits in grouped.items():
            best = chunk_hits[0]
            channels = tuple(sorted({channel for hit in chunk_hits for channel in hit.channels}))
            notes.append(
                NoteHit(
                    knowledge_id=knowledge_id,
                    video_code=best.chunk.video_code,
                    title=best.chunk.title,
                    author=best.chunk.author,
                    tags=best.chunk.tags,
                    created_at=best.chunk.created_at,
                    timestamp=best.chunk.timestamp,
                    score=best.score,
                    best_heading=best.chunk.heading_path,
                    best_snippet=snippet(best.chunk.text),
                    matched_chunks=len(chunk_hits),
                    channels=channels,
                    best_cosine=best.cosine,
                    published_at=best.chunk.published_at,
                    domain=best.chunk.domain,
                    temporality=best.chunk.temporality,
                )
            )
        notes.sort(key=lambda note: (-note.score, -note.matched_chunks))
        return SearchResult(
            notes=notes[: max(1, limit)],
            total_chunk_candidates=len(hits),
            semantic_available=semantic_ok,
        )

    async def collect(
        self,
        query: str,
        *,
        max_chars: int = 12000,
        max_per_note: int = 3,
        max_sections: int = 40,
        require_all_terms: bool = False,
        domain: Optional[str] = None,
        note_ids: Optional[Sequence[int]] = None,
        exclude_note_ids: Optional[Sequence[int]] = None,
        min_cosine: Optional[float] = None,
        dedupe: bool = True,
    ) -> CollectResult:
        """Pick diverse, relevant sections up to a character budget.

        ``note_ids`` restricts to member notes (topic compilation), while
        ``exclude_note_ids`` with ``min_cosine`` finds strong outside matches.
        ``dedupe=False`` keeps near-identical sections from different notes,
        which a synthesis needs so every source stays citable.
        """
        hits, cache, semantic_ok, _ = await self._fused_chunks(
            query, require_all_terms=require_all_terms, domain=domain
        )
        allowed = {int(i) for i in note_ids} if note_ids is not None else None
        excluded = {int(i) for i in exclude_note_ids} if exclude_note_ids else set()
        selected: list[CollectedSection] = []
        selected_vectors: list[np.ndarray] = []
        per_note: dict[int, int] = {}
        total = 0
        for hit in hits:
            if len(selected) >= max_sections:
                break
            row = hit.chunk
            if allowed is not None and row.knowledge_id not in allowed:
                continue
            if row.knowledge_id in excluded:
                continue
            if min_cosine is not None and (hit.cosine is None or hit.cosine < min_cosine):
                continue
            if per_note.get(row.knowledge_id, 0) >= max_per_note:
                continue
            size = len(row.text)
            if selected and total + size > max_chars:
                continue
            vector = None
            if cache.matrix is not None:
                position = cache.row_to_vector.get(cache.positions[row.chunk_id])
                if position is not None:
                    vector = cache.matrix[position]
            if dedupe and vector is not None and selected_vectors:
                if max(float(vector @ other) for other in selected_vectors) >= NEAR_DUPLICATE_COSINE:
                    continue
            selected.append(CollectedSection(chunk=row, score=hit.score))
            if vector is not None:
                selected_vectors.append(vector)
            per_note[row.knowledge_id] = per_note.get(row.knowledge_id, 0) + 1
            total += size
            if total >= max_chars:
                break
        return CollectResult(
            sections=selected,
            total_chars=total,
            note_count=len(per_note),
            semantic_available=semantic_ok,
        )

    def note_chunks(self, knowledge_id: int) -> list[ChunkRow]:
        """All chunks of one note in document order (from the cached index)."""
        cache = self._load()
        return [row for row in cache.rows if row.knowledge_id == int(knowledge_id)]

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict:
        conn = self._get_conn()
        try:
            chunks = int(conn.execute("SELECT COUNT(*) AS n FROM note_chunks").fetchone()["n"])
            notes = int(
                conn.execute("SELECT COUNT(DISTINCT knowledge_id) AS n FROM note_chunks").fetchone()["n"]
            )
            vectors = int(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM chunk_embeddings WHERE model = ? AND dimensions = ?",
                    (self.model, self.dimensions),
                ).fetchone()["n"]
            )
            pending = self._count_pending(conn)
        finally:
            conn.close()
        return {
            "indexed_notes": notes,
            "chunks": chunks,
            "vectors": vectors,
            "pending_embeddings": pending,
            "model": self.model,
            "dimensions": self.dimensions,
        }
