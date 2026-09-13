"""Compile topic pages from note sections, and split topics that grow too big.

The compiler never sees whole notes: it gets the sections most relevant to
the topic (member notes first, plus strong matches from outside the member
set as a safety net for tagging misses), the previous page when one exists,
and the notes' publish dates and temporality. Every conclusion must cite the
video codes it came from; a conclusion the material does not support has to
go to the 待验证 section. The source list is appended by code, not written
by the model, so provenance cannot drift.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Sequence

from app.config import TOPIC_COMPILE_MODEL, TOPIC_RECOMPILE_DIRTY
from app.database.knowledge_store import KnowledgeStore
from app.database.note_index import CollectedSection, NoteIndex
from app.database.topics import Topic, TopicStore, TopicVersion
from app.database.vocabulary import VocabEntry
from app.services.aliyun_client import aliyun_client

logger = logging.getLogger(__name__)

ChatFn = Callable[..., Awaitable[str]]

MEMBER_SECTION_BUDGET = 26000
OUTSIDE_SECTION_BUDGET = 5000
OUTSIDE_MIN_COSINE = 0.62
_CODE_RE = re.compile(r"`?([a-z0-9]{5})`?")
TEMPORALITY_LABELS = {
    "stable": "稳定",
    "version_sensitive": "版本敏感",
    "time_bound": "时效性",
}

COMPILE_SYSTEM_PROMPT = """你是个人视频知识库的主题页编辑。给定一个主题、若干条笔记的相关章节（每段前标有视频码、作者、发布日期、时效类型），以及可能存在的上一版页面，写出这个主题的当前知识页。

硬性规则：
1. 每一条结论、做法、数据后面必须用方括号标注来源视频码，如 [k4e56] 或 [k4e56, gj8mn]；材料里没有依据的内容只能写进"待验证"一节，不得混入正文。
2. 不同来源说法不一致时，不要替读者选边：在"不同来源的分歧"里并列两种说法、各自的适用条件和来源。
3. 标注为"版本敏感"或"时效性"的来源，其结论要带上发布日期（如"截至 2026-03"），并在"可能已过时"一节提示。
4. 若给了上一版页面：保留其中没有被新材料反驳的结论；被反驳的改写并说明依据；不要因为新材料没提到就删除旧结论。
5. 观点、预测、个人经验要带归属（"某作者认为"），不写成公认事实。
6. 语言精炼，用作者的信息，不加自己的常识扩展；不写"本文""视频中"之类的元话语。

输出 Markdown，固定结构（没有内容的小节写"暂无"）：

# {主题名}

> 一句话定义（一到两句，说明这个主题在本库语境下指什么）

## 核心结论与主流做法
（分条，每条一到三句，带视频码）

## 不同来源的分歧
## 可能已过时
## 待验证

不要输出"来源笔记"一节，系统会自动附上。只输出 Markdown。"""

SPLIT_SYSTEM_PROMPT = """你是个人视频知识库的主题编辑。一个主题的成员笔记已经多到无法在一页里综述，需要拆成 2 到 5 个子主题。你会收到主题名、成员笔记清单（视频码、标题、当前标签），以及库里已有的其他主题名。

规则：
1. 优先复用【已有主题】作为子主题（名称必须完全一致），只有确实没有合适的才新建；新名短而通用（2 到 12 字）。
2. 每个子主题至少 4 条笔记；一条笔记可以属于多个子主题；实在不属于任何子主题的笔记留空即可。
3. 子主题之间应当是这个主题下不同的方面或对象，而不是按时间或作者划分。

只输出 JSON：{"children": [{"name": "子主题名", "existing": true, "codes": ["k4e56", "gj8mn"]}]}"""


@dataclass(frozen=True)
class CompileResult:
    topic: Topic
    version: TopicVersion
    member_notes: int
    outside_notes: int
    uncited_lines: int


@dataclass(frozen=True)
class SplitResult:
    parent: Topic
    children: tuple[Topic, ...]
    created_entries: tuple[str, ...]


def _date(note: dict) -> str:
    published = note.get("published_at") or ""
    if published:
        return f"发布 {published[:10]}"
    stamp = note.get("timestamp") or note.get("created_at") or ""
    return f"入库 {stamp[:10]}"


def _note_header(note: dict) -> str:
    badge = TEMPORALITY_LABELS.get(note.get("temporality") or "", "未标注")
    return f"[{note['video_code']}] {note['title'][:60]} — {note['author']} · {_date(note)} · {badge}"


def build_material(
    sections: Sequence[CollectedSection], notes: dict[int, dict]
) -> tuple[str, list[int]]:
    """Group selected sections by note, in the order notes first appear."""
    lines: list[str] = []
    order: list[int] = []
    grouped: dict[int, list[CollectedSection]] = {}
    for section in sections:
        knowledge_id = section.chunk.knowledge_id
        if knowledge_id not in grouped:
            order.append(knowledge_id)
        grouped.setdefault(knowledge_id, []).append(section)
    for knowledge_id in order:
        note = notes.get(knowledge_id)
        if note is None:
            continue
        lines.append(f"### {_note_header(note)}")
        # Document order within a note reads better than selection order.
        for section in sorted(grouped[knowledge_id], key=lambda item: item.chunk.chunk_index):
            heading = section.chunk.heading_path or "概述"
            lines.append(f"#### {heading}\n{section.chunk.text}")
        lines.append("")
    return "\n".join(lines), order


def source_list(notes: Sequence[dict]) -> str:
    ordered = sorted(notes, key=lambda n: (n.get("published_at") or n.get("created_at") or ""), reverse=True)
    lines = ["## 来源笔记", ""]
    for note in ordered:
        badge = TEMPORALITY_LABELS.get(note.get("temporality") or "", "")
        lines.append(
            f"- `{note['video_code']}` {note['title'][:60]} — {note['author']} · {_date(note)}"
            + (f" · {badge}" if badge else "")
        )
    return "\n".join(lines)


_LIST_ITEM_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.*)$")
_LABEL_ONLY_RE = re.compile(r"^\*\*[^*]+\*\*[:：]?$")


def count_uncited_lines(markdown: str, known_codes: set[str]) -> int:
    """List items in the conclusions section that cite no known video code.

    Bold-only lines (group labels such as ``**一、…**``) are not claims.
    """
    section = re.split(r"^## ", markdown, flags=re.MULTILINE)
    body = next((part for part in section if part.startswith("核心结论")), "")
    uncited = 0
    for line in body.splitlines():
        match = _LIST_ITEM_RE.match(line.strip())
        if match is None:
            continue
        content = match.group(1).strip()
        if _LABEL_ONLY_RE.match(content):
            continue
        cited = {m.group(1) for m in _CODE_RE.finditer(content)}
        if not (cited & known_codes):
            uncited += 1
    return uncited


class TopicCompiler:
    def __init__(
        self,
        *,
        store: KnowledgeStore,
        index: NoteIndex,
        topics: TopicStore,
        chat_fn: Optional[ChatFn] = None,
        model: str = TOPIC_COMPILE_MODEL,
    ):
        self.store = store
        self.index = index
        self.topics = topics
        self.model = model
        self._chat = chat_fn or self._default_chat

    async def _default_chat(self, *, system: str, user: str, max_tokens: int, operation: str, thinking_budget: Optional[int]) -> str:
        return await aliyun_client.chat(
            model=self.model,
            operation=operation,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.2,
            enable_thinking=thinking_budget is not None,
            thinking_budget=thinking_budget,
        )

    def _topic_query(self, topic: Topic) -> str:
        return " ".join(dict.fromkeys((topic.name, *topic.aliases[:4])))

    async def gather(self, topic: Topic) -> tuple[list[CollectedSection], list[CollectedSection], dict[int, dict]]:
        member_ids = self.topics.member_note_ids(topic)
        query = self._topic_query(topic)
        members = await self.index.collect(
            query,
            max_chars=MEMBER_SECTION_BUDGET,
            max_per_note=3,
            max_sections=120,
            note_ids=member_ids,
            dedupe=False,
        )
        outside = await self.index.collect(
            query,
            max_chars=OUTSIDE_SECTION_BUDGET,
            max_per_note=1,
            max_sections=12,
            exclude_note_ids=member_ids,
            min_cosine=OUTSIDE_MIN_COSINE,
        )
        member_sections = list(members.sections)
        # Relevance ordering can starve a member note entirely; every member
        # contributes at least its opening section so the page sees all sources.
        covered = {section.chunk.knowledge_id for section in member_sections}
        for knowledge_id in member_ids:
            if knowledge_id in covered:
                continue
            chunks = await asyncio.to_thread(self.index.note_chunks, knowledge_id)
            if chunks:
                member_sections.append(CollectedSection(chunk=chunks[0], score=0.0))
        notes: dict[int, dict] = {}
        for section in (*member_sections, *outside.sections):
            knowledge_id = section.chunk.knowledge_id
            if knowledge_id not in notes:
                note = self.store.get_by_id(knowledge_id)
                if note:
                    notes[knowledge_id] = note
        return member_sections, list(outside.sections), notes

    async def compile(self, topic: Topic, *, activate: bool = True) -> CompileResult:
        if topic.is_hub:
            return await self.compile_hub(topic)
        member_sections, outside_sections, notes = await self.gather(topic)
        if not member_sections:
            raise ValueError(f"主题 {topic.name} 没有可用的成员章节")
        material, member_order = build_material(member_sections, notes)
        outside_material, outside_order = build_material(outside_sections, notes)
        previous = self.topics.version(topic.id)
        parts = [f"主题：{topic.name}"]
        if topic.aliases:
            parts.append(f"别名：{' / '.join(topic.aliases)}")
        parts.append(f"\n【成员笔记的相关章节】\n{material}")
        if outside_material.strip():
            parts.append(f"\n【未打此标签但高度相关的章节（可引用）】\n{outside_material}")
        if previous is not None:
            parts.append(f"\n【上一版页面（{previous.valid_as_of}）】\n{previous.markdown}")
        user = "\n".join(parts)
        markdown = await self._chat(
            system=COMPILE_SYSTEM_PROMPT,
            user=user,
            max_tokens=6000,
            operation="topic_compile",
            thinking_budget=4096,
        )
        markdown = markdown.strip()
        if not markdown.startswith("#"):
            raise ValueError("主题页输出不是 Markdown")
        # Strip a model-written source list, then append the authoritative one.
        markdown = re.split(r"^## 来源笔记\s*$", markdown, flags=re.MULTILINE)[0].rstrip()
        used_ids = [*member_order, *outside_order]
        used_notes = [notes[i] for i in used_ids if i in notes]
        codes = {note["video_code"] for note in used_notes}
        uncited = count_uncited_lines(markdown, codes)
        page = f"{markdown}\n\n{source_list(used_notes)}\n"
        version = self.topics.save_version(
            topic.id,
            page,
            compiled_by=self.model,
            source_note_ids=used_ids,
            input_chars=len(user),
            activate=activate,
        )
        refreshed = self.topics.get(topic.id) or topic
        if uncited:
            logger.warning("主题页 %s v%s 有 %s 条结论未引用来源", topic.name, version.version, uncited)
        return CompileResult(
            topic=refreshed,
            version=version,
            member_notes=len(member_order),
            outside_notes=len(outside_order),
            uncited_lines=uncited,
        )

    async def compile_hub(self, topic: Topic) -> CompileResult:
        """A hub page is an index of its children plus their one-line definitions."""
        children = self.topics.children(topic.id)
        lines = [f"# {topic.name}", "", f"> 枢纽页：本主题共 {topic.note_count} 条笔记，已按方面拆为 {len(children)} 个子主题。", ""]
        lines.append("## 子主题")
        lines.append("")
        for child in sorted(children, key=lambda c: -c.note_count):
            latest = self.topics.version(child.id)
            definition = ""
            if latest is not None:
                quote = re.search(r"^>\s*(.+)$", latest.markdown, flags=re.MULTILINE)
                definition = f" — {quote.group(1).strip()}" if quote else ""
            lines.append(f"- **{child.name}**（{child.note_count} 条 / {child.author_count} 位作者）{definition}")
        member_ids = self.topics.member_note_ids(topic)
        notes = [n for n in (self.store.get_by_id(i) for i in member_ids) if n]
        page = "\n".join(lines) + "\n\n" + source_list(notes) + "\n"
        version = self.topics.save_version(
            topic.id, page, compiled_by="hub", source_note_ids=member_ids, input_chars=0
        )
        refreshed = self.topics.get(topic.id) or topic
        return CompileResult(topic=refreshed, version=version, member_notes=len(member_ids), outside_notes=0, uncited_lines=0)

    async def split(self, topic: Topic) -> SplitResult:
        """Split an oversized topic into children (reusing existing topics where they fit)."""
        member_ids = self.topics.member_note_ids(topic)
        notes = [n for n in (self.store.get_by_id(i) for i in member_ids) if n]
        existing = [
            t for t in self.topics.list_topics(statuses=("candidate", "active"))
            if t.id != topic.id and not t.is_hub
        ]
        existing_names = sorted({t.name for t in existing})
        listing = "\n".join(
            f"- {n['video_code']}\t{n['title'][:50]}\t标签：{n['tags'][:60]}" for n in notes
        )
        user = (
            f"主题：{topic.name}（{len(notes)} 条笔记）\n\n【已有主题】\n"
            + ("\n".join(f"- {name}" for name in existing_names) or "（无）")
            + f"\n\n【成员笔记】\n视频码\t标题\t标签\n{listing}"
        )
        raw = await self._chat(
            system=SPLIT_SYSTEM_PROMPT,
            user=user,
            max_tokens=3000,
            operation="topic_split",
            thinking_budget=2048,
        )
        data = _extract_json(raw)
        by_code = {n["video_code"]: int(n["id"]) for n in notes}
        created: list[str] = []
        children: list[Topic] = []
        for item in data.get("children") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            codes = [c for c in (item.get("codes") or []) if c in by_code]
            if len(name) < 2 or len(codes) < 4:
                continue
            entry = self.topics.vocabulary.resolve(name)
            if entry is None:
                entry = self.topics.vocabulary.upsert_entry(name, kind="topic", source="ai")
                created.append(entry.canonical)
            if entry.id == topic.vocabulary_id:
                continue
            for code in codes:
                knowledge_id = by_code[code]
                current = [e.id for e in self.topics.vocabulary.note_entries(knowledge_id)]
                if entry.id not in current:
                    self.topics.vocabulary.set_note_tags(knowledge_id, [*current, entry.id], source="ai")
            entry = self.topics.vocabulary.get(entry.id) or entry
            child = self.topics.get_by_vocabulary(entry.id)
            if child is None:
                child = self.topics.born(entry, reason=f"由 {topic.name} 拆分", parent_id=topic.id)
            elif child.parent_id is None:
                self.topics.set_parent(child.id, topic.id)
                child = self.topics.get(child.id) or child
            children.append(child)
        if len(children) < 2:
            raise ValueError(f"主题 {topic.name} 的拆分提案不足两个子主题")
        self.topics.mark_hub(topic.id)
        parent = self.topics.get(topic.id) or topic
        return SplitResult(parent=parent, children=tuple(children), created_entries=tuple(created))

    async def grow(self, *, compile_new: bool = True, recompile_dirty: bool = True) -> dict:
        """Birth eligible entries, split oversized topics, compile what is due."""
        report: dict = {"born": [], "split": [], "compiled": [], "failed": []}
        for entry in self.topics.eligible_entries():
            topic = self.topics.born(entry)
            report["born"].append(topic.name)
        for topic in self.topics.list_topics():
            if self.topics.needs_split(topic):
                try:
                    result = await self.split(topic)
                    report["split"].append((topic.name, [child.name for child in result.children]))
                except Exception as exc:
                    logger.exception("主题拆分失败: %s", topic.name)
                    report["failed"].append((topic.name, f"split: {type(exc).__name__}"))
        compiled: set[str] = set()
        for topic in self.topics.list_topics():
            if topic.is_hub:
                continue
            due = (compile_new and topic.latest_version == 0) or (
                recompile_dirty and topic.latest_version > 0 and topic.dirty >= TOPIC_RECOMPILE_DIRTY
            )
            if not due:
                continue
            try:
                result = await self.compile(topic)
                report["compiled"].append((topic.name, result.version.version))
                compiled.add(topic.name)
            except Exception as exc:
                logger.exception("主题编译失败: %s", topic.name)
                report["failed"].append((topic.name, f"compile: {type(exc).__name__}"))
        # Hub pages are cheap and depend on their children's definitions.
        for topic in self.topics.list_topics():
            if not topic.is_hub:
                continue
            children = self.topics.children(topic.id)
            if topic.latest_version == 0 or any(child.name in compiled for child in children):
                try:
                    result = await self.compile_hub(topic)
                    report["compiled"].append((topic.name, result.version.version))
                except Exception as exc:
                    logger.exception("枢纽页刷新失败: %s", topic.name)
                    report["failed"].append((topic.name, f"hub: {type(exc).__name__}"))
        return report


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("拆分结果不是 JSON 对象")
    return data
