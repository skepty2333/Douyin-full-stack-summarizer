"""Classify a note (domain, temporality) and tag it against the vocabulary.

One flash-tier call replaces the free-form tag prompt. The model sees the
current vocabulary and must prefer canonical names; it may propose a new
name only when nothing fits, and it classifies each new name's ``kind`` so
that content types and vacuous words never become topics later.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.config import ALIYUN_TAG_MODEL
from app.database.knowledge_store import KnowledgeStore
from app.database.vocabulary import KINDS, VocabularyStore, clean_display_name
from app.services.aliyun_client import aliyun_client

logger = logging.getLogger(__name__)

DOMAINS = ("ai", "trading", "life", "other")
TEMPORALITIES = ("stable", "version_sensitive", "time_bound")
MAX_TAGS = 6
MAX_NOTE_CHARS = 9000
_VACUOUS_HINTS = ("技巧", "分享", "干货", "指南", "提升", "效率", "必看", "神器", "大全", "合集", "推荐", "信息差")

SYSTEM_PROMPT = """你是个人视频知识库的分类与标签助手。你会收到一条视频笔记和当前【词表】，只输出一个 JSON 对象：

{
  "domain": "ai" | "trading" | "life" | "other",
  "temporality": "stable" | "version_sensitive" | "time_bound",
  "tags": [{"name": "规范名或新名", "kind": "topic" | "entity" | "content_type"}]
}

domain 判定：
- ai：人工智能、大模型、Agent、编程与软件开发、AI 产品与工具
- trading：交易、投资、量化、理财、金融市场
- life：学习方法、健身健康、职业与求职、心理与生活方式
- other：以上都不是

temporality 判定（问自己"这条笔记的核心内容多久会失效"）：
- stable：原理、方法论、心理、数学、交易纪律等多年不变的内容
- version_sensitive：绑定具体产品、框架、模型版本或价格的用法与评测，几个月内可能变化
- time_bound：新闻、发布、榜单、行情、活动等事件性内容，很快过时

tags 规则（3 到 6 个，按重要性排序，最核心的主题放第一个）：
- 只给笔记的核心内容打标签——读者会为了这个主题来找这条笔记；顺带提到的工具、平台、人物一律不打。
- 优先从【词表】中选用规范名，名称必须与词表完全一致；别名也请写成规范名。
- 只有词表中确实没有合适条目时才新建，新名要短而通用（2 到 12 个字），用主题或实体的常用名，不带"技巧/指南/分享/干货"等修饰。
- kind：topic = 主题、概念、方法领域；entity = 具体产品、框架、模型、公司、人物、指标；content_type = 内容形式（教程、评测、新闻、访谈、面试题）。
- 不要输出泛化空话（效率提升、信息差、干货分享、自我提升之类）。
- 一个具体产品同时给出它所属的主题，例如 OpenClaw 同时给 Agent。

只输出 JSON，不要解释。"""


@dataclass(frozen=True)
class TagProposal:
    name: str
    kind: str


@dataclass(frozen=True)
class NoteClassification:
    domain: str
    temporality: str
    tags: tuple[TagProposal, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AppliedTags:
    entry_ids: tuple[int, ...]
    canonical_names: tuple[str, ...]
    created: tuple[str, ...]


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("分类结果不是 JSON 对象")
    return data


def parse_classification(raw: str) -> NoteClassification:
    """Validate the model's JSON; invalid pieces are dropped, not guessed."""
    data = _extract_json(raw)
    domain = str(data.get("domain", "")).strip().lower()
    if domain not in DOMAINS:
        raise ValueError(f"domain 无效: {domain!r}")
    # Flash models sometimes misspell the key ("temporability"); the value is
    # still usable, so accept any key that starts with "tempor".
    temporality_raw = next(
        (value for key, value in data.items() if str(key).lower().startswith("tempor")), ""
    )
    temporality = str(temporality_raw or "").strip().lower()
    if temporality not in TEMPORALITIES:
        raise ValueError(f"temporality 无效: {temporality!r}")
    proposals: list[TagProposal] = []
    seen: set[str] = set()
    for item in data.get("tags") or []:
        if isinstance(item, str):
            item = {"name": item, "kind": "topic"}
        if not isinstance(item, dict):
            continue
        name = clean_display_name(str(item.get("name", "")))
        kind = str(item.get("kind", "topic")).strip().lower()
        if kind not in KINDS:
            kind = "topic"
        if len(name) < 2 or name.lower() in seen:
            continue
        if any(hint in name for hint in _VACUOUS_HINTS) and kind != "entity":
            continue
        seen.add(name.lower())
        proposals.append(TagProposal(name=name, kind=kind))
        if len(proposals) >= MAX_TAGS:
            break
    if not proposals:
        raise ValueError("分类结果没有可用标签")
    return NoteClassification(domain=domain, temporality=temporality, tags=tuple(proposals))


def build_user_message(summary_markdown: str, title: str, author: str, vocabulary_listing: str) -> str:
    body = summary_markdown if len(summary_markdown) <= MAX_NOTE_CHARS else summary_markdown[:MAX_NOTE_CHARS] + "\n…（已截断）"
    listing = vocabulary_listing.strip() or "（词表为空，请按规则新建）"
    return (
        f"标题：{title}\n作者：{author}\n\n"
        f"【词表】\n{listing}\n\n"
        f"【笔记内容】\n{body}"
    )


async def classify_note(
    summary_markdown: str,
    title: str,
    author: str,
    *,
    vocabulary: VocabularyStore,
    model: str = ALIYUN_TAG_MODEL,
) -> Optional[NoteClassification]:
    """Return the classification, or None when the model call or parse fails."""
    listing = vocabulary.prompt_listing()
    try:
        raw = await aliyun_client.chat(
            model=model,
            operation="classify_note",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(summary_markdown, title, author, listing)},
            ],
            max_tokens=600,
            temperature=0.1,
            enable_thinking=False,
            response_format={"type": "json_object"},
        )
        return parse_classification(raw)
    except Exception:
        logger.exception("笔记分类失败，保留旧标签流程")
        return None


def resolve_tags(
    classification: NoteClassification,
    *,
    vocabulary: VocabularyStore,
    source: str = "ai",
) -> AppliedTags:
    """Map proposals to vocabulary entries, creating entries for genuinely new names."""
    entry_ids: list[int] = []
    names: list[str] = []
    created: list[str] = []
    for proposal in classification.tags:
        entry = vocabulary.resolve(proposal.name)
        if entry is None:
            entry = vocabulary.upsert_entry(proposal.name, kind=proposal.kind, source=source)
            created.append(entry.canonical)
        if entry.status != "active" or entry.id in entry_ids:
            continue
        entry_ids.append(entry.id)
        names.append(entry.canonical)
    return AppliedTags(entry_ids=tuple(entry_ids), canonical_names=tuple(names), created=tuple(created))


def link_note_tags(
    entry_id: int,
    applied: AppliedTags,
    *,
    vocabulary: VocabularyStore,
    source: str = "ai",
) -> int:
    """Attach resolved entries to a saved note."""
    return vocabulary.set_note_tags(entry_id, applied.entry_ids, source=source)


def apply_classification(
    entry_id: int,
    classification: NoteClassification,
    *,
    store: KnowledgeStore,
    vocabulary: VocabularyStore,
    source: str = "ai",
) -> AppliedTags:
    """Resolve, link, and update the note columns for an already-saved note."""
    applied = resolve_tags(classification, vocabulary=vocabulary, source=source)
    link_note_tags(entry_id, applied, vocabulary=vocabulary, source=source)
    store.update_metadata(
        entry_id,
        domain=classification.domain,
        temporality=classification.temporality,
        tags=",".join(applied.canonical_names),
    )
    return applied
