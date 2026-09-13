"""Split one knowledge note's Markdown into section-level retrieval chunks.

Notes are compiled with a stable H1 / H2 / H3 structure, so headings are the
natural retrieval boundary: a query usually matches one section of a note, not
the whole note. Chunks keep the original Markdown (image references included)
for display, and expose a cleaned ``embed_text`` used for hashing and
embedding.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 40
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_EMPHASIS_RE = re.compile(r"(\*\*|__|`)")
_LIST_MARKER_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+", re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class NoteChunk:
    """One retrieval unit of a note."""

    index: int
    heading_path: str
    text: str
    embed_text: str
    embed_text_hash: str

    @property
    def char_count(self) -> int:
        return len(self.text)


def clean_for_embedding(markdown: str) -> str:
    """Reduce Markdown to plain prose so formatting does not dominate vectors."""
    text = _IMAGE_RE.sub(lambda m: f"[图: {m.group(1).strip()}]" if m.group(1).strip() else "", markdown)
    text = _LINK_RE.sub(r"\1", text)
    text = _EMPHASIS_RE.sub("", text)
    text = _LIST_MARKER_RE.sub("", text)
    text = re.sub(r"^[ \t]*>[ \t]?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\|?[-:| ]+\|?[ \t]*$", "", text, flags=re.MULTILINE)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


def _normalize_block(lines: list[str]) -> str:
    text = "\n".join(line.rstrip() for line in lines)
    return _BLANK_RUN_RE.sub("\n\n", text).strip()


def _split_long(text: str, limit: int) -> list[str]:
    """Split on paragraph, then line, then hard boundaries; never exceed limit."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > limit:
            if current:
                pieces.append(current)
                current = ""
            lines = paragraph.split("\n")
            buffer = ""
            for line in lines:
                if len(line) > limit:
                    if buffer:
                        pieces.append(buffer)
                        buffer = ""
                    pieces.extend(line[i : i + limit] for i in range(0, len(line), limit))
                    continue
                candidate = f"{buffer}\n{line}" if buffer else line
                if len(candidate) > limit:
                    pieces.append(buffer)
                    buffer = line
                else:
                    buffer = candidate
            if buffer:
                pieces.append(buffer)
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > limit:
            pieces.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece.strip()]


def _sections(markdown: str) -> tuple[str, list[tuple[tuple[str, ...], list[str]]]]:
    """Walk the Markdown once, returning the H1 title and (heading path, lines)."""
    title = ""
    sections: list[tuple[tuple[str, ...], list[str]]] = [((), [])]
    h2 = ""
    in_fence = False
    for raw_line in markdown.splitlines():
        if _FENCE_RE.match(raw_line):
            in_fence = not in_fence
            sections[-1][1].append(raw_line)
            continue
        match = None if in_fence else _HEADING_RE.match(raw_line)
        if match is None:
            sections[-1][1].append(raw_line)
            continue
        level = len(match.group(1))
        heading = match.group(2).strip()
        if level == 1:
            if not title:
                title = heading
                continue
            # A second H1 behaves like a top-level section.
            level = 2
        if level == 2:
            h2 = heading
            sections.append(((heading,), []))
        elif level == 3:
            path = (h2, heading) if h2 else (heading,)
            sections.append((path, []))
        else:
            sections[-1][1].append(raw_line)
    return title, sections


def chunk_note(title: str, markdown: str, *, max_chars: int = MAX_CHUNK_CHARS) -> list[NoteChunk]:
    """Return ordered chunks; tiny sections merge backwards, long ones split."""
    h1_title, sections = _sections(markdown or "")
    display_title = (h1_title or title or "").strip()
    blocks: list[tuple[str, str]] = []
    for path, lines in sections:
        body = _normalize_block(lines)
        if not body:
            continue
        heading_path = " > ".join(part for part in path if part)
        for piece in _split_long(body, max_chars):
            # Only stray fragments merge backwards; a short but headed section is
            # still its own unit, otherwise its heading would disappear.
            if (
                blocks
                and len(clean_for_embedding(piece)) < MIN_CHUNK_CHARS
                and heading_path in ("", blocks[-1][0])
                and len(blocks[-1][1]) + len(piece) + 2 <= max_chars
            ):
                prev_path, prev_text = blocks[-1]
                blocks[-1] = (prev_path, f"{prev_text}\n\n{piece}")
                continue
            blocks.append((heading_path, piece))

    chunks: list[NoteChunk] = []
    for index, (heading_path, text) in enumerate(blocks):
        cleaned = clean_for_embedding(text)
        header = "\n".join(part for part in (display_title, heading_path) if part)
        embed_text = f"{header}\n{cleaned}" if header else cleaned
        digest = hashlib.sha256(embed_text.encode("utf-8")).hexdigest()
        chunks.append(
            NoteChunk(
                index=index,
                heading_path=heading_path,
                text=text,
                embed_text=embed_text,
                embed_text_hash=digest,
            )
        )
    return chunks


def snippet(text: str, limit: int = 120) -> str:
    """One-line preview of a chunk for compact search results."""
    cleaned = clean_for_embedding(text).replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"
