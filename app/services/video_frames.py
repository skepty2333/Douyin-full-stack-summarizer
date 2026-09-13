"""Safe, best-effort extraction and embedding of evidence frames from videos.

The vision model is allowed to propose time-aligned annotations, but it is not
trusted to choose a readable video frame or a filesystem path.  This module
validates the proposal, samples a few nearby timestamps with ffmpeg, selects
the most readable candidate with Pillow, and exposes only fixed-name JPEGs.

All public helpers degrade to an empty result when media processing fails.  A
missing screenshot must never prevent the text-only knowledge note from being
generated.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import itertools
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Iterable, Mapping, Optional, Sequence, Union

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError


logger = logging.getLogger(__name__)

FRAME_ID_RE = re.compile(r"^V(?:0{3}[1-9]|0{2}[1-9]\d|0[1-9]\d{2}|[1-9]\d{3})$")
FRAME_MARKER_RE = re.compile(r"\[\[VIDEO_FRAME:(V\d{4})\]\]")
ANY_FRAME_MARKER_RE = re.compile(r"\[\[VIDEO_FRAME:[^\]\r\n]{0,80}\]\]")
VIDEO_CODE_RE = re.compile(r"^[A-Za-z0-9]{1,32}$")
PERSISTED_ASSET_PATH_RE = re.compile(
    r"^blobs/[0-9a-f]{2}/[0-9a-f]{64}\.jpg$"
)

ALLOWED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm"})
ALLOWED_KINDS = frozenset({"ocr", "ui_structure", "state_change", "visible_fact"})
ALLOWED_CONFIDENCE = frozenset({"high", "medium"})

MAX_VIDEO_BYTES = 1024 * 1024 * 1024
MAX_VIDEO_DURATION_MS = 6 * 60 * 60 * 1000
MAX_ANNOTATIONS = 80
DEFAULT_MAX_FRAMES = 6
HARD_MAX_FRAMES = 8
MAX_ANNOTATION_SPAN_MS = 2 * 60 * 1000
MAX_ANNOTATION_TEXT_CHARS = 1000
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_TOTAL_EMBEDDED_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_FRAME_DIMENSION = 1600
FFPROBE_TIMEOUT_SECONDS = 15
FFMPEG_TIMEOUT_SECONDS = 25
PORTRAIT_CROP_TRIGGER_RATIO = 1.35
FRAME_HASH_SIZE = 16
FRAME_DEDUP_MAX_GAP_MS = 30_000
FRAME_DEDUP_MAX_HASH_DISTANCE = 8

_GENERIC_VISUAL_TEXT = frozenset(
    {
        "画面",
        "画面内容",
        "视频画面",
        "人物",
        "人物出镜",
        "说话人",
        "字幕",
        "标题",
        "界面",
        "无法判断",
        "不确定",
        "unknown",
    }
)


@dataclass(frozen=True)
class VisualAnnotation:
    """A validated, body-worthy visual fact proposed by the vision stage."""

    id: str
    start_ms: int
    end_ms: int
    kind: str
    confidence: str
    text: str

    @property
    def center_ms(self) -> int:
        return self.start_ms + (self.end_ms - self.start_ms) // 2


@dataclass(frozen=True)
class ExtractedVideoFrame:
    """A locally verified screenshot that may be embedded in the final PDF."""

    annotation_id: str
    timestamp_ms: int
    path: Path
    kind: str
    confidence: str
    caption: str
    quality_score: float


@dataclass(frozen=True)
class PersistedVideoFrame:
    """A content-addressed screenshot referenced by canonical Markdown."""

    annotation_id: str
    timestamp_ms: int
    relative_path: str
    mime_type: str
    kind: str
    confidence: str
    caption: str
    quality_score: float
    width: int
    height: int
    byte_size: int
    sha256: str
    display_order: int


RawAnnotation = Union[VisualAnnotation, Mapping[str, object]]


def select_frame_annotations(
    annotations: Iterable[RawAnnotation],
    *,
    duration_ms: int,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[VisualAnnotation]:
    """Validate and rank annotations that justify editor context frames.

    OCR, UI structure, and state changes are accepted at high or medium confidence.
    ``visible_fact`` is deliberately stricter: only high-confidence facts are
    eligible because a generic visual fact rarely justifies a full-width image.
    Invalid entries are ignored rather than raising.
    """

    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
        return []
    if duration_ms <= 0 or duration_ms > MAX_VIDEO_DURATION_MS:
        return []
    safe_limit = _safe_frame_limit(max_frames)
    if safe_limit == 0:
        return []

    validated: list[VisualAnnotation] = []
    seen_ids: set[str] = set()
    try:
        bounded = itertools.islice(iter(annotations), MAX_ANNOTATIONS)
    except (TypeError, ValueError):
        return []

    for raw in bounded:
        item = _validate_annotation(raw, duration_ms=duration_ms)
        if item is None or item.id in seen_ids:
            continue
        seen_ids.add(item.id)
        validated.append(item)

    ranked = sorted(validated, key=_annotation_rank, reverse=True)
    # Selection is importance-first; the returned order is chronological so the
    # downstream prompt and generated article remain easy to follow.
    return sorted(ranked[:safe_limit], key=lambda item: (item.center_ms, item.id))


def extract_video_frames(
    video_path: Union[str, os.PathLike[str]],
    annotations: Iterable[RawAnnotation],
    *,
    output_dir: Optional[Union[str, os.PathLike[str]]] = None,
    allowed_video_root: Optional[Union[str, os.PathLike[str]]] = None,
    allowed_output_root: Optional[Union[str, os.PathLike[str]]] = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_video_bytes: int = MAX_VIDEO_BYTES,
) -> dict[str, ExtractedVideoFrame]:
    """Extract the best local JPEG for each eligible visual annotation.

    Paths must be absolute.  By default screenshots are written only under the
    video's parent directory.  Callers using a separate job output directory
    must explicitly provide ``allowed_output_root``.  The returned mapping is
    empty on any video-level failure; individual frame failures are skipped.
    """

    try:
        source = _validate_video_path(
            video_path,
            allowed_root=allowed_video_root,
            max_video_bytes=max_video_bytes,
        )
        duration_ms = _probe_video_duration_ms(source)
        selected = select_frame_annotations(
            annotations,
            duration_ms=duration_ms,
            max_frames=max_frames,
        )
        if not selected:
            return {}

        destination = _prepare_output_dir(
            source,
            output_dir=output_dir,
            allowed_output_root=allowed_output_root,
        )
    except Exception as exc:
        logger.warning("视频截图准备失败，继续生成纯文本笔记: %s", exc)
        return {}

    extracted: dict[str, ExtractedVideoFrame] = {}
    for annotation in selected:
        try:
            frame = _extract_best_frame(
                source,
                annotation,
                destination,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            logger.warning("视觉标注 %s 抽帧失败，已跳过: %s", annotation.id, exc)
            frame = None
        if frame is not None:
            extracted[annotation.id] = frame
    deduplicated = _deduplicate_similar_frames(extracted)
    if len(deduplicated) < len(extracted):
        logger.info(
            "近重复视频截图已合并: before=%s after=%s",
            len(extracted),
            len(deduplicated),
        )
    return deduplicated


def render_video_frame_markers_for_pdf(
    markdown_text: str,
    frames: Union[Mapping[str, ExtractedVideoFrame], Sequence[ExtractedVideoFrame]],
    *,
    max_total_embedded_bytes: int = MAX_TOTAL_EMBEDDED_BYTES,
) -> str:
    """Replace frame markers with self-contained, PDF-safe JPEG HTML.

    A marker is replaced at most once, preventing a model from repeating one
    image until the rendered document becomes excessively large.  Missing or
    invalid frames simply remove their marker.
    """

    if not isinstance(markdown_text, str) or not markdown_text:
        return markdown_text if isinstance(markdown_text, str) else ""
    frame_map = _normalise_frame_mapping(frames)
    byte_budget = _safe_embedded_byte_budget(max_total_embedded_bytes)
    used_ids: set[str] = set()
    embedded_bytes = 0

    def replace_marker(match: re.Match[str]) -> str:
        nonlocal embedded_bytes
        annotation_id = match.group(1)
        if annotation_id in used_ids:
            return ""
        used_ids.add(annotation_id)
        frame = frame_map.get(annotation_id)
        if frame is None:
            return ""
        try:
            image_bytes = _read_verified_frame(frame)
        except Exception as exc:
            logger.warning("PDF 截图 %s 校验失败，已移除标记: %s", annotation_id, exc)
            return ""
        if embedded_bytes + len(image_bytes) > byte_budget:
            logger.warning("PDF 截图达到总大小上限，已跳过 %s", annotation_id)
            return ""
        embedded_bytes += len(image_bytes)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        caption = html.escape(frame.caption.strip()[:MAX_ANNOTATION_TEXT_CHARS])
        timestamp = _format_timestamp(frame.timestamp_ms)
        alt = html.escape(f"视频截图 {timestamp}：{frame.caption.strip()[:240]}", quote=True)
        return (
            "\n\n<figure class=\"video-evidence-frame\" "
            "style=\"margin:12px auto 16px;text-align:center;page-break-inside:avoid;\">"
            f"<img src=\"data:image/jpeg;base64,{encoded}\" alt=\"{alt}\" "
            "style=\"display:block;max-width:100%;height:auto;margin:0 auto;\"/>"
            f"<figcaption style=\"margin-top:5px;color:#666;font-size:0.86em;\">"
            f"视频画面 {timestamp} · {caption}</figcaption></figure>\n\n"
        )

    rendered = FRAME_MARKER_RE.sub(replace_marker, markdown_text)
    # Never expose malformed model-generated markers in the PDF.
    return ANY_FRAME_MARKER_RE.sub("", rendered)


def persist_video_frame_markers_for_storage(
    markdown_text: str,
    frames: Union[Mapping[str, ExtractedVideoFrame], Sequence[ExtractedVideoFrame]],
    *,
    video_code: str,
    asset_root: Union[str, os.PathLike[str]],
) -> tuple[str, tuple[PersistedVideoFrame, ...]]:
    """Build canonical multimodal Markdown and persist only referenced frames.

    Text remains the primary representation: every image reference contains an
    objective timestamped caption.  JPEG bytes live in a content-addressed
    directory outside the temporary job tree and are fetched only when a
    multimodal client asks for them.  Any invalid or failed image is removed
    from the Markdown without preventing the note from being stored.
    """

    if not isinstance(markdown_text, str) or not markdown_text:
        return (markdown_text if isinstance(markdown_text, str) else "", ())
    if not isinstance(video_code, str) or not VIDEO_CODE_RE.fullmatch(video_code):
        logger.warning("知识库截图持久化失败：视频码无效，降级为纯文字")
        return strip_video_frame_markers_for_storage(markdown_text), ()

    frame_map = _normalise_frame_mapping(frames)
    if not frame_map:
        return strip_video_frame_markers_for_storage(markdown_text), ()

    try:
        root = _prepare_persistent_asset_root(asset_root)
    except Exception as exc:
        logger.warning("知识库截图目录不可用，降级为纯文字: %s", exc)
        return strip_video_frame_markers_for_storage(markdown_text), ()

    used_ids: set[str] = set()
    persisted: list[PersistedVideoFrame] = []

    def replace_marker(match: re.Match[str]) -> str:
        annotation_id = match.group(1)
        if annotation_id in used_ids:
            return ""
        used_ids.add(annotation_id)
        frame = frame_map.get(annotation_id)
        if frame is None:
            return ""
        try:
            image_bytes = _read_verified_frame(frame)
            digest = hashlib.sha256(image_bytes).hexdigest()
            relative_path = f"blobs/{digest[:2]}/{digest}.jpg"
            destination = _write_content_addressed_jpeg(
                root,
                relative_path,
                image_bytes,
            )
            with Image.open(destination) as image:
                width, height = image.size
            display_order = len(persisted)
            record = PersistedVideoFrame(
                annotation_id=annotation_id,
                timestamp_ms=max(0, int(frame.timestamp_ms)),
                relative_path=relative_path,
                mime_type="image/jpeg",
                kind=str(frame.kind)[:40],
                confidence=str(frame.confidence)[:20],
                caption=frame.caption.strip()[:MAX_ANNOTATION_TEXT_CHARS],
                quality_score=float(frame.quality_score),
                width=width,
                height=height,
                byte_size=len(image_bytes),
                sha256=digest,
                display_order=display_order,
            )
            persisted.append(record)
        except Exception as exc:
            logger.warning("知识库截图 %s 持久化失败，已移除引用: %s", annotation_id, exc)
            return ""

        caption = _escape_markdown_alt(record.caption)
        timestamp = _format_timestamp(record.timestamp_ms)
        return (
            f"\n\n![视频画面 {timestamp}：{caption}]"
            f"(knowledge-asset://{video_code}/{annotation_id})\n\n"
        )

    rendered = FRAME_MARKER_RE.sub(replace_marker, markdown_text)
    rendered = ANY_FRAME_MARKER_RE.sub("", rendered)
    rendered = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", rendered)
    return rendered, tuple(persisted)


def strip_video_frame_markers_for_storage(markdown_text: str) -> str:
    """Remove valid and malformed screenshot markers before database storage."""

    if not isinstance(markdown_text, str):
        return ""
    cleaned = ANY_FRAME_MARKER_RE.sub("", markdown_text)
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", cleaned)


def _prepare_persistent_asset_root(
    asset_root: Union[str, os.PathLike[str]],
) -> Path:
    root = Path(asset_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("资产目录必须是无跳转段的绝对路径")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("资产目录不能是符号链接")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("资产目录不是目录")
    os.chmod(resolved, 0o700)
    return resolved


def _write_content_addressed_jpeg(
    root: Path,
    relative_path: str,
    image_bytes: bytes,
) -> Path:
    if not PERSISTED_ASSET_PATH_RE.fullmatch(relative_path):
        raise ValueError("资产相对路径无效")
    expected_digest = Path(relative_path).stem
    if hashlib.sha256(image_bytes).hexdigest() != expected_digest:
        raise ValueError("资产摘要不匹配")

    destination = root.joinpath(*Path(relative_path).parts)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("资产分片目录不能是符号链接")
    resolved_parent = destination.parent.resolve(strict=True)
    if os.path.commonpath((str(root), str(resolved_parent))) != str(root):
        raise ValueError("资产路径超出允许根目录")
    os.chmod(resolved_parent, 0o700)

    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("既有资产路径类型无效")
        if not (0 < destination.stat().st_size <= MAX_FRAME_BYTES):
            raise ValueError("既有资产大小无效")
        existing = destination.read_bytes()
        if hashlib.sha256(existing).hexdigest() != expected_digest:
            raise ValueError("既有资产内容与摘要不匹配")
        os.chmod(destination, 0o600)
        return destination

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{expected_digest}.",
        suffix=".tmp",
        dir=str(resolved_parent),
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(temporary_fd, 0o600)
        with os.fdopen(temporary_fd, "wb", closefd=True) as handle:
            temporary_fd = -1
            handle.write(image_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        directory_fd = os.open(resolved_parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return destination
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _escape_markdown_alt(value: str) -> str:
    return " ".join(value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").split())


def _validate_annotation(
    raw: RawAnnotation,
    *,
    duration_ms: int,
) -> Optional[VisualAnnotation]:
    if isinstance(raw, VisualAnnotation):
        values: Mapping[str, object] = {
            "id": raw.id,
            "start_ms": raw.start_ms,
            "end_ms": raw.end_ms,
            "kind": raw.kind,
            "confidence": raw.confidence,
            "text": raw.text,
        }
    elif isinstance(raw, Mapping):
        # Stage 1 may preserve a useful visual fact without requesting a local
        # context frame. Only literal booleans opt into extraction (truthy
        # strings do not). The old key remains accepted for replayed jobs.
        if (
            raw.get("frame_recommended") is not True
            and raw.get("screenshot_recommended") is not True
        ):
            return None
        values = raw
    else:
        return None

    annotation_id = values.get("id")
    kind = values.get("kind")
    text = values.get("text")
    start_ms = values.get("start_ms")
    end_ms = values.get("end_ms")
    if not isinstance(annotation_id, str) or not FRAME_ID_RE.fullmatch(annotation_id):
        return None
    if not isinstance(kind, str) or kind not in ALLOWED_KINDS:
        return None
    if not isinstance(text, str):
        return None
    text = " ".join(text.split()).strip()
    if not _is_body_worthy_text(text, kind=kind):
        return None
    if len(text) > MAX_ANNOTATION_TEXT_CHARS or _contains_unsafe_control(text):
        return None
    if (
        isinstance(start_ms, bool)
        or isinstance(end_ms, bool)
        or not isinstance(start_ms, int)
        or not isinstance(end_ms, int)
    ):
        return None
    if start_ms < 0 or end_ms < start_ms or end_ms > duration_ms:
        return None
    if end_ms - start_ms > MAX_ANNOTATION_SPAN_MS:
        return None

    confidence = _normalise_confidence(values.get("confidence"))
    if confidence not in ALLOWED_CONFIDENCE:
        return None
    if kind == "visible_fact" and confidence != "high":
        return None
    return VisualAnnotation(
        id=annotation_id,
        start_ms=start_ms,
        end_ms=end_ms,
        kind=kind,
        confidence=confidence,
        text=text,
    )


def _normalise_confidence(raw: object) -> Optional[str]:
    if isinstance(raw, str):
        value = raw.strip().lower()
        return value if value in ALLOWED_CONFIDENCE else None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0 or value > 1:
        return None
    if value >= 0.85:
        return "high"
    if value >= 0.65:
        return "medium"
    return None


def _is_body_worthy_text(text: str, *, kind: str) -> bool:
    compact = re.sub(r"[^0-9A-Za-z\u3400-\u9fff]+", "", text).lower()
    if compact in _GENERIC_VISUAL_TEXT:
        return False
    minimum = 6 if kind == "visible_fact" else 4
    return len(compact) >= minimum


def _contains_unsafe_control(text: str) -> bool:
    return any(ord(char) < 32 and char not in "\t\n\r" for char in text)


def _annotation_rank(annotation: VisualAnnotation) -> tuple[int, int, int, int, str]:
    kind_priority = {
        "state_change": 4,
        "ui_structure": 3,
        "ocr": 3,
        "visible_fact": 2,
    }[annotation.kind]
    confidence_priority = 2 if annotation.confidence == "high" else 1
    information = min(len(annotation.text), 240)
    # Earlier facts win exact ties so selection is deterministic.
    return (
        kind_priority,
        confidence_priority,
        information,
        -annotation.center_ms,
        annotation.id,
    )


def _safe_frame_limit(raw: object) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 0
    return min(max(raw, 0), HARD_MAX_FRAMES)


def _validate_video_path(
    raw_path: Union[str, os.PathLike[str]],
    *,
    allowed_root: Optional[Union[str, os.PathLike[str]]],
    max_video_bytes: int,
) -> Path:
    path = Path(raw_path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("视频路径必须是无跳转段的绝对路径")
    if path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError("视频扩展名不受支持")
    if path.is_symlink():
        raise ValueError("视频路径不能是符号链接")
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("视频路径不是普通文件")
    if (
        isinstance(max_video_bytes, bool)
        or not isinstance(max_video_bytes, int)
        or max_video_bytes <= 0
        or max_video_bytes > MAX_VIDEO_BYTES
    ):
        raise ValueError("视频大小上限无效")
    if info.st_size <= 0 or info.st_size > max_video_bytes:
        raise ValueError("视频文件为空或超过安全上限")
    if allowed_root is not None:
        root = _validate_existing_absolute_dir(allowed_root, label="视频根目录")
        if not _is_relative_to(resolved, root):
            raise ValueError("视频路径超出允许根目录")
    return resolved


def _prepare_output_dir(
    source: Path,
    *,
    output_dir: Optional[Union[str, os.PathLike[str]]],
    allowed_output_root: Optional[Union[str, os.PathLike[str]]],
) -> Path:
    if allowed_output_root is None:
        root = source.parent.resolve(strict=True)
    else:
        root = _validate_existing_absolute_dir(allowed_output_root, label="截图根目录")
    raw_destination = Path(output_dir) if output_dir is not None else source.parent / "video_frames"
    if not raw_destination.is_absolute() or ".." in raw_destination.parts:
        raise ValueError("截图目录必须是无跳转段的绝对路径")
    destination = raw_destination.resolve(strict=False)
    if not _is_relative_to(destination, root):
        raise ValueError("截图目录超出允许根目录")
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("截图目录不是安全的普通目录")
    else:
        parent = destination.parent.resolve(strict=True)
        if not _is_relative_to(parent, root):
            raise ValueError("截图目录父级超出允许根目录")
        destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    return destination.resolve(strict=True)


def _validate_existing_absolute_dir(
    raw_path: Union[str, os.PathLike[str]],
    *,
    label: str,
) -> Path:
    path = Path(raw_path)
    if not path.is_absolute() or ".." in path.parts or path.is_symlink():
        raise ValueError(f"{label}不安全")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label}不是目录")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _probe_video_duration_ms(source: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type,duration:format=duration",
        "-of",
        "json",
        str(source),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=FFPROBE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise ValueError("ffprobe 无法读取视频")
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams") or []
        if not streams or streams[0].get("codec_type") != "video":
            raise ValueError("文件没有视频轨道")
        raw_durations = (
            streams[0].get("duration"),
            (payload.get("format") or {}).get("duration"),
        )
        duration_seconds = next(
            value
            for value in (_finite_positive_float(raw) for raw in raw_durations)
            if value is not None
        )
    except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
        raise ValueError("视频时长无效") from exc
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("视频时长无效")
    duration_ms = int(round(duration_seconds * 1000))
    if duration_ms > MAX_VIDEO_DURATION_MS:
        raise ValueError("视频时长超过安全上限")
    return duration_ms


def _extract_best_frame(
    source: Path,
    annotation: VisualAnnotation,
    destination: Path,
    *,
    duration_ms: int,
) -> Optional[ExtractedVideoFrame]:
    candidates = _candidate_timestamps(annotation, duration_ms=duration_ms)
    best: Optional[tuple[float, int, Path]] = None
    with tempfile.TemporaryDirectory(prefix=".frame-candidates-", dir=str(destination)) as temp_name:
        temp_dir = Path(temp_name)
        for index, timestamp_ms in enumerate(candidates):
            candidate = temp_dir / f"candidate_{annotation.id}_{index}.jpg"
            if not _run_ffmpeg_frame_extract(source, timestamp_ms, candidate):
                continue
            try:
                score = _score_candidate(candidate)
            except (OSError, ValueError, UnidentifiedImageError):
                continue
            if best is None or score > best[0]:
                best = (score, timestamp_ms, candidate)
        if best is None:
            return None

        score, timestamp_ms, selected = best
        final_path = destination / f"video_frame_{annotation.id}.jpg"
        _write_sanitized_jpeg(selected, final_path)
    return ExtractedVideoFrame(
        annotation_id=annotation.id,
        timestamp_ms=timestamp_ms,
        path=final_path,
        kind=annotation.kind,
        confidence=annotation.confidence,
        caption=annotation.text,
        quality_score=score,
    )


def _candidate_timestamps(annotation: VisualAnnotation, *, duration_ms: int) -> list[int]:
    center = annotation.center_ms
    span = annotation.end_ms - annotation.start_ms
    offset = max(250, min(1000, span // 3 if span else 500))
    upper = max(0, duration_ms - 50)
    effective_end = min(annotation.end_ms, upper)
    effective_start = min(annotation.start_ms, effective_end)
    raw = [center, center - offset, center + offset]
    timestamps: list[int] = []
    for value in raw:
        clamped = min(max(value, effective_start, 0), effective_end)
        if clamped not in timestamps:
            timestamps.append(clamped)
    return timestamps


def _run_ffmpeg_frame_extract(source: Path, timestamp_ms: int, destination: Path) -> bool:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_ms / 1000:.3f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-vf",
        "scale='min(1600,iw)':-2:flags=lanczos",
        "-q:v",
        "2",
        "-y",
        str(destination),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    try:
        size = destination.stat().st_size
    except OSError:
        return False
    return result.returncode == 0 and 0 < size <= MAX_FRAME_BYTES * 4


def _score_candidate(path: Path) -> float:
    with Image.open(path) as image:
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError("候选帧尺寸无效")
        image.load()
        gray = ImageOps.grayscale(image)
        gray.thumbnail((640, 640), _resampling_lanczos())
        width, height = gray.size
        margin_x = max(1, width // 30)
        margin_y = max(1, height // 30)
        if width > margin_x * 2 + 8 and height > margin_y * 2 + 8:
            gray = gray.crop((margin_x, margin_y, width - margin_x, height - margin_y))

        stats = ImageStat.Stat(gray)
        brightness = float(stats.mean[0])
        contrast = float(stats.stddev[0])
        histogram = gray.histogram()
        pixels = max(1, gray.width * gray.height)
        clipped = (sum(histogram[:8]) + sum(histogram[248:])) / pixels

        edges = gray.filter(ImageFilter.FIND_EDGES)
        if edges.width > 4 and edges.height > 4:
            edges = edges.crop((2, 2, edges.width - 2, edges.height - 2))
        edge_variance = float(ImageStat.Stat(edges).var[0])

        brightness_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        exposure_score = max(0.0, brightness_score - min(1.0, clipped * 3.0))
        contrast_score = min(1.0, contrast / 64.0)
        return math.log1p(max(0.0, edge_variance)) + 1.5 * exposure_score + 0.4 * contrast_score


def _write_sanitized_jpeg(source: Path, destination: Path) -> None:
    if not FRAME_ID_RE.fullmatch(destination.stem.removeprefix("video_frame_")):
        raise ValueError("截图文件名无效")
    if destination.name != f"video_frame_{destination.stem.removeprefix('video_frame_')}.jpg":
        raise ValueError("截图文件名无效")
    with Image.open(source) as image:
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError("截图尺寸无效")
        image.load()
        clean = ImageOps.exif_transpose(image).convert("RGB")
        clean = _crop_portrait_evidence(clean)
        clean.thumbnail((MAX_FRAME_DIMENSION, MAX_FRAME_DIMENSION), _resampling_lanczos())

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-",
            suffix=".jpg",
            dir=str(destination.parent),
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            clean.save(temporary, format="JPEG", quality=86, optimize=True, progressive=True)
            if temporary.stat().st_size > MAX_FRAME_BYTES:
                clean.save(temporary, format="JPEG", quality=70, optimize=True, progressive=True)
            if temporary.stat().st_size <= 0 or temporary.stat().st_size > MAX_FRAME_BYTES:
                raise ValueError("截图超过内嵌大小上限")
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _crop_portrait_evidence(image: Image.Image) -> Image.Image:
    """Crop tall social-video frames to the densest square evidence region.

    Full 9:16 frames often contain large title/subtitle bands around a central
    slide or interface.  Rendering the whole frame at document width makes the
    actual evidence tiny and can consume an entire PDF page.  A deterministic
    edge/detail scan keeps the most information-dense square; the independent
    multimodal reviewer still sees this final crop and can reject it.
    """

    width, height = image.size
    if width < 32 or height / max(width, 1) < PORTRAIT_CROP_TRIGGER_RATIO:
        return image

    crop_height = width
    analysis_width = min(320, width)
    scale = analysis_width / width
    analysis_height = max(analysis_width, round(height * scale))
    gray = ImageOps.grayscale(image).resize(
        (analysis_width, analysis_height),
        _resampling_lanczos(),
    )
    edges = gray.filter(ImageFilter.FIND_EDGES)
    gray_bytes = gray.tobytes()
    edge_bytes = edges.tobytes()

    row_scores: list[float] = []
    for row in range(analysis_height):
        start = row * analysis_width
        end = start + analysis_width
        edge_mean = sum(edge_bytes[start:end]) / analysis_width
        non_dark_ratio = sum(value > 24 for value in gray_bytes[start:end]) / analysis_width
        row_scores.append(edge_mean + 8.0 * non_dark_ratio)

    prefix = [0.0]
    for value in row_scores:
        prefix.append(prefix[-1] + value)
    window = analysis_width
    centre = analysis_height / 2.0
    best_row = max(
        range(analysis_height - window + 1),
        key=lambda row: (
            prefix[row + window] - prefix[row],
            -abs(row + window / 2.0 - centre),
        ),
    )
    top = min(max(0, round(best_row / scale)), height - crop_height)
    return image.crop((0, top, width, top + crop_height))


def _resampling_lanczos() -> int:
    resampling = getattr(Image, "Resampling", Image)
    return int(resampling.LANCZOS)


def _normalise_frame_mapping(
    frames: Union[Mapping[str, ExtractedVideoFrame], Sequence[ExtractedVideoFrame]],
) -> dict[str, ExtractedVideoFrame]:
    output: dict[str, ExtractedVideoFrame] = {}
    try:
        items = frames.items() if isinstance(frames, Mapping) else (
            (frame.annotation_id, frame) for frame in frames
        )
        for key, frame in itertools.islice(items, HARD_MAX_FRAMES):
            if (
                isinstance(key, str)
                and FRAME_ID_RE.fullmatch(key)
                and isinstance(frame, ExtractedVideoFrame)
                and frame.annotation_id == key
            ):
                output[key] = frame
    except (AttributeError, TypeError, ValueError):
        return {}
    return output


def _safe_embedded_byte_budget(raw: object) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        return 0
    return min(raw, MAX_TOTAL_EMBEDDED_BYTES)


def _read_verified_frame(frame: ExtractedVideoFrame) -> bytes:
    if not FRAME_ID_RE.fullmatch(frame.annotation_id):
        raise ValueError("截图 ID 无效")
    path = Path(frame.path)
    expected_name = f"video_frame_{frame.annotation_id}.jpg"
    if not path.is_absolute() or ".." in path.parts or path.name != expected_name:
        raise ValueError("截图路径无效")
    if path.is_symlink():
        raise ValueError("截图不能是符号链接")
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > MAX_FRAME_BYTES:
        raise ValueError("截图文件大小无效")
    image_bytes = resolved.read_bytes()
    if not image_bytes.startswith(b"\xff\xd8") or not image_bytes.endswith(b"\xff\xd9"):
        raise ValueError("截图不是完整 JPEG")
    try:
        with Image.open(resolved) as image:
            width, height = image.size
            if image.format != "JPEG" or width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("截图图片格式无效")
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("截图图片已损坏") from exc
    return image_bytes


def _frame_average_hash(frame: ExtractedVideoFrame) -> int:
    image_bytes = _read_verified_frame(frame)
    with Image.open(io.BytesIO(image_bytes)) as image:
        gray = ImageOps.grayscale(ImageOps.exif_transpose(image)).resize(
            (FRAME_HASH_SIZE, FRAME_HASH_SIZE),
            _resampling_lanczos(),
        )
        pixels = gray.tobytes()
    average = sum(pixels) / max(1, len(pixels))
    fingerprint = 0
    for value in pixels:
        fingerprint = (fingerprint << 1) | int(value > average)
    return fingerprint


def _deduplicate_similar_frames(
    frames: Union[Mapping[str, ExtractedVideoFrame], Sequence[ExtractedVideoFrame]],
) -> dict[str, ExtractedVideoFrame]:
    """Keep the clearest frame from each short-lived near-duplicate scene."""

    frame_map = _normalise_frame_mapping(frames)
    if len(frame_map) < 2:
        return frame_map

    kept: list[ExtractedVideoFrame] = []
    hashes: dict[str, int] = {}
    semantic_keys: dict[str, str] = {}
    ordered = sorted(
        frame_map.values(),
        key=lambda frame: (frame.timestamp_ms, frame.annotation_id),
    )
    for candidate in ordered:
        candidate_semantic_key = " ".join(candidate.caption.split()).casefold()
        try:
            candidate_hash = _frame_average_hash(candidate)
        except Exception as exc:
            logger.warning("近重复截图校验失败，已跳过 %s: %s", candidate.annotation_id, exc)
            continue

        duplicate_indexes = [
            index
            for index, existing in enumerate(kept)
            if (
                candidate_semantic_key == semantic_keys[existing.annotation_id]
                and abs(candidate.timestamp_ms - existing.timestamp_ms)
                <= FRAME_DEDUP_MAX_GAP_MS
                and (candidate_hash ^ hashes[existing.annotation_id]).bit_count()
                <= FRAME_DEDUP_MAX_HASH_DISTANCE
            )
        ]
        if not duplicate_indexes:
            kept.append(candidate)
            hashes[candidate.annotation_id] = candidate_hash
            semantic_keys[candidate.annotation_id] = candidate_semantic_key
            continue

        duplicate_index = min(
            duplicate_indexes,
            key=lambda index: (
                (candidate_hash ^ hashes[kept[index].annotation_id]).bit_count(),
                abs(candidate.timestamp_ms - kept[index].timestamp_ms),
            ),
        )
        existing = kept[duplicate_index]
        candidate_quality = (
            candidate.quality_score if math.isfinite(candidate.quality_score) else 0.0
        )
        existing_quality = (
            existing.quality_score if math.isfinite(existing.quality_score) else 0.0
        )
        if candidate_quality > existing_quality:
            hashes.pop(existing.annotation_id, None)
            semantic_keys.pop(existing.annotation_id, None)
            kept[duplicate_index] = candidate
            hashes[candidate.annotation_id] = candidate_hash
            semantic_keys[candidate.annotation_id] = candidate_semantic_key

    return {
        frame.annotation_id: frame
        for frame in sorted(kept, key=lambda value: (value.timestamp_ms, value.annotation_id))
    }


def _format_timestamp(timestamp_ms: int) -> str:
    total_seconds = max(0, int(round(timestamp_ms / 1000)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _finite_positive_float(raw: object) -> Optional[float]:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value
