"""Offline contract tests for the 图文 (picture album) summarization path."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image, ImageDraw


with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-key"}, clear=False):
    with patch("dotenv.load_dotenv", return_value=False):
        from app.services import ai_summarizer
        from app.services import album_summarizer as album
        from app.services.video_frames import ExtractedVideoFrame


def _write_card(path: Path, *, fmt: str = "JPEG", size: tuple[int, int] = (720, 1018)) -> None:
    image = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, size[0] - 40, 160), fill=(30, 30, 30))
    image.save(path, format=fmt)


def _frames(directory: Path, count: int) -> tuple[ExtractedVideoFrame, ...]:
    paths = []
    for index in range(1, count + 1):
        path = directory / f"image_{index:02d}.jpg"
        _write_card(path)
        paths.append(str(path))
    return album.prepare_album_frames(paths, directory / "video_frames")


def _page(
    page_id: str,
    index: int,
    *,
    text: str = "",
    notes: str = "",
    kind: str = "ocr",
    confidence: str = "high",
    editor: bool = False,
    screenshot: bool = False,
    sensitive: bool = False,
) -> album.AlbumPage:
    return album.AlbumPage(
        page_id=page_id,
        index=index,
        text_transcript=text,
        visual_notes=notes,
        kind=kind,
        confidence=confidence,
        editor_frame_recommended=editor,
        screenshot_recommended=screenshot,
        screenshot_reason=None,
        contains_sensitive_data=sensitive,
    )


class PrepareFramesTests(unittest.TestCase):
    def test_pictures_keep_their_album_index_and_broken_ones_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            first, second, third = (directory / f"image_0{n}.jpg" for n in (1, 2, 3))
            _write_card(first)
            second.write_bytes(b"broken")
            _write_card(third, fmt="WEBP")

            frames = album.prepare_album_frames(
                [str(first), str(second), str(third)], directory / "video_frames"
            )

            self.assertEqual([frame.annotation_id for frame in frames], ["V0001", "V0003"])
            self.assertEqual([frame.source_label for frame in frames], ["图片 1", "图片 3"])
            self.assertEqual(
                [frame.path.name for frame in frames],
                ["video_frame_V0001.jpg", "video_frame_V0003.jpg"],
            )
            for frame in frames:
                self.assertTrue(frame.path.read_bytes().startswith(b"\xff\xd8\xff"))

    def test_no_readable_picture_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            broken = directory / "image_01.jpg"
            broken.write_bytes(b"broken")

            with self.assertRaises(ValueError):
                album.prepare_album_frames([str(broken)], directory / "video_frames")


class PageValidationTests(unittest.TestCase):
    def _frames(self, count: int) -> tuple[ExtractedVideoFrame, ...]:
        return tuple(
            ExtractedVideoFrame(
                annotation_id=f"V{index:04d}",
                timestamp_ms=0,
                path=Path(f"/nonexistent/video_frame_V{index:04d}.jpg"),
                kind="visible_fact",
                confidence="high",
                caption=f"图片 {index}",
                quality_score=0.0,
                source_label=f"图片 {index}",
            )
            for index in range(1, count + 1)
        )

    def test_every_picture_gets_a_page_and_model_fields_are_bounded(self) -> None:
        raw = json.dumps(
            {
                "schema_version": "stage1.image-album.v1",
                "pages": [
                    {
                        "id": "V0002",
                        "text_transcript": "第二张的文字",
                        "visual_notes": "流程图：三个状态互相转移",
                        "kind": "diagram",
                        "confidence": "certain",
                        "editor_frame_recommended": True,
                        "screenshot_recommended": True,
                        "screenshot_reason": "箭头关系文字难以表达",
                        "contains_sensitive_data": False,
                    },
                    {"id": "V0002", "text_transcript": "重复 ID 不应覆盖第一条"},
                    {"id": "V0009", "text_transcript": "不存在的图片"},
                    {"id": "../x", "text_transcript": "非法 ID"},
                    {
                        "id": "V0003",
                        "text_transcript": "x" * 9000,
                        "visual_notes": "",
                        "kind": "ocr",
                        "confidence": "high",
                        "editor_frame_recommended": False,
                        "screenshot_recommended": True,
                    },
                ],
            },
            ensure_ascii=False,
        )

        pages = album.validated_album_pages(raw, self._frames(3))

        self.assertEqual([page.page_id for page in pages], ["V0001", "V0002", "V0003"])
        self.assertEqual([page.index for page in pages], [1, 2, 3])
        first, second, third = pages
        self.assertEqual(first.text_transcript, "")
        self.assertFalse(first.editor_frame_recommended)
        self.assertEqual(second.text_transcript, "第二张的文字")
        self.assertEqual(second.visual_notes, "流程图：三个状态互相转移")
        self.assertEqual(second.kind, "visible_fact")   # unknown kind falls back
        self.assertEqual(second.confidence, "medium")   # unknown confidence falls back
        self.assertTrue(second.editor_frame_recommended)
        self.assertTrue(second.screenshot_recommended)
        self.assertEqual(second.screenshot_reason, "箭头关系文字难以表达")
        self.assertEqual(len(third.text_transcript), album.MAX_PAGE_TEXT_CHARS)
        self.assertFalse(third.screenshot_recommended)  # needs editor_frame_recommended

    def test_sensitive_pictures_are_never_offered_to_the_editor(self) -> None:
        raw = json.dumps(
            {
                "pages": [
                    {
                        "id": "V0001",
                        "text_transcript": "账号截图",
                        "editor_frame_recommended": True,
                        "screenshot_recommended": True,
                        "contains_sensitive_data": True,
                    }
                ]
            }
        )

        (page,) = album.validated_album_pages(raw, self._frames(1))

        self.assertTrue(page.contains_sensitive_data)
        self.assertFalse(page.editor_frame_recommended)
        self.assertFalse(page.screenshot_recommended)

    def test_malformed_json_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            album.validated_album_pages("这不是 JSON", self._frames(1))
        with self.assertRaises(ValueError):
            album.validated_album_pages('{"pages": "nope"}', self._frames(1))


class TranscriptAssemblyTests(unittest.TestCase):
    PAGES = (
        _page("V0001", 1, text="第一张：定义问题"),
        _page("V0002", 2),
        _page(
            "V0003",
            3,
            text="第三张\n- 要点",
            notes="流程图：三个状态之间的转移箭头",
            kind="visible_fact",
            editor=True,
            screenshot=True,
        ),
    )

    def test_transcript_lists_caption_then_every_picture_in_order(self) -> None:
        self.assertEqual(
            album.build_album_transcript("量化策略思路分享", self.PAGES),
            "【帖子文案】\n量化策略思路分享\n\n"
            "【图片 1】\n第一张：定义问题\n\n"
            "【图片 2】\n（无可读文字）\n\n"
            "【图片 3】\n第三张\n- 要点",
        )

    def test_empty_caption_is_omitted(self) -> None:
        self.assertEqual(
            album.build_album_transcript("  ", self.PAGES[:1]),
            "【图片 1】\n第一张：定义问题",
        )

    def test_enhanced_transcript_appends_visual_notes_to_their_picture(self) -> None:
        self.assertEqual(
            album.build_enhanced_album_transcript("量化策略思路分享", self.PAGES),
            "【帖子文案】\n量化策略思路分享\n\n"
            "【图片 1】\n第一张：定义问题\n\n"
            "【图片 2】\n（无可读文字）\n\n"
            "【图片 3】\n第三张\n- 要点\n"
            "> [画面 V0003｜visible_fact｜high] 流程图：三个状态之间的转移箭头",
        )

    def test_visual_evidence_lists_only_pictures_with_notes_and_frame_state(self) -> None:
        self.assertEqual(
            album.format_album_visual_evidence(self.PAGES, frozenset({"V0003"})),
            "- [V0003｜图片 3｜visible_fact｜high｜editor_frame_available=true｜"
            "screenshot_hint=true] 流程图：三个状态之间的转移箭头",
        )
        self.assertEqual(
            album.format_album_visual_evidence(self.PAGES[:2], frozenset()),
            "图片文字之外无实质画面信息。",
        )


class PolicyAndFrameSelectionTests(unittest.TestCase):
    def test_policy_grows_with_transcribed_text_volume(self) -> None:
        expectations = [
            (0, ai_summarizer.COMPACT_POLICY),
            (600, ai_summarizer.COMPACT_POLICY),
            (601, ai_summarizer.STANDARD_POLICY),
            (2000, ai_summarizer.STANDARD_POLICY),
            (2001, ai_summarizer.EXTENDED_POLICY),
            (4500, ai_summarizer.EXTENDED_POLICY),
            (4501, ai_summarizer.LONG_POLICY),
        ]
        for char_count, expected in expectations:
            with self.subTest(chars=char_count):
                self.assertIs(album.select_album_generation_policy(char_count), expected)

    def test_editor_frames_prefer_screenshot_hints_then_reading_order_capped_at_eight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            frames = _frames(directory, 12)
            pages = tuple(
                _page(
                    f"V{index:04d}",
                    index,
                    text=f"第 {index} 张文字",
                    notes="图表" if index >= 10 else "",
                    kind="visible_fact" if index >= 10 else "ocr",
                    editor=True,
                    screenshot=index >= 10,
                )
                for index in range(1, 13)
            )

            selected = album.select_editor_frames(pages, frames)

            self.assertEqual(
                [frame.annotation_id for frame in selected],
                ["V0001", "V0002", "V0003", "V0004", "V0005", "V0010", "V0011", "V0012"],
            )
            self.assertEqual(selected[0].caption, "第 1 张文字")
            self.assertEqual(selected[0].kind, "ocr")
            self.assertEqual(selected[-1].caption, "图表")
            self.assertEqual(selected[-1].kind, "visible_fact")
            self.assertEqual(selected[-1].source_label, "图片 12")

    def test_unrecommended_and_missing_frames_are_not_offered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            frames = _frames(directory, 2)
            pages = (
                _page("V0001", 1, text="纯文字", editor=False),
                _page("V0002", 2, text="图", editor=True),
                _page("V0003", 3, text="没有对应文件", editor=True),
            )

            selected = album.select_editor_frames(pages, frames)

            self.assertEqual([frame.annotation_id for frame in selected], ["V0002"])


class ReadAlbumPagesTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_picture_is_sent_once_in_order_with_its_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            frames = _frames(directory, 3)
            raw = json.dumps(
                {
                    "schema_version": "stage1.image-album.v1",
                    "pages": [
                        {"id": "V0001", "text_transcript": "一", "editor_frame_recommended": False},
                        {"id": "V0002", "text_transcript": "二", "editor_frame_recommended": False},
                        {"id": "V0003", "text_transcript": "三", "visual_notes": "图表", "editor_frame_recommended": True},
                    ],
                }
            )
            with patch.object(
                album.aliyun_client, "chat", new_callable=AsyncMock, return_value=raw
            ) as chat:
                pages = await album.read_album_pages(
                    frames,
                    video_title="量化策略思路分享",
                    video_author="新用户1401",
                    caption="量化策略思路分享。摘要……",
                    user_requirement="关注风险控制",
                )

            self.assertEqual([page.text_transcript for page in pages], ["一", "二", "三"])
            kwargs = chat.await_args.kwargs
            self.assertEqual(kwargs["model"], ai_summarizer.ALIYUN_VISUAL_MODEL)
            self.assertEqual(kwargs["operation"], "album_reading")
            self.assertEqual(kwargs["response_format"], {"type": "json_object"})
            self.assertIs(kwargs["enable_thinking"], False)
            self.assertIs(kwargs["preserve_thinking"], False)
            self.assertEqual(kwargs["messages"][0]["content"], album.ALBUM_STAGE1_SYSTEM)
            self.assertEqual(kwargs["timeout"], album.AI_LONG_REQUEST_TIMEOUT_SECONDS)
            parts = kwargs["messages"][1]["content"]
            image_parts = [part for part in parts if part["type"] == "image_url"]
            self.assertEqual(len(image_parts), 3)
            for part in image_parts:
                url = part["image_url"]["url"]
                self.assertTrue(url.startswith("data:image/jpeg;base64,"))
                self.assertTrue(base64.b64decode(url.split(",", 1)[1]).startswith(b"\xff\xd8\xff"))
            markers = [
                json.loads(part["text"])["picture"]
                for part in parts
                if part["type"] == "text" and part["text"].startswith('{"picture"')
            ]
            self.assertEqual(
                markers,
                [{"id": "V0001", "index": 1}, {"id": "V0002", "index": 2}, {"id": "V0003", "index": 3}],
            )
            self.assertEqual(parts.index(image_parts[0]) - 1, parts.index(
                next(p for p in parts if p["type"] == "text" and '"V0001"' in p["text"])
            ))
            metadata = json.loads(parts[-1]["text"])
            self.assertEqual(metadata["post"]["title"], "量化策略思路分享")
            self.assertEqual(metadata["post"]["author"], "新用户1401")
            self.assertEqual(metadata["post"]["caption"], "量化策略思路分享。摘要……")
            self.assertEqual(metadata["post"]["picture_count"], 3)
            self.assertEqual(metadata["user_focus"], "关注风险控制")


class BatchedReadingTests(unittest.IsolatedAsyncioTestCase):
    """Long albums are read in bounded batches so one call never outgrows the client timeout."""

    async def test_large_album_is_read_in_concurrent_batches_and_merged_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            frames = _frames(directory, 10)
            requests: list[dict] = []
            in_flight = 0
            max_in_flight = 0

            async def fake_chat(**kwargs):
                nonlocal in_flight, max_in_flight
                parts = kwargs["messages"][1]["content"]
                ids = [
                    json.loads(part["text"])["picture"]["id"]
                    for part in parts
                    if part["type"] == "text" and part["text"].startswith('{"picture"')
                ]
                requests.append({"ids": ids, "metadata": json.loads(parts[-1]["text"])})
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                await asyncio.sleep(0.05)
                in_flight -= 1
                return json.dumps(
                    {
                        "pages": [
                            {"id": page_id, "text_transcript": f"文字{page_id}", "editor_frame_recommended": False}
                            for page_id in ids
                        ]
                    }
                )

            with patch.object(album, "ALBUM_READ_BATCH_SIZE", 4), patch.object(
                album.aliyun_client, "chat", side_effect=fake_chat
            ):
                pages = await album.read_album_pages(
                    frames, video_title="标题", video_author="作者", caption="文案", user_requirement=""
                )

            requests.sort(key=lambda request: request["ids"][0])  # batches run concurrently
            self.assertEqual(
                [request["ids"] for request in requests],
                [
                    ["V0001", "V0002", "V0003", "V0004"],
                    ["V0005", "V0006", "V0007", "V0008"],
                    ["V0009", "V0010"],
                ],
            )
            self.assertEqual(
                [request["metadata"]["batch"] for request in requests],
                [
                    {"first_index": 1, "last_index": 4, "picture_count": 10},
                    {"first_index": 5, "last_index": 8, "picture_count": 10},
                    {"first_index": 9, "last_index": 10, "picture_count": 10},
                ],
            )
            self.assertTrue(all(request["metadata"]["post"]["picture_count"] == 10 for request in requests))
            self.assertEqual([page.page_id for page in pages], [f"V{i:04d}" for i in range(1, 11)])
            self.assertEqual([page.text_transcript for page in pages], [f"文字V{i:04d}" for i in range(1, 11)])
            self.assertGreaterEqual(max_in_flight, 2)

    async def test_a_failed_batch_fails_the_whole_reading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            frames = _frames(directory, 5)

            async def fake_chat(**kwargs):
                parts = kwargs["messages"][1]["content"]
                if '"V0005"' in parts[0]["text"]:
                    raise RuntimeError("batch down")
                ids = [
                    json.loads(part["text"])["picture"]["id"]
                    for part in parts
                    if part["type"] == "text" and part["text"].startswith('{"picture"')
                ]
                return json.dumps({"pages": [{"id": page_id, "text_transcript": "x"} for page_id in ids]})

            with patch.object(album, "ALBUM_READ_BATCH_SIZE", 4), patch.object(
                album.aliyun_client, "chat", side_effect=fake_chat
            ):
                with self.assertRaises(RuntimeError):
                    await album.read_album_pages(
                        frames, video_title="标题", video_author="作者", caption="", user_requirement=""
                    )


class RequestBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_album_is_reencoded_smaller_but_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            frames = _frames(directory, 3)
            original_sizes = [frame.path.stat().st_size for frame in frames]
            budget = sum(original_sizes) - 1

            with patch.object(album, "ALBUM_REQUEST_BYTE_BUDGET", budget):
                parts = await album._picture_parts(frames)

            image_parts = [part for part in parts if part["type"] == "image_url"]
            self.assertEqual(len(image_parts), 3)
            sent_sizes = [
                len(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))
                for part in image_parts
            ]
            self.assertLessEqual(sum(sent_sizes), budget)
            for sent, original in zip(sent_sizes, original_sizes):
                self.assertLess(sent, original)
            for frame, original in zip(frames, original_sizes):
                self.assertEqual(frame.path.stat().st_size, original)  # disk copies untouched

    async def test_album_that_cannot_fit_the_budget_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            frames = _frames(directory, 2)

            with patch.object(album, "ALBUM_REQUEST_BYTE_BUDGET", 10):
                with self.assertRaises(ValueError):
                    await album._picture_parts(frames)


class SummarizeAlbumTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_then_researches_then_finalizes_with_album_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            image_paths = []
            for index in (1, 2):
                path = directory / f"image_0{index}.jpg"
                _write_card(path)
                image_paths.append(str(path))
            pages = (
                _page("V0001", 1, text="第一张：问题定义"),
                _page("V0002", 2, text="第二张", notes="状态转移图", kind="visible_fact", editor=True, screenshot=True),
            )
            progress = AsyncMock()

            with patch.object(
                album, "read_album_pages", new_callable=AsyncMock, return_value=pages
            ) as reader, patch.object(
                album, "stage2_deep_research", new_callable=AsyncMock, return_value="研究备忘"
            ) as research, patch.object(
                album, "stage3_enrich_and_finalize", new_callable=AsyncMock, return_value="# 终稿"
            ) as final:
                result = await album.summarize_album_with_artifacts(
                    image_paths,
                    "量化策略思路分享",
                    "新用户1401",
                    "关注风险控制",
                    progress_callback=progress,
                    caption="量化策略思路分享。摘要",
                    work_dir=str(directory),
                )

            self.assertEqual(result.markdown, "# 终稿")
            self.assertEqual(
                result.video_frames[0].path, directory / "video_frames" / "video_frame_V0002.jpg"
            )
            self.assertEqual([frame.annotation_id for frame in result.video_frames], ["V0002"])
            self.assertEqual(result.video_frames[0].caption, "状态转移图")
            self.assertEqual(result.diagnostics.duration_seconds, None)
            self.assertEqual(result.diagnostics.timestamp_quality, "image_sequence")
            self.assertEqual(result.diagnostics.visual_annotation_count, 2)
            self.assertEqual(result.diagnostics.approved_screenshot_count, 1)
            self.assertEqual(result.diagnostics.policy_name, ai_summarizer.COMPACT_POLICY.name)

            sent_frames = reader.await_args.args[0]
            self.assertEqual([frame.annotation_id for frame in sent_frames], ["V0001", "V0002"])
            self.assertEqual(reader.await_args.kwargs["caption"], "量化策略思路分享。摘要")

            transcript = (
                "【帖子文案】\n量化策略思路分享。摘要\n\n"
                "【图片 1】\n第一张：问题定义\n\n"
                "【图片 2】\n第二张"
            )
            self.assertEqual(research.await_args.args[0], transcript)
            self.assertIs(research.await_args.args[1], ai_summarizer.COMPACT_POLICY)
            self.assertEqual(research.await_args.kwargs["source_kind"], "image_album")
            self.assertIsNone(research.await_args.kwargs["duration_seconds"])
            self.assertEqual(research.await_args.kwargs["transcript_char_count"], len(transcript))

            final_kwargs = final.await_args.kwargs
            self.assertEqual(final.await_args.args[0], transcript + "\n> [画面 V0002｜visible_fact｜high] 状态转移图")
            self.assertEqual(final.await_args.args[1], "研究备忘")
            self.assertEqual(final_kwargs["source_kind"], "image_album")
            self.assertEqual(final_kwargs["source_transcript"], transcript)
            self.assertIn("editor_frame_available=true", final_kwargs["visual_evidence"])
            self.assertEqual([f.annotation_id for f in final_kwargs["video_frames"]], ["V0002"])
            self.assertIsNone(final_kwargs["duration_seconds"])
            self.assertIs(final_kwargs["policy"], ai_summarizer.COMPACT_POLICY)
            self.assertEqual(final_kwargs["request_timeout"], album.AI_LONG_REQUEST_TIMEOUT_SECONDS)
            self.assertEqual(final_kwargs["video_title"], "量化策略思路分享")
            self.assertEqual(final_kwargs["user_requirement"], "关注风险控制")

    async def test_reader_failure_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            path = directory / "image_01.jpg"
            _write_card(path)

            with patch.object(
                album, "read_album_pages", new_callable=AsyncMock, side_effect=RuntimeError("model down")
            ), patch.object(album, "stage2_deep_research", new_callable=AsyncMock) as research:
                with self.assertRaises(RuntimeError):
                    await album.summarize_album_with_artifacts(
                        [str(path)], "标题", "作者", "", work_dir=str(directory)
                    )

            research.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
