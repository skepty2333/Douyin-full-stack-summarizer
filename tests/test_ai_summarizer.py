"""Fully offline contract tests for the three-stage summarization pipeline."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image


with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-key"}, clear=False):
    with patch("dotenv.load_dotenv", return_value=False):
        from app.services import ai_summarizer
        from app.services.aliyun_client import (
            AliyunAPIError,
            TranscriptionResult,
            TranscriptionSentence,
        )
        from app.services.video_frames import ExtractedVideoFrame


RAW_TRANSCRIPT = "第一句原话。\n第二句原话！"


def _timestamped_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        text=RAW_TRANSCRIPT,
        sentences=(
            TranscriptionSentence("第一句原话。", 0, 4_000, 0),
            TranscriptionSentence("第二句原话！", 4_000, 9_000, 0),
        ),
    )


def _timestamped_segments() -> tuple[ai_summarizer.TranscriptSegment, ...]:
    segments, quality = ai_summarizer._normalize_transcript_segments(
        _timestamped_transcription(),
        duration_seconds=9,
    )
    if quality != "sentence_exact":
        raise AssertionError("test fixture must retain exact FileTrans timestamps")
    return segments


def _visual_annotation(
    *,
    annotation_id: str = "V0001",
    anchor_segment_id: str = "S0001",
    text: str = "画面新增了一个三列对比表。",
) -> ai_summarizer.VisualAnnotation:
    return ai_summarizer.VisualAnnotation(
        annotation_id=annotation_id,
        start_ms=1_000,
        end_ms=1_500,
        anchor_segment_id=anchor_segment_id,
        kind="ui_structure",
        confidence="high",
        text=text,
    )


class DetailedTranscriptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_filetrans_detailed_is_primary_and_preserves_timestamps(self) -> None:
        callback = AsyncMock()
        detailed = _timestamped_transcription()

        with patch.object(
            ai_summarizer.aliyun_client,
            "transcribe_file_url_detailed",
            new_callable=AsyncMock,
            return_value=detailed,
        ) as filetrans, patch.object(
            ai_summarizer,
            "_transcribe_local_audio_detailed",
            new_callable=AsyncMock,
        ) as local_fallback:
            result = await ai_summarizer._transcribe_audio_detailed(
                "/virtual/video.mp4",
                media_url="https://media.invalid/video.mp4",
                video_title="离线标题",
                video_author="离线作者",
                callback=callback,
            )

        self.assertIs(result, detailed)
        self.assertEqual(result.sentences[1].begin_time_ms, 4_000)
        filetrans.assert_awaited_once_with(
            model=ai_summarizer.ALIYUN_ASR_MODEL,
            file_url="https://media.invalid/video.mp4",
            language_hints=["zh", "en"],
            channel_ids=[0],
            poll_interval=ai_summarizer.ASR_FILE_POLL_INTERVAL_SECONDS,
            timeout=ai_summarizer.ASR_FILE_TIMEOUT_SECONDS,
        )
        local_fallback.assert_not_awaited()
        callback.assert_not_awaited()

    async def test_filetrans_failure_routes_to_detailed_local_fallback(self) -> None:
        callback = AsyncMock()
        fallback = TranscriptionResult(
            text="本地回退原句。",
            sentences=(TranscriptionSentence("本地回退原句。", 0, 2_000, 0),),
        )

        with patch.object(
            ai_summarizer.aliyun_client,
            "transcribe_file_url_detailed",
            new_callable=AsyncMock,
            side_effect=AliyunAPIError("offline FileTrans failure"),
        ) as filetrans, patch.object(
            ai_summarizer,
            "_transcribe_local_audio_detailed",
            new_callable=AsyncMock,
            return_value=fallback,
        ) as local_fallback:
            result = await ai_summarizer._transcribe_audio_detailed(
                "/virtual/audio.mp3",
                media_url="https://media.invalid/video.mp4",
                video_title="离线标题",
                video_author="离线作者",
                callback=callback,
            )

        self.assertIs(result, fallback)
        self.assertEqual(filetrans.await_args.kwargs["model"], ai_summarizer.ALIYUN_ASR_MODEL)
        local_fallback.assert_awaited_once_with(
            "/virtual/audio.mp3",
            video_title="离线标题",
            video_author="离线作者",
            callback=callback,
        )
        self.assertEqual(
            [call.args[0] for call in callback.await_args_list],
            ["文件转写不可用，正在切换本地音频识别"],
        )

    async def test_local_fallback_builds_structured_approximate_timestamps(self) -> None:
        callback = AsyncMock()
        with patch.object(
            ai_summarizer,
            "_split_audio_if_needed",
            new_callable=AsyncMock,
            return_value=(["/virtual/part-1.mp3", "/virtual/part-2.mp3"], False),
        ), patch.object(
            ai_summarizer,
            "_audio_duration",
            new_callable=AsyncMock,
            side_effect=[2.0, 3.0],
        ) as duration_probe, patch.object(
            ai_summarizer.aliyun_client,
            "transcribe_audio",
            new_callable=AsyncMock,
            side_effect=["甲句。乙句。", "丙句。"],
        ) as local_asr:
            result = await ai_summarizer._transcribe_local_audio_detailed(
                "/virtual/audio.mp3",
                video_title="离线标题",
                video_author="离线作者",
                callback=callback,
            )

        self.assertEqual(result.text, "甲句。乙句。\n\n丙句。")
        self.assertEqual(result.sentences[0].begin_time_ms, 0)
        self.assertEqual(result.sentences[1].end_time_ms, 2_000)
        self.assertEqual(result.sentences[2].begin_time_ms, 2_000)
        self.assertEqual(result.sentences[2].end_time_ms, 5_000)
        self.assertEqual(duration_probe.await_count, 2)
        self.assertEqual(local_asr.await_count, 2)
        for call in local_asr.await_args_list:
            self.assertEqual(call.kwargs["model"], ai_summarizer.ALIYUN_ASR_FALLBACK_MODEL)
            self.assertIn("视频标题：离线标题", call.kwargs["context"])
            self.assertIn("视频作者：离线作者", call.kwargs["context"])
        self.assertEqual(
            [call.args[0] for call in callback.await_args_list],
            ["语音转写 1/2", "语音转写 2/2"],
        )


class DurationPolicyTests(unittest.TestCase):
    def test_six_duration_bands_have_monotonic_final_thinking_budgets(self) -> None:
        cases = (
            (60, ai_summarizer.MICRO_POLICY, False, None, 3_200),
            (61, ai_summarizer.COMPACT_POLICY, True, 1_024, 6_000),
            (181, ai_summarizer.STANDARD_POLICY, True, 2_048, 8_500),
            (301, ai_summarizer.EXTENDED_POLICY, True, 4_096, 12_000),
            (451, ai_summarizer.LONG_POLICY, True, 8_192, 18_000),
            (600, ai_summarizer.ULTRA_POLICY, True, 16_384, None),
        )

        for duration, expected, thinking, thinking_budget, max_tokens in cases:
            with self.subTest(duration=duration, tier=expected.name):
                policy = ai_summarizer.select_generation_policy(
                    duration_seconds=duration,
                    transcript_char_count=1,
                    draft_char_count=1,
                )
                self.assertIs(policy, expected)
                self.assertIs(policy.final_enable_thinking, thinking)
                self.assertEqual(policy.final_thinking_budget, thinking_budget)
                self.assertEqual(policy.final_max_completion_tokens, max_tokens)
                self.assertIs(policy.research_enable_thinking, False)
                self.assertIsNone(policy.research_thinking_budget)
                self.assertGreater(policy.research_body_char_budget, 0)
                self.assertGreater(policy.research_max_output_tokens, 0)

    def test_duration_boundaries_select_all_six_tiers(self) -> None:
        cases = (
            (1, ai_summarizer.MICRO_POLICY),
            (60, ai_summarizer.MICRO_POLICY),
            (61, ai_summarizer.COMPACT_POLICY),
            (180, ai_summarizer.COMPACT_POLICY),
            (181, ai_summarizer.STANDARD_POLICY),
            (300, ai_summarizer.STANDARD_POLICY),
            (301, ai_summarizer.EXTENDED_POLICY),
            (450, ai_summarizer.EXTENDED_POLICY),
            (451, ai_summarizer.LONG_POLICY),
            (599.999, ai_summarizer.LONG_POLICY),
            (600, ai_summarizer.ULTRA_POLICY),
        )

        for duration, expected in cases:
            with self.subTest(duration=duration):
                self.assertIs(
                    ai_summarizer.select_generation_policy(duration_seconds=duration),
                    expected,
                )

    def test_text_volume_never_promotes_or_demotes_a_duration_tier(self) -> None:
        for duration in (30, 90, 240, 360, 500, 900):
            with self.subTest(duration=duration):
                sparse = ai_summarizer.select_generation_policy(
                    duration_seconds=duration,
                    transcript_char_count=0,
                    draft_char_count=0,
                )
                dense = ai_summarizer.select_generation_policy(
                    duration_seconds=duration,
                    transcript_char_count=10_000_000,
                    draft_char_count=10_000_000,
                )
                self.assertIs(sparse, dense)
                self.assertEqual(sparse.final_thinking_budget, dense.final_thinking_budget)


class VisualDeltaTests(unittest.IsolatedAsyncioTestCase):
    def test_visual_json_is_validated_deduplicated_and_bounded_by_duration(self) -> None:
        segments = _timestamped_segments()
        raw = json.dumps(
            {
                "schema_version": "stage1.visual-delta.v1",
                "annotations": [
                    {
                        "start_ms": 1_000,
                        "end_ms": 1_200,
                        "anchor_segment_id": "S0001",
                        "kind": "ocr",
                        "confidence": "high",
                        "text": "关键 数字 42",
                        "details_md": "- 左列：旧值\n- 右列：新值",
                        "novelty_reason": "逐字稿没有念出这组数字",
                        "screenshot_recommended": True,
                        "screenshot_reason": "数字清晰可见",
                        "screenshot_ms": 1_100,
                    },
                    {
                        "start_ms": 1_500,
                        "end_ms": 1_600,
                        "anchor_segment_id": "S0001",
                        "kind": "ocr",
                        "confidence": "high",
                        "text": "关键数字42",
                        "novelty_reason": "逐字稿没有念出这组数字",
                    },
                    {
                        "start_ms": 3_000,
                        "end_ms": 3_100,
                        "anchor_segment_id": None,
                        "kind": "visible_fact",
                        "confidence": "low",
                        "text": "低置信度但合法的可见事实",
                        "novelty_reason": "逐字稿没有描述该对象",
                        "screenshot_recommended": True,
                        "screenshot_reason": "低置信度截图不得保留",
                    },
                    {
                        "start_ms": 8_000,
                        "end_ms": 8_500,
                        "anchor_segment_id": "不存在",
                        "kind": "ui_structure",
                        "confidence": "medium",
                        "text": "后段出现三列界面",
                        "novelty_reason": "逐字稿没有说明界面结构",
                    },
                    {
                        "start_ms": -1,
                        "end_ms": 100,
                        "kind": "ocr",
                        "confidence": "high",
                        "text": "负时间",
                    },
                    {
                        "start_ms": 8_000,
                        "end_ms": 7_000,
                        "kind": "ocr",
                        "confidence": "high",
                        "text": "倒置时间",
                    },
                    {
                        "start_ms": 9_000,
                        "end_ms": 9_001,
                        "kind": "summary",
                        "confidence": "high",
                        "text": "非法类型",
                    },
                    {
                        "start_ms": 9_100,
                        "end_ms": 9_200,
                        "kind": "uncertain_inference",
                        "confidence": "high",
                        "text": "不允许的高置信推断",
                    },
                    {
                        "start_ms": 9_500,
                        "end_ms": 10_001,
                        "kind": "ocr",
                        "confidence": "high",
                        "text": "超出视频时长",
                    },
                ],
            },
            ensure_ascii=False,
        )

        annotations = ai_summarizer._validated_visual_annotations(
            raw,
            segments,
            duration_seconds=10,
        )

        self.assertEqual([item.annotation_id for item in annotations], ["V0001", "V0002", "V0003"])
        self.assertEqual([item.start_ms for item in annotations], [1_000, 3_000, 8_000])
        self.assertEqual(annotations[0].text, "关键 数字 42")
        self.assertTrue(annotations[0].screenshot_recommended)
        self.assertEqual(annotations[0].details_md, "- 左列：旧值\n- 右列：新值")
        self.assertFalse(annotations[1].screenshot_recommended)
        self.assertIsNone(annotations[1].screenshot_reason)
        self.assertEqual(annotations[1].anchor_segment_id, "S0001")
        self.assertEqual(annotations[2].anchor_segment_id, "S0002")

    def test_malformed_visual_json_is_rejected(self) -> None:
        segments = _timestamped_segments()
        cases = (
            "not json",
            "[]",
            '{"schema_version":"stage1.visual-delta.v1"}',
            '{"annotations":{}}',
        )
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                ai_summarizer._validated_visual_annotations(raw, segments, 9)

    async def test_visual_model_is_json_only_and_prompt_forbids_summarization(self) -> None:
        callback = AsyncMock()
        segments = _timestamped_segments()
        with patch.object(
            ai_summarizer.aliyun_client,
            "chat",
            new_callable=AsyncMock,
            return_value=(
                "```json\n"
                '{"schema_version":"stage1.visual-delta.v1","annotations":[]}'
                "\n```"
            ),
        ) as chat:
            result = await ai_summarizer._analyze_visual_deltas(
                media_url="https://media.invalid/video.mp4",
                video_title="离线标题",
                video_author="离线作者",
                user_requirement="关注表格",
                duration_seconds=9,
                segments=segments,
                timestamp_quality="sentence_exact",
                callback=callback,
            )

        self.assertEqual(result, ())
        kwargs = chat.await_args.kwargs
        self.assertEqual(kwargs["model"], ai_summarizer.ALIYUN_DRAFT_MODEL)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertIs(kwargs["enable_thinking"], False)
        self.assertIs(kwargs["preserve_thinking"], False)
        system_prompt = kwargs["messages"][0]["content"]
        self.assertIn("你的唯一职责是观察输入视频", system_prompt)
        self.assertIn("你不负责总结、改写、润色", system_prompt)
        self.assertIn("只提取视觉增量", system_prompt)
        self.assertIn("不输出标题、摘要、结论", system_prompt)
        user_parts = kwargs["messages"][1]["content"]
        self.assertEqual(user_parts[0]["type"], "video_url")
        payload = json.loads(user_parts[1]["text"])
        self.assertEqual(
            [item["text"] for item in payload["transcript_segments"]],
            ["第一句原话。", "第二句原话！"],
        )
        callback.assert_not_awaited()

    async def test_enhanced_transcript_preserves_each_sentence_and_inserts_visual(self) -> None:
        transcription = _timestamped_transcription()
        segments = _timestamped_segments()
        annotation = _visual_annotation(anchor_segment_id="S0001")

        with patch.object(
            ai_summarizer,
            "_analyze_visual_deltas",
            new_callable=AsyncMock,
            return_value=(annotation,),
        ):
            result = await ai_summarizer._stage1_result_from_transcription(
                transcription=transcription,
                segments=segments,
                timestamp_quality="sentence_exact",
                local_media_path="/virtual/video.mp4",
                media_url="https://media.invalid/video.mp4",
                video_title="离线标题",
                video_author="离线作者",
                user_requirement="",
                duration_seconds=9,
                callback=None,
            )

        enhanced = result.enhanced_transcript
        self.assertEqual(result.source_transcript, RAW_TRANSCRIPT)
        self.assertEqual(result.transcript_char_count, len(RAW_TRANSCRIPT))
        self.assertEqual(enhanced.count("第一句原话。"), 1)
        self.assertEqual(enhanced.count("第二句原话！"), 1)
        self.assertEqual(enhanced.count("画面新增了一个三列对比表。"), 1)
        self.assertLess(enhanced.index("第一句原话。"), enhanced.index("[视觉增量 V0001"))
        self.assertLess(enhanced.index("[视觉增量 V0001"), enhanced.index("第二句原话！"))


class Stage3WritingPromptTests(unittest.TestCase):
    def test_prompt_converts_parallel_paragraphs_into_semantic_markdown(self) -> None:
        prompt = ai_summarizer.STAGE3_SYSTEM
        self.assertIn("Markdown 信息架构", prompt)
        self.assertIn("三个及以上并列", prompt)
        self.assertIn("项目列表", prompt)
        self.assertIn("编号列表", prompt)
        self.assertIn("用表格", prompt)
        self.assertIn("伪列表段", prompt)
        self.assertIn("少而完整的论证段", prompt)

    def test_screenshot_prompts_require_set_level_deduplication(self) -> None:
        self.assertIn("全局去重", ai_summarizer.SCREENSHOT_REVIEW_SYSTEM)
        self.assertIn("最小非重复图片集合", ai_summarizer.SCREENSHOT_REVIEW_SYSTEM)
        self.assertIn("同一稳定视觉状态最多推荐一张截图", ai_summarizer.STAGE1_SYSTEM)
        self.assertIn("同一二级章节通常最多一张", ai_summarizer.STAGE3_SYSTEM)


class ScreenshotReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_review_keeps_only_supported_frame_and_corrects_caption(self) -> None:
        annotations = (
            ai_summarizer.VisualAnnotation(
                "V0001", 1_000, 2_000, "S0001", "ocr", "high",
                "模型声称画面有一串细字。", screenshot_recommended=True,
                screenshot_ms=1_500, novelty_reason="语音没有念出该文字",
            ),
            ai_summarizer.VisualAnnotation(
                "V0002", 5_000, 6_000, "S0002", "ui_structure", "medium",
                "模型声称画面是三栏界面。", screenshot_recommended=True,
                screenshot_ms=5_500, novelty_reason="语音没有描述布局",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            frames = {}
            for index, annotation in enumerate(annotations, start=1):
                path = Path(directory) / f"video_frame_{annotation.annotation_id}.jpg"
                Image.new("RGB", (64, 48), (40 * index, 80, 120)).save(path, "JPEG")
                paths.append(path)
                frames[annotation.annotation_id] = ExtractedVideoFrame(
                    annotation_id=annotation.annotation_id,
                    timestamp_ms=annotation.screenshot_ms or 0,
                    path=path,
                    kind=annotation.kind,
                    confidence=annotation.confidence,
                    caption=annotation.text,
                    quality_score=9.0,
                )

            review_json = json.dumps(
                {
                    "schema_version": "stage1.frame-review.v1",
                    "reviews": [
                        {
                            "id": "V0001",
                            "decision": "keep",
                            "correspondence": "exact",
                            "readability": "clear",
                            "article_value": "helpful",
                            "caption": "截图中清楚显示数字 42。",
                            "reason": "文字清晰且补充正文",
                        },
                        {
                            "id": "V0002",
                            "decision": "reject",
                            "correspondence": "mismatch",
                            "readability": "usable",
                            "article_value": "decorative",
                            "caption": None,
                            "reason": "截图没有显示所述三栏结构",
                        },
                    ],
                },
                ensure_ascii=False,
            )
            with patch.object(
                ai_summarizer.aliyun_client,
                "chat",
                new_callable=AsyncMock,
                return_value=review_json,
            ) as chat:
                approved = await ai_summarizer._review_extracted_frames(
                    frames,
                    annotations,
                    _timestamped_segments(),
                )

        self.assertEqual(list(approved), ["V0001"])
        self.assertEqual(approved["V0001"].caption, "截图中清楚显示数字 42。")
        kwargs = chat.await_args.kwargs
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertIs(kwargs["enable_thinking"], False)
        image_parts = [part for part in kwargs["messages"][1]["content"] if part["type"] == "image_url"]
        self.assertEqual(len(image_parts), 2)
        self.assertTrue(image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    async def test_review_failure_fails_closed_for_images(self) -> None:
        annotation = ai_summarizer.VisualAnnotation(
            "V0001", 1_000, 2_000, "S0001", "visible_fact", "high",
            "画面显示一个关键实物。", screenshot_recommended=True,
            screenshot_ms=1_500, novelty_reason="语音没有描述实物外观",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_frame_V0001.jpg"
            Image.new("RGB", (32, 32), "white").save(path, "JPEG")
            frame = ExtractedVideoFrame(
                "V0001", 1_500, path, "visible_fact", "high",
                annotation.text, 8.0,
            )
            with patch.object(
                ai_summarizer.aliyun_client,
                "chat",
                new_callable=AsyncMock,
                side_effect=RuntimeError("offline review failure"),
            ):
                approved = await ai_summarizer._review_extracted_frames(
                    {"V0001": frame},
                    (annotation,),
                    _timestamped_segments(),
                )

        self.assertEqual(approved, {})


class StageInputContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_stage2_gets_complete_speech_without_visual_and_with_budget(self) -> None:
        complete_speech = ai_summarizer._build_enhanced_transcript(
            _timestamped_segments(),
            (),
        )
        policy = ai_summarizer.EXTENDED_POLICY

        with patch.object(
            ai_summarizer.aliyun_client,
            "chat",
            new_callable=AsyncMock,
            return_value="<internal_research_memo>离线研究</internal_research_memo>",
        ) as chat:
            result = await ai_summarizer.stage2_deep_research(
                complete_speech,
                policy,
                video_title="离线标题",
                video_author="离线作者",
                user_requirement="关注边界",
                duration_seconds=360,
                transcript_char_count=len(RAW_TRANSCRIPT),
            )

        self.assertIn("离线研究", result)
        kwargs = chat.await_args.kwargs
        payload = json.loads(kwargs["messages"][1]["content"])
        self.assertEqual(
            set(payload),
            {"research_policy", "video_metadata", "complete_audio_transcript"},
        )
        self.assertEqual(payload["complete_audio_transcript"], complete_speech)
        self.assertNotIn("视觉增量", payload["complete_audio_transcript"])
        self.assertFalse(any("visual" in key.lower() for key in payload))
        self.assertEqual(
            payload["research_policy"]["result_body_char_budget"],
            policy.research_body_char_budget,
        )
        self.assertGreater(payload["research_policy"]["result_body_char_budget"], 0)
        self.assertEqual(kwargs["max_tokens"], policy.research_max_output_tokens)
        self.assertIs(kwargs["enable_search"], True)
        self.assertIs(kwargs["enable_thinking"], policy.research_enable_thinking)
        self.assertEqual(kwargs["thinking_budget"], policy.research_thinking_budget)
        self.assertIs(kwargs["preserve_thinking"], False)

    async def test_stage3_every_tier_receives_all_sources_and_its_budget(self) -> None:
        policies = (
            ai_summarizer.MICRO_POLICY,
            ai_summarizer.COMPACT_POLICY,
            ai_summarizer.STANDARD_POLICY,
            ai_summarizer.EXTENDED_POLICY,
            ai_summarizer.LONG_POLICY,
            ai_summarizer.ULTRA_POLICY,
        )
        raw_transcript = "RAW-BEGIN\n" + ("完整逐字原句。" * 40) + "\nRAW-END"
        enhanced_transcript = "[S0001｜约 00:00-00:05]\n完整逐字原句。\n\n> [视觉增量 V0001] 三列图表"
        visual_evidence = "- [V0001｜约 00:01-00:02｜ui_structure｜high] 三列图表"
        research_memo = "<internal_research_memo>必要边界与来源</internal_research_memo>"

        for policy in policies:
            with self.subTest(tier=policy.name), patch.object(
                ai_summarizer.aliyun_client,
                "chat",
                new_callable=AsyncMock,
                return_value="最终笔记",
            ) as chat:
                result = await ai_summarizer.stage3_enrich_and_finalize(
                    enhanced_transcript,
                    research_memo,
                    video_author="离线作者",
                    user_requirement="保留细节",
                    policy=policy,
                    source_transcript=raw_transcript,
                    visual_evidence=visual_evidence,
                    video_title="离线标题",
                    duration_seconds=600,
                )

                self.assertEqual(result, "最终笔记")
                kwargs = chat.await_args.kwargs
                payload = json.loads(kwargs["messages"][1]["content"])
                self.assertEqual(payload["primary_audio_transcript_verbatim"], raw_transcript)
                self.assertEqual(payload["primary_enhanced_transcript"], enhanced_transcript)
                self.assertEqual(payload["primary_visual_evidence"], visual_evidence)
                self.assertEqual(payload["internal_research_memo_do_not_quote"], research_memo)
                self.assertEqual(payload["generation_policy"]["tier"], policy.name)
                self.assertIsNone(kwargs["max_tokens"])
                self.assertEqual(
                    kwargs["max_completion_tokens"],
                    policy.final_max_completion_tokens,
                )
                self.assertIs(kwargs["enable_thinking"], policy.final_enable_thinking)
                self.assertEqual(kwargs["thinking_budget"], policy.final_thinking_budget)
                self.assertIs(kwargs["preserve_thinking"], False)
                system_prompt = kwargs["messages"][0]["content"]
                self.assertIn("适度而清楚的层次", system_prompt)
                self.assertIn("二级标题", system_prompt)
                self.assertIn("三级标题", system_prompt)
                self.assertIn("标题—一句话—标题", system_prompt)


class PipelineOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_success_emits_none_of_the_old_four_stage_messages(self) -> None:
        callback = AsyncMock()
        transcription = _timestamped_transcription()
        memo = "<internal_research_memo>无必要补充</internal_research_memo>"

        async def offline_chat(**kwargs: object) -> str:
            system_prompt = kwargs["messages"][0]["content"]  # type: ignore[index]
            if system_prompt == ai_summarizer.STAGE1_SYSTEM:
                return '{"schema_version":"stage1.visual-delta.v1","annotations":[]}'
            if system_prompt == ai_summarizer.STAGE2_SYSTEM:
                return memo
            if system_prompt == ai_summarizer.STAGE3_SYSTEM:
                return "最终离线笔记"
            raise AssertionError("unexpected offline chat request")

        with patch.object(
            ai_summarizer,
            "_media_duration_or_none",
            new_callable=AsyncMock,
            return_value=9,
        ) as duration_probe, patch.object(
            ai_summarizer,
            "_transcribe_audio_detailed",
            new_callable=AsyncMock,
            return_value=transcription,
        ) as transcribe, patch.object(
            ai_summarizer.aliyun_client,
            "chat",
            new_callable=AsyncMock,
            side_effect=offline_chat,
        ) as chat:
            result = await ai_summarizer.summarize_with_audio(
                "/virtual/video.mp4",
                video_title="离线标题",
                video_author="离线作者",
                user_requirement="保留所有原句",
                progress_callback=callback,
                media_url="https://media.invalid/video.mp4",
            )

        self.assertEqual(result, "最终离线笔记")
        duration_probe.assert_awaited_once_with("/virtual/video.mp4")
        transcribe.assert_awaited_once()
        self.assertEqual(chat.await_count, 3)
        callback.assert_not_awaited()

    async def test_visual_and_research_branches_overlap_after_asr(self) -> None:
        transcription = _timestamped_transcription()
        segments = _timestamped_segments()
        visual_started = asyncio.Event()
        research_started = asyncio.Event()
        trace: list[str] = []

        stage1_result = ai_summarizer.Stage1Result(
            source_transcript=RAW_TRANSCRIPT,
            transcript_char_count=len(RAW_TRANSCRIPT),
            transcript_segments=segments,
            timestamp_quality="sentence_exact",
            visual_annotations=(_visual_annotation(),),
            visual_evidence_markdown="视觉证据",
            enhanced_transcript="完整语音加视觉",
        )

        async def delayed_visual(**kwargs: object) -> ai_summarizer.Stage1Result:
            del kwargs
            trace.append("visual:start")
            visual_started.set()
            await asyncio.wait_for(research_started.wait(), timeout=0.5)
            await asyncio.sleep(0.02)
            trace.append("visual:end")
            return stage1_result

        async def delayed_research(
            transcript: str,
            policy: ai_summarizer.GenerationPolicy,
            **kwargs: object,
        ) -> str:
            del transcript, policy, kwargs
            trace.append("research:start")
            research_started.set()
            await asyncio.wait_for(visual_started.wait(), timeout=0.5)
            await asyncio.sleep(0.02)
            trace.append("research:end")
            return "并行研究"

        with patch.object(
            ai_summarizer,
            "_media_duration_or_none",
            new_callable=AsyncMock,
            return_value=120,
        ), patch.object(
            ai_summarizer,
            "_transcribe_audio_detailed",
            new_callable=AsyncMock,
            return_value=transcription,
        ), patch.object(
            ai_summarizer,
            "_stage1_result_from_transcription",
            new_callable=AsyncMock,
            side_effect=delayed_visual,
        ) as visual_branch, patch.object(
            ai_summarizer,
            "stage2_deep_research",
            new_callable=AsyncMock,
            side_effect=delayed_research,
        ) as research_branch, patch.object(
            ai_summarizer,
            "stage3_enrich_and_finalize",
            new_callable=AsyncMock,
            return_value="并行终稿",
        ) as final_stage:
            result = await ai_summarizer.summarize_with_audio(
                "/virtual/video.mp4",
                media_url="https://media.invalid/video.mp4",
            )

        self.assertEqual(result, "并行终稿")
        self.assertTrue(visual_started.is_set())
        self.assertTrue(research_started.is_set())
        self.assertEqual(set(trace[:2]), {"visual:start", "research:start"})
        self.assertEqual(set(trace[2:]), {"visual:end", "research:end"})
        visual_branch.assert_awaited_once()
        research_branch.assert_awaited_once()
        final_stage.assert_awaited_once()
        self.assertEqual(final_stage.await_args.args[:2], ("完整语音加视觉", "并行研究"))


if __name__ == "__main__":
    unittest.main()
