"""Offline tests for safe video evidence-frame extraction and embedding."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageStat

from app.services import video_frames


def _annotation(
    annotation_id: str,
    *,
    start_ms: int = 1000,
    end_ms: int = 2000,
    kind: str = "ui_structure",
    confidence: object = "high",
    text: str = "界面显示项目任务列表与当前状态",
) -> dict[str, object]:
    return {
        "id": annotation_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "kind": kind,
        "confidence": confidence,
        "text": text,
        "screenshot_recommended": True,
    }


def _save_test_jpeg(path: Path, *, detailed: bool = True) -> None:
    image = Image.new("RGB", (320, 180), (128, 128, 128))
    if detailed:
        draw = ImageDraw.Draw(image)
        for x in range(0, image.width, 12):
            colour = (20, 20, 20) if (x // 12) % 2 else (235, 235, 235)
            draw.rectangle((x, 0, min(x + 5, image.width - 1), image.height - 1), fill=colour)
        draw.rectangle((40, 45, 280, 135), outline=(220, 30, 30), width=4)
    image.save(path, format="JPEG", quality=88)


class AnnotationSelectionTests(unittest.TestCase):
    def test_filters_unworthy_or_unsafe_entries_and_ranks_useful_kinds(self) -> None:
        entries = [
            _annotation("V0001", kind="visible_fact", confidence="medium"),
            _annotation("../../x", kind="state_change"),
            _annotation("V0002", kind="ocr", confidence=0.70, text="屏幕文字显示申请已经通过"),
            _annotation("V0003", kind="state_change", text="操作后页面从失败状态切换为成功状态"),
            _annotation("V0004", kind="visible_fact", text="图表中的红线明显高于蓝线"),
            _annotation("V0005", kind="ui_structure", text="画面"),
            _annotation("V0006", start_ms=9000, end_ms=11_000),
            _annotation("V0003", kind="state_change", text="重复 ID 不应覆盖第一条"),
        ]

        selected = video_frames.select_frame_annotations(
            entries,
            duration_ms=10_000,
            max_frames=3,
        )

        self.assertEqual([item.id for item in selected], ["V0002", "V0003", "V0004"])
        self.assertEqual(selected[0].confidence, "medium")

    def test_hard_frame_limit_cannot_be_overridden(self) -> None:
        selected = video_frames.select_frame_annotations(
            [
                _annotation(
                    f"V{index:04d}",
                    start_ms=index * 100,
                    end_ms=index * 100 + 50,
                    kind="state_change",
                    text=f"第 {index} 个明确的界面状态变化",
                )
                for index in range(1, 20)
            ],
            duration_ms=10_000,
            max_frames=10_000,
        )

        self.assertEqual(len(selected), video_frames.HARD_MAX_FRAMES)

    def test_bad_duration_and_bad_limit_fail_closed(self) -> None:
        self.assertEqual(
            video_frames.select_frame_annotations(
                [_annotation("V0001")],
                duration_ms=-1,
            ),
            [],
        )

    def test_mapping_requires_literal_screenshot_recommendation(self) -> None:
        not_recommended = _annotation("V0001")
        not_recommended["screenshot_recommended"] = False
        truthy_string = _annotation("V0002")
        truthy_string["screenshot_recommended"] = "true"

        selected = video_frames.select_frame_annotations(
            [not_recommended, truthy_string],
            duration_ms=10_000,
        )

        self.assertEqual(selected, [])
        self.assertEqual(
            video_frames.select_frame_annotations(
                [_annotation("V0001")],
                duration_ms=10_000,
                max_frames=True,
            ),
            [],
        )


class FrameQualityTests(unittest.TestCase):
    def test_detail_and_normal_exposure_beat_a_flat_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            flat = temp_dir / "flat.jpg"
            detailed = temp_dir / "detailed.jpg"
            _save_test_jpeg(flat, detailed=False)
            _save_test_jpeg(detailed, detailed=True)

            self.assertGreater(
                video_frames._score_candidate(detailed),
                video_frames._score_candidate(flat),
            )

    def test_candidate_timestamps_stay_inside_video_at_final_boundary(self) -> None:
        annotation = video_frames.VisualAnnotation(
            id="V0001",
            start_ms=9990,
            end_ms=10_000,
            kind="state_change",
            confidence="high",
            text="视频结束前页面显示明确的完成状态",
        )
        timestamps = video_frames._candidate_timestamps(annotation, duration_ms=10_000)
        self.assertEqual(timestamps, [9950])

    def test_tall_social_frame_is_cropped_to_square_evidence_region(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            source = directory / "source.jpg"
            destination = directory / "video_frame_V0001.jpg"
            image = Image.new("RGB", (120, 260), "black")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 75, 119, 185), fill=(180, 180, 180))
            for y in range(80, 181, 10):
                draw.line((0, y, 119, y), fill="white", width=2)
            image.save(source, "JPEG")

            video_frames._write_sanitized_jpeg(source, destination)

            with Image.open(destination) as cropped:
                self.assertEqual(cropped.size, (120, 120))
                self.assertGreater(sum(ImageStat.Stat(cropped).mean), 80)

    def test_near_duplicate_frames_keep_only_the_clearer_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()

            def make_frame(
                annotation_id: str,
                timestamp_ms: int,
                quality: float,
                *,
                horizontal: bool,
            ) -> video_frames.ExtractedVideoFrame:
                path = directory / f"video_frame_{annotation_id}.jpg"
                image = Image.new("RGB", (160, 160), "black")
                draw = ImageDraw.Draw(image)
                if horizontal:
                    draw.rectangle((0, 0, 159, 79), fill="white")
                else:
                    draw.rectangle((0, 0, 79, 159), fill="white")
                image.save(path, "JPEG", quality=90)
                return video_frames.ExtractedVideoFrame(
                    annotation_id=annotation_id,
                    timestamp_ms=timestamp_ms,
                    path=path,
                    kind="ocr",
                    confidence="high",
                    caption="关键图表",
                    quality_score=quality,
                )

            first = make_frame("V0001", 1_000, 9.0, horizontal=False)
            duplicate = make_frame("V0002", 5_000, 8.0, horizontal=False)
            distinct = make_frame("V0003", 20_000, 7.0, horizontal=True)

            result = video_frames._deduplicate_similar_frames(
                {"V0001": first, "V0002": duplicate, "V0003": distinct}
            )

        self.assertEqual(list(result), ["V0001", "V0003"])


class ExtractionTests(unittest.TestCase):
    def test_extracts_best_candidate_to_fixed_safe_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            job_dir = Path(temp_name).resolve()
            video = job_dir / "video.mp4"
            video.write_bytes(b"offline video placeholder")
            requested_timestamps: list[int] = []

            def fake_extract(_source: Path, timestamp_ms: int, destination: Path) -> bool:
                requested_timestamps.append(timestamp_ms)
                # The centre is intentionally flat; a neighbouring sample should win.
                _save_test_jpeg(destination, detailed=timestamp_ms != 1500)
                return True

            with patch.object(video_frames, "_probe_video_duration_ms", return_value=10_000), patch.object(
                video_frames,
                "_run_ffmpeg_frame_extract",
                side_effect=fake_extract,
            ):
                result = video_frames.extract_video_frames(
                    video,
                    [_annotation("V0001")],
                    allowed_video_root=job_dir,
                )

            self.assertEqual(set(result), {"V0001"})
            frame = result["V0001"]
            self.assertEqual(frame.path.name, "video_frame_V0001.jpg")
            self.assertEqual(frame.path.parent, job_dir / "video_frames")
            self.assertIn(frame.timestamp_ms, (1167, 1833))
            self.assertEqual(requested_timestamps, [1500, 1167, 1833])
            self.assertTrue(frame.path.is_file())
            self.assertEqual(os.stat(frame.path).st_mode & 0o777, 0o600)
            with Image.open(frame.path) as image:
                self.assertEqual(image.format, "JPEG")

    def test_path_and_probe_failures_are_nonfatal(self) -> None:
        self.assertEqual(
            video_frames.extract_video_frames(
                "relative/video.mp4",
                [_annotation("V0001")],
            ),
            {},
        )
        with tempfile.TemporaryDirectory() as temp_name:
            video = Path(temp_name).resolve() / "video.mp4"
            video.write_bytes(b"not a real video")
            with patch.object(video_frames, "_probe_video_duration_ms", side_effect=ValueError("bad")):
                self.assertEqual(
                    video_frames.extract_video_frames(video, [_annotation("V0001")]),
                    {},
                )

    def test_output_cannot_escape_video_job_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name, tempfile.TemporaryDirectory() as other_name:
            job_dir = Path(temp_name).resolve()
            video = job_dir / "video.mp4"
            video.write_bytes(b"offline video placeholder")
            with patch.object(video_frames, "_probe_video_duration_ms", return_value=10_000):
                result = video_frames.extract_video_frames(
                    video,
                    [_annotation("V0001")],
                    output_dir=Path(other_name).resolve(),
                )
            self.assertEqual(result, {})


class MarkerRenderingTests(unittest.TestCase):
    def _frame(self, directory: Path, annotation_id: str, caption: str = "关键界面") -> video_frames.ExtractedVideoFrame:
        path = directory / f"video_frame_{annotation_id}.jpg"
        _save_test_jpeg(path)
        return video_frames.ExtractedVideoFrame(
            annotation_id=annotation_id,
            timestamp_ms=65_000,
            path=path,
            kind="ui_structure",
            confidence="high",
            caption=caption,
            quality_score=10.0,
        )

    def test_marker_becomes_inline_jpeg_html_once_and_is_escaped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            frame = self._frame(directory, "V0001", '按钮 <完成> "已启用"')
            rendered = video_frames.render_video_frame_markers_for_pdf(
                "正文\n\n[[VIDEO_FRAME:V0001]]\n\n重复 [[VIDEO_FRAME:V0001]]",
                {"V0001": frame},
            )

        self.assertEqual(rendered.count("data:image/jpeg;base64,"), 1)
        self.assertNotIn("[[VIDEO_FRAME:", rendered)
        self.assertIn("视频画面 01:05", rendered)
        self.assertIn("&lt;完成&gt;", rendered)
        self.assertIn("&quot;已启用&quot;", rendered)

    def test_missing_malformed_or_wrong_filename_markers_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            wrong_path = directory / "not_fixed.jpg"
            _save_test_jpeg(wrong_path)
            forged = video_frames.ExtractedVideoFrame(
                annotation_id="V0001",
                timestamp_ms=1000,
                path=wrong_path,
                kind="ui_structure",
                confidence="high",
                caption="关键界面信息",
                quality_score=1.0,
            )
            rendered = video_frames.render_video_frame_markers_for_pdf(
                "[[VIDEO_FRAME:V0001]] [[VIDEO_FRAME:../../etc/passwd]] [[VIDEO_FRAME:V9999]]",
                {"V0001": forged},
            )

        self.assertNotIn("VIDEO_FRAME", rendered)
        self.assertNotIn("data:image", rendered)

    def test_total_image_budget_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            frame = self._frame(directory, "V0001")
            rendered = video_frames.render_video_frame_markers_for_pdf(
                "before [[VIDEO_FRAME:V0001]] after",
                [frame],
                max_total_embedded_bytes=1,
            )
        self.assertEqual(rendered, "before  after")

    def test_storage_cleanup_removes_valid_and_malformed_markers(self) -> None:
        cleaned = video_frames.strip_video_frame_markers_for_storage(
            "第一段\n\n[[VIDEO_FRAME:V0001]]\n\n\n[[VIDEO_FRAME:bad/id]]\n\n第二段"
        )
        self.assertEqual(cleaned, "第一段\n\n第二段")

    def test_storage_markers_become_persistent_ai_readable_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name).resolve()
            job_dir = root / "job"
            asset_root = root / "assets"
            job_dir.mkdir()
            frame = self._frame(job_dir, "V0001", "任务看板 [进行中]\\下一步")

            markdown, assets = video_frames.persist_video_frame_markers_for_storage(
                "正文\n\n[[VIDEO_FRAME:V0001]]\n\n重复 [[VIDEO_FRAME:V0001]]",
                {"V0001": frame},
                video_code="A0001",
                asset_root=asset_root,
            )
            frame.path.unlink()

            self.assertEqual(len(assets), 1)
            asset = assets[0]
            persisted_path = asset_root / asset.relative_path
            self.assertTrue(persisted_path.is_file())
            self.assertEqual(os.stat(persisted_path).st_mode & 0o777, 0o600)
            self.assertIn("knowledge-asset://A0001/V0001", markdown)
            self.assertIn("任务看板 \\[进行中\\]", markdown)
            self.assertNotIn("[[VIDEO_FRAME", markdown)
            self.assertEqual(markdown.count("knowledge-asset://"), 1)

    def test_storage_persistence_fails_closed_for_invalid_video_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name).resolve()
            frame = self._frame(directory, "V0001")
            markdown, assets = video_frames.persist_video_frame_markers_for_storage(
                "前文 [[VIDEO_FRAME:V0001]] 后文",
                [frame],
                video_code="../../bad",
                asset_root=directory / "assets",
            )

        self.assertEqual(markdown, "前文  后文")
        self.assertEqual(assets, ())


if __name__ == "__main__":
    unittest.main()
