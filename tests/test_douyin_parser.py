"""Filesystem-isolation tests for per-job Douyin temporary files."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-key"}, clear=False):
    with patch("dotenv.load_dotenv", return_value=False):
        from app.services import douyin_parser


class DouyinParserJobDirectoryTests(unittest.TestCase):
    def test_job_directories_are_isolated_and_cleanup_removes_only_target(self) -> None:
        first_job_id = "job_alpha_001"
        second_job_id = "job_beta_002"

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(douyin_parser, "TEMP_DIR", temp_dir):
                first_dir = douyin_parser._job_directory(first_job_id)
                second_dir = douyin_parser._job_directory(second_job_id)
                first_video = first_dir / "video.mp4"
                second_video = second_dir / "video.mp4"
                root_marker = Path(temp_dir) / "jobs" / "keep.txt"
                first_video.write_bytes(b"first job")
                second_video.write_bytes(b"second job")
                root_marker.write_text("keep", encoding="utf-8")

                self.assertNotEqual(first_dir, second_dir)
                self.assertEqual(first_dir.parent, second_dir.parent)
                self.assertEqual(first_video.read_bytes(), b"first job")
                self.assertEqual(second_video.read_bytes(), b"second job")

                douyin_parser.cleanup_files(first_job_id)

                self.assertFalse(first_dir.exists())
                self.assertTrue(second_dir.is_dir())
                self.assertEqual(second_video.read_bytes(), b"second job")
                self.assertEqual(root_marker.read_text(encoding="utf-8"), "keep")


class DouyinShareRequirementTests(unittest.TestCase):
    def test_standard_share_card_caption_is_not_treated_as_user_requirement(self) -> None:
        text = (
            "6.15 复制打开抖音，看看【大师的AI小灶的作品】"
            "不要再用对话管理codex了，我开发了一个任务管理... "
            "https://v.douyin.com/QgguTQ5xM2A/ 05/30 ipD:/ :7pm J@I.vF"
        )
        url = douyin_parser.extract_url_from_text(text)

        self.assertEqual(url, "https://v.douyin.com/QgguTQ5xM2A/")
        self.assertEqual(douyin_parser.extract_user_requirement(text, url), "")

    def test_explicit_text_before_share_card_is_preserved_as_requirement(self) -> None:
        text = (
            "请重点整理任务管理方法，忽略广告。 6.15 "
            "复制打开抖音，看看【大师的AI小灶的作品】... "
            "https://v.douyin.com/QgguTQ5xM2A/ 05/30 ipD:/"
        )
        url = douyin_parser.extract_url_from_text(text)

        self.assertEqual(
            douyin_parser.extract_user_requirement(text, url),
            "请重点整理任务管理方法，忽略广告。",
        )


if __name__ == "__main__":
    unittest.main()
