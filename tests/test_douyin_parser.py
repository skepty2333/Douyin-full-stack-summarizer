"""Filesystem-isolation tests for per-job Douyin temporary files."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs
from unittest.mock import patch

import httpx


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


class DouyinCurrentParserTests(unittest.TestCase):
    VIDEO_ID = "7672445901280955694"

    def test_extract_video_id_from_current_share_redirect(self) -> None:
        redirected = (
            "https://www.iesdouyin.com/share/video/7672445901280955694/"
            "?region=CN&from_aid=1128"
        )

        self.assertEqual(douyin_parser._extract_video_id(redirected), self.VIDEO_ID)

    def test_empty_router_payload_requests_detail_fallback(self) -> None:
        html = (
            '<script>window._ROUTER_DATA = {"loaderData":{'
            '"video_(id)/page":{"itemId":"7672445901280955694"}}}</script>'
        )

        self.assertIsNone(douyin_parser._extract_router_item(html))

    def test_select_video_url_prefers_direct_cdn(self) -> None:
        item = {
            "video": {
                "play_addr": {
                    "url_list": [
                        "https://www.douyin.com/aweme/v1/play/?video_id=abc",
                        "//v3-dy-o.examplecdn.com/video.mp4",
                    ]
                }
            }
        }

        self.assertEqual(
            douyin_parser._select_video_url(item),
            "https://v3-dy-o.examplecdn.com/video.mp4",
        )


class DouyinSignedDetailTests(unittest.IsolatedAsyncioTestCase):
    async def test_signed_detail_uses_anonymous_ttwid(self) -> None:
        video_id = "7672445901280955694"
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.host)
            if request.url.host == "ttwid.bytedance.com":
                return httpx.Response(
                    200,
                    json={"status_code": 0},
                    headers={"set-cookie": "ttwid=test-visitor; Path=/; HttpOnly"},
                )

            self.assertEqual(request.url.host, "www.douyin.com")
            query = parse_qs(request.url.query.decode("ascii"))
            self.assertEqual(query["aweme_id"], [video_id])
            self.assertIn("a_bogus", query)
            self.assertIn("ttwid=test-visitor", request.headers.get("cookie", ""))
            return httpx.Response(
                200,
                json={
                    "status_code": 0,
                    "aweme_detail": {
                        "aweme_id": video_id,
                        "desc": "测试作品",
                        "author": {"nickname": "测试作者"},
                        "video": {
                            "play_addr": {
                                "url_list": ["https://cdn.example.com/video.mp4"]
                            }
                        },
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            detail = await douyin_parser._fetch_aweme_detail(client, video_id)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["desc"], "测试作品")
        self.assertEqual(calls, ["ttwid.bytedance.com", "www.douyin.com"])


if __name__ == "__main__":
    unittest.main()
