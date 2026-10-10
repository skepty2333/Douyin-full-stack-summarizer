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


class DouyinDetailTests(unittest.IsolatedAsyncioTestCase):
    VIDEO_ID = "7672445901280955694"

    def _detail(self) -> dict:
        return {
            "aweme_id": self.VIDEO_ID,
            "desc": "测试作品",
            "author": {"nickname": "测试作者"},
            "video": {"play_addr": {"url_list": ["https://cdn.example.com/video.mp4"]}},
        }

    async def test_detail_is_an_unsigned_open_douyin_request(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"status_code": 0, "aweme_detail": self._detail()})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            detail = await douyin_parser._fetch_aweme_detail(client, self.VIDEO_ID)

        self.assertEqual(detail["desc"], "测试作品")
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.url.host, "www.douyin.com")
        self.assertEqual(
            parse_qs(request.url.query.decode("ascii")),
            {"aweme_id": [self.VIDEO_ID], "aid": ["6383"]},
        )
        self.assertEqual(request.headers["origin"], "https://open.douyin.com")
        self.assertEqual(request.headers["referer"], "https://open.douyin.com/")
        self.assertNotIn("cookie", request.headers)

    async def test_argus_block_falls_through_to_the_next_aid(self) -> None:
        aids: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            aid = parse_qs(request.url.query.decode("ascii"))["aid"][0]
            aids.append(aid)
            if aid == "6383":
                return httpx.Response(403, text="Blocked by ArgusSecurityPlugin Uifid Not Found")
            return httpx.Response(200, json={"status_code": 0, "aweme_detail": self._detail()})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertLogs(douyin_parser.logger, "WARNING") as logs:
                detail = await douyin_parser._fetch_aweme_detail(client, self.VIDEO_ID)

        self.assertEqual(detail["aweme_id"], self.VIDEO_ID)
        self.assertEqual(aids, ["6383", "1128"])
        self.assertIn("Uifid Not Found", logs.output[0])


class PublishTimeTests(unittest.TestCase):
    def test_unix_seconds_and_strings_become_iso_utc(self) -> None:
        self.assertEqual(douyin_parser._publish_time({"create_time": 1786265237}), "2026-08-09T08:47:17+00:00")
        self.assertEqual(douyin_parser._publish_time({"create_time": "1786265237"}), "2026-08-09T08:47:17+00:00")
        self.assertEqual(douyin_parser._publish_time({"create_time": 1786265237000}), "2026-08-09T08:47:17+00:00")

    def test_missing_or_implausible_values_yield_empty(self) -> None:
        for value in (None, True, 5, -1, "soon", 99999999999999):
            self.assertEqual(douyin_parser._publish_time({"create_time": value}), "", repr(value))
        self.assertEqual(douyin_parser._publish_time({}), "")


if __name__ == "__main__":
    unittest.main()


def _album_image(uri: str, *, with_clip: bool = False) -> dict:
    """Mirror one ``images[]`` entry from the web detail API for a 图文 post."""
    entry = {
        "uri": f"tos-cn-i-dy/{uri}",
        "width": 1440,
        "height": 2036,
        "resolution_log_param": {},
        "url_list": [
            f"https://p3-pc-sign.douyinpic.com/tos-cn-i-dy/{uri}~tplv-dy-aweme-images-v2:1440:2036:q75.webp?x-expires=1",
            f"https://p3-pc-sign.douyinpic.com/tos-cn-i-dy/{uri}~tplv-dy-aweme-images-v2:1440:2036:q75.jpeg?x-expires=1",
        ],
        "download_url_list": [
            f"https://p3-pc-sign.douyinpic.com/tos-cn-i-dy/{uri}~tplv-dy-water-v2:4762:6735.webp?x-expires=1",
        ],
    }
    if with_clip:
        entry["video"] = {"play_addr": {"url_list": ["https://cdn.example.com/clip.mp4"]}}
    return entry


class DouyinImageAlbumSelectionTests(unittest.TestCase):
    def test_album_selects_unwatermarked_jpeg_per_image_in_order(self) -> None:
        item = {
            "aweme_type": 68,
            "media_type": 2,
            "images": [_album_image("aaa"), _album_image("bbb")],
            "video": {"play_addr": {"url_list": ["https://sf3-sign.douyinstatic.com/bgm"]}},
        }

        self.assertEqual(
            douyin_parser._select_image_urls(item),
            [
                "https://p3-pc-sign.douyinpic.com/tos-cn-i-dy/aaa~tplv-dy-aweme-images-v2:1440:2036:q75.jpeg?x-expires=1",
                "https://p3-pc-sign.douyinpic.com/tos-cn-i-dy/bbb~tplv-dy-aweme-images-v2:1440:2036:q75.jpeg?x-expires=1",
            ],
        )

    def test_album_without_jpeg_variant_falls_back_to_first_https_url(self) -> None:
        image = _album_image("ccc")
        image["url_list"] = ["//p3-pc-sign.douyinpic.com/ccc.webp", "https://p9-pc-sign.douyinpic.com/ccc.webp"]
        item = {"aweme_type": 68, "images": [image]}

        self.assertEqual(
            douyin_parser._select_image_urls(item),
            ["https://p3-pc-sign.douyinpic.com/ccc.webp"],
        )

    def test_album_skips_clip_entries_and_caps_image_count(self) -> None:
        images = [_album_image(f"img{index:02d}") for index in range(40)]
        images[1] = _album_image("clip", with_clip=True)
        item = {"aweme_type": 68, "images": images}

        selected = douyin_parser._select_image_urls(item)

        self.assertEqual(len(selected), 35)
        self.assertNotIn("clip", "".join(selected))
        self.assertIn("img00", selected[0])
        self.assertIn("img02", selected[1])

    def test_video_item_yields_no_album_images(self) -> None:
        item = {
            "aweme_type": 0,
            "images": None,
            "video": {"play_addr": {"url_list": ["https://cdn.example.com/video.mp4"]}},
        }

        self.assertEqual(douyin_parser._select_image_urls(item), [])
        self.assertFalse(douyin_parser._is_image_album(item))

    def test_album_is_detected_by_type_or_by_images(self) -> None:
        self.assertTrue(douyin_parser._is_image_album({"aweme_type": 68, "images": [_album_image("a")]}))
        self.assertTrue(douyin_parser._is_image_album({"aweme_type": 2, "images": [_album_image("a")]}))
        self.assertTrue(douyin_parser._is_image_album({"aweme_type": 0, "images": [_album_image("a")]}))
        self.assertFalse(douyin_parser._is_image_album({"aweme_type": 68, "images": []}))


class DouyinNoteLinkTests(unittest.TestCase):
    def test_desktop_and_share_note_links_are_recognised(self) -> None:
        self.assertEqual(
            douyin_parser.extract_url_from_text("看这个 https://www.douyin.com/note/7687268235284090154 图文"),
            "https://www.douyin.com/note/7687268235284090154",
        )
        self.assertEqual(
            douyin_parser.extract_url_from_text(
                "https://www.iesdouyin.com/share/note/7687268235284090154/?region=CN"
            ),
            "https://www.iesdouyin.com/share/note/7687268235284090154",
        )


def _jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int] = (64, 96)) -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _webp_bytes(color: tuple[int, int, int]) -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (64, 96), color).save(buffer, format="WEBP")
    return buffer.getvalue()


class DouyinAlbumDownloadTests(unittest.IsolatedAsyncioTestCase):
    NOTE_ID = "7687268235284090154"
    BGM_URL = "https://sf3-sign.douyinstatic.com/tos-cn-ve-2774/bgm?x-expires=1"

    def _album_detail(self, images: list[dict]) -> dict:
        return {
            "aweme_id": self.NOTE_ID,
            "aweme_type": 68,
            "media_type": 2,
            "desc": "量化策略思路分享。摘要 交易的直接问题不是“现在像不像上涨状态”。",
            "create_time": 1789900000,
            "author": {"nickname": "新用户1401"},
            "images": images,
            "video": {"play_addr": {"url_list": [self.BGM_URL]}, "duration": 0},
            "music": {"title": "Chemtrails Over The Country Club", "is_original_sound": False},
        }

    def _transport(
        self,
        detail: dict,
        image_responses: dict[str, httpx.Response],
        requested: list[str],
    ) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            requested.append(url)
            if request.url.host == "v.douyin.com":
                return httpx.Response(
                    302,
                    headers={
                        "location": f"https://www.iesdouyin.com/share/note/{self.NOTE_ID}/?region=CN"
                    },
                )
            if request.url.host == "www.iesdouyin.com":
                return httpx.Response(200, text="<html><body>no router data</body></html>")
            if request.url.host == "www.douyin.com":
                return httpx.Response(200, json={"status_code": 0, "aweme_detail": detail})
            for prefix, response in image_responses.items():
                if url.startswith(prefix):
                    return response
            return httpx.Response(404)

        return httpx.MockTransport(handler)

    async def _resolve(self, transport: httpx.MockTransport, temp_dir: str) -> dict:
        import functools

        client_factory = functools.partial(httpx.AsyncClient, transport=transport)
        with patch.object(douyin_parser, "TEMP_DIR", temp_dir), patch.object(
            douyin_parser.httpx, "AsyncClient", client_factory
        ):
            return await douyin_parser.resolve_and_download(
                "https://v.douyin.com/VNv0Dv0R5VA/", "job_album_0001"
            )

    async def test_album_downloads_every_picture_and_never_touches_the_bgm(self) -> None:
        detail = self._album_detail([_album_image("aaa"), _album_image("bbb")])
        requested: list[str] = []
        transport = self._transport(
            detail,
            {
                "https://p3-pc-sign.douyinpic.com/tos-cn-i-dy/aaa": httpx.Response(
                    200, content=_jpeg_bytes((200, 0, 0)), headers={"content-type": "image/jpeg"}
                ),
                "https://p3-pc-sign.douyinpic.com/tos-cn-i-dy/bbb": httpx.Response(
                    200, content=_webp_bytes((0, 0, 200)), headers={"content-type": "image/webp"}
                ),
            },
            requested,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = await self._resolve(transport, temp_dir)

            self.assertEqual(result["media_kind"], "images")
            self.assertEqual(result["video_id"], self.NOTE_ID)
            self.assertEqual(result["title"], detail["desc"])
            self.assertEqual(result["caption"], detail["desc"])
            self.assertEqual(result["author"], "新用户1401")
            self.assertEqual(result["published_at"], "2026-09-20T10:26:40+00:00")  # date -u -d @1789900000
            self.assertEqual(result["video_path"], "")
            self.assertEqual(result["video_url"], "")
            self.assertEqual(result["job_id"], "job_album_0001")
            self.assertEqual(len(result["image_paths"]), 2)
            work_dir = Path(result["work_dir"])
            for path in result["image_paths"]:
                self.assertTrue(Path(path).is_relative_to(work_dir))
                self.assertGreater(Path(path).stat().st_size, 0)
            self.assertTrue(Path(result["image_paths"][0]).read_bytes().startswith(b"\xff\xd8\xff"))
            self.assertTrue(Path(result["image_paths"][1]).read_bytes().startswith(b"RIFF"))
            self.assertNotIn(self.BGM_URL, requested)

    async def test_album_keeps_readable_pictures_when_one_download_fails(self) -> None:
        detail = self._album_detail([_album_image("aaa"), _album_image("bbb"), _album_image("ccc")])
        transport = self._transport(
            detail,
            {
                "https://p3-pc-sign.douyinpic.com/tos-cn-i-dy/aaa": httpx.Response(404),
                "https://p3-pc-sign.douyinpic.com/tos-cn-i-dy/bbb": httpx.Response(
                    200, content=b"<html>blocked</html>", headers={"content-type": "text/html"}
                ),
                "https://p3-pc-sign.douyinpic.com/tos-cn-i-dy/ccc": httpx.Response(
                    200, content=_jpeg_bytes((0, 200, 0)), headers={"content-type": "image/jpeg"}
                ),
            },
            [],
        )

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            douyin_parser.asyncio, "sleep", return_value=None
        ):
            result = await self._resolve(transport, temp_dir)

            self.assertEqual(result["media_kind"], "images")
            self.assertEqual(len(result["image_paths"]), 1)
            self.assertTrue(Path(result["image_paths"][0]).read_bytes().startswith(b"\xff\xd8\xff"))

    async def test_album_with_no_readable_picture_is_an_error(self) -> None:
        detail = self._album_detail([_album_image("aaa")])
        transport = self._transport(
            detail,
            {
                "https://p3-pc-sign.douyinpic.com/tos-cn-i-dy/aaa": httpx.Response(
                    200, content=b"not an image at all", headers={"content-type": "image/jpeg"}
                ),
            },
            [],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "图文"):
                await self._resolve(transport, temp_dir)

    async def test_video_result_declares_its_media_kind(self) -> None:
        detail = {
            "aweme_id": self.NOTE_ID,
            "aweme_type": 0,
            "desc": "普通视频",
            "author": {"nickname": "作者"},
            "images": None,
            "video": {"play_addr": {"url_list": ["https://cdn.example.com/video.mp4"]}},
        }
        transport = self._transport(
            detail,
            {
                "https://cdn.example.com/video.mp4": httpx.Response(
                    200, content=b"\x00" * (120 * 1024), headers={"content-type": "video/mp4"}
                ),
            },
            [],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = await self._resolve(transport, temp_dir)

            self.assertEqual(result["media_kind"], "video")
            self.assertEqual(result["image_paths"], [])
            self.assertEqual(result["caption"], "普通视频")
            self.assertTrue(result["video_path"].endswith("video.mp4"))


class DouyinPublicMediaUrlTests(unittest.IsolatedAsyncioTestCase):
    """Bailian fetches media_url without Douyin's Referer (FILE_403_FORBIDDEN)."""

    VIDEO_ID = "7677941996337833381"
    GATED_URL = "https://v3-web.douyinvod.com/sig/video/tos/cn/abc/?a=6383"
    PLAY_API_URL = "https://www.douyin.com/aweme/v1/play/?video_id=abc&line=0"
    PUBLIC_URL = "https://v3-dy-o.zjcdn.com/sig/video/tos/cn/abc/?a=6383"

    def _transport(self, *, play_api_works: bool) -> httpx.MockTransport:
        detail = {
            "aweme_id": self.VIDEO_ID,
            "desc": "DeepSeek harness 手机控制",
            "author": {"nickname": "小7说AI"},
            "video": {"play_addr": {"url_list": [self.PLAY_API_URL, self.GATED_URL]}},
        }
        video = httpx.Response(
            200, content=b"\x00" * (120 * 1024), headers={"content-type": "video/mp4"}
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "v.douyin.com":
                return httpx.Response(
                    302,
                    headers={"location": f"https://www.iesdouyin.com/share/video/{self.VIDEO_ID}/"},
                )
            if request.url.host == "www.iesdouyin.com":
                return httpx.Response(200, text="<html>no router data</html>")
            if request.url.host == "www.douyin.com" and request.url.path == "/aweme/v1/play/":
                if not play_api_works:
                    return httpx.Response(403)
                return httpx.Response(302, headers={"location": self.PUBLIC_URL})
            if request.url.host == "www.douyin.com":
                return httpx.Response(200, json={"status_code": 0, "aweme_detail": detail})
            if request.url.host == "v3-web.douyinvod.com":
                if "douyin.com" not in request.headers.get("referer", ""):
                    return httpx.Response(403, headers={"content-type": "video/mp4"})
                return video
            if request.url.host == "v3-dy-o.zjcdn.com":
                return video
            return httpx.Response(404)

        return httpx.MockTransport(handler)

    async def _resolve(self, transport: httpx.MockTransport, temp_dir: str) -> dict:
        import functools

        client_factory = functools.partial(httpx.AsyncClient, transport=transport)
        with patch.object(douyin_parser, "TEMP_DIR", temp_dir), patch.object(
            douyin_parser.httpx, "AsyncClient", client_factory
        ):
            return await douyin_parser.resolve_and_download(
                "https://v.douyin.com/iwlrcdZ4EuQ/", "job_gated_0001"
            )

    async def test_referer_gated_cdn_hands_bailian_the_redirect_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await self._resolve(self._transport(play_api_works=True), temp_dir)

            self.assertEqual(result["video_url"], self.GATED_URL)
            self.assertTrue(Path(result["video_path"]).is_file())
            self.assertEqual(result["media_url"], self.PUBLIC_URL)

    async def test_no_public_candidate_leaves_media_url_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await self._resolve(self._transport(play_api_works=False), temp_dir)

            self.assertTrue(Path(result["video_path"]).is_file())
            self.assertEqual(result["media_url"], "")
