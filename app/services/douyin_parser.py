"""基于 HTTP 页面数据的抖音视频解析与隔离下载模块。"""
import asyncio
import os
import re
import json
import logging
import shutil
import time
import uuid
import httpx
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode, urlparse
from app.config import TEMP_DIR, TEMP_FILE_TTL_HOURS
from app.services.douyin_abogus import ABogus, BrowserFingerprintGenerator

logger = logging.getLogger(__name__)

DESKTOP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/139.0.0.0 Safari/537.36"
)

# 短链与分享页继续使用移动端 UA；作品详情接口要求桌面端参数与 UA 一致。
MOBILE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) EdgiOS/121.0.2277.107 Version/17.0 Mobile/15E148 Safari/604.1'
}
DOUYIN_WEB_HEADERS = {
    "User-Agent": DESKTOP_USER_AGENT,
    "Referer": "https://www.douyin.com/?recommend=1",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
}
DOUYIN_DOWNLOAD_HEADERS = {
    "User-Agent": DESKTOP_USER_AGENT,
    "Referer": "https://www.douyin.com/",
    "Accept": "*/*",
}
TTWID_REGISTER_URL = "https://ttwid.bytedance.com/ttwid/union/register/"
DOUYIN_DETAIL_URL = "https://www.douyin.com/aweme/v1/web/aweme/detail/"
DETAIL_AID_CANDIDATES = ("6383", "1128")


def extract_url_from_text(text: str) -> Optional[str]:
    """提取抖音分享链接"""
    patterns = [
        r'https?://v\.douyin\.com/[A-Za-z0-9_-]+/?',
        r'https?://www\.douyin\.com/video/\d+',
        r'https?://www\.iesdouyin\.com/share/video/\d+',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            url = match.group(0)
            if 'v.douyin.com' in url and not url.endswith('/'):
                url += '/'
            return url
    return None


def extract_user_requirement(text: str, url: str) -> str:
    """提取用户附加要求 (过滤链接和模板文案)"""
    # A standard Douyin share card puts its caption before the URL.  That
    # caption describes the video; it is not an instruction to the model.
    # Preserve only text the user deliberately placed before the share marker.
    share_marker = "复制打开抖音"
    if share_marker in text:
        remaining = text.split(share_marker, 1)[0]
        remaining = re.sub(r"\d+(?:\.\d+)?\s*$", "", remaining).strip()
    else:
        remaining = text.replace(url, " ").strip()
    cleaners = [
        r'\d+\.?\d*\s+', r'复制.*?打开.*?抖音[，,]?\s*', r'看看[【\[]?[^】\]]*?的作品[】\]]?\s*',
        r'#\s*[^\s#]+\s*', r'@[^\s@]+\s*', r'"[^"]*\.\.\.\s*', 
        r'[A-Za-z]{2,4}:[/\\]\s*', r'[a-zA-Z]@[A-Za-z]\.[A-Z]{2}\s*', r'\d{1,2}/\d{1,2}\s*'
    ]
    for p in cleaners:
        remaining = re.sub(p, ' ', remaining)
    return re.sub(r'\s+', ' ', remaining).strip()


def _extract_video_id(url: str) -> str:
    """Extract a Douyin work ID from a redirected share URL."""
    parsed = urlparse(url)
    path_matches = re.findall(r"(?<!\d)\d{15,22}(?!\d)", parsed.path)
    if path_matches:
        return path_matches[-1]
    query_match = re.search(r"(?:^|[?&])(?:modal_id|vid)=(\d{15,22})(?:&|$)", url)
    return query_match.group(1) if query_match else ""


def _extract_router_item(html: str) -> Optional[dict[str, Any]]:
    """Read an item from the legacy share-page ``_ROUTER_DATA`` payload."""
    match = re.search(
        r"window\._ROUTER_DATA\s*=\s*(.*?)</script>",
        html,
        flags=re.DOTALL,
    )
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
    except (TypeError, json.JSONDecodeError):
        logger.warning("_ROUTER_DATA JSON 解析失败")
        return None

    loader_data = data.get("loaderData") or {}
    for route_name in ("video_(id)/page", "note_(id)/page"):
        route_data = loader_data.get(route_name) or {}
        video_info = route_data.get("videoInfoRes") or {}
        item_list = video_info.get("item_list") or []
        if item_list and isinstance(item_list[0], dict):
            return item_list[0]
    return None


def _normalize_video_url(url: str) -> str:
    normalized = str(url or "").strip()
    if normalized.startswith("//"):
        normalized = "https:" + normalized
    if normalized.startswith("http://"):
        normalized = "https://" + normalized[len("http://") :]
    return normalized.replace("playwm", "play")


def _select_video_url(item: dict[str, Any]) -> str:
    """Select an H.264 MP4 URL, preferring a direct CDN over redirect APIs."""
    video = item.get("video") or {}
    addresses: list[dict[str, Any]] = []

    for key in ("play_addr", "play_addr_h264"):
        address = video.get(key)
        if isinstance(address, dict):
            addresses.append(address)

    h264_rates = [
        rate
        for rate in (video.get("bit_rate") or [])
        if isinstance(rate, dict) and not rate.get("is_h265")
    ]
    h264_rates.sort(
        key=lambda rate: int(
            (rate.get("play_addr") or {}).get("data_size")
            or rate.get("bit_rate")
            or 0
        ),
        reverse=True,
    )
    addresses.extend(
        rate["play_addr"]
        for rate in h264_rates
        if isinstance(rate.get("play_addr"), dict)
    )

    download_address = video.get("download_addr")
    if isinstance(download_address, dict):
        addresses.append(download_address)

    seen: set[str] = set()
    for address in addresses:
        candidates = [
            _normalize_video_url(candidate)
            for candidate in (address.get("url_list") or [])
        ]
        candidates.sort(
            key=lambda candidate: urlparse(candidate).hostname in {
                "www.douyin.com",
                "www.iesdouyin.com",
            }
        )
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if urlparse(candidate).scheme in {"http", "https"}:
                return candidate
    return ""


def _detail_query(video_id: str, aid: str) -> str:
    params = {
        "device_platform": "webapp",
        "aid": aid,
        "channel": "channel_pc_web",
        "update_version_code": "170400",
        "pc_client_type": "1",
        "pc_libra_divert": "Windows",
        "version_code": "290100",
        "version_name": "29.1.0",
        "cookie_enabled": "true",
        "screen_width": "1536",
        "screen_height": "864",
        "browser_language": "zh-CN",
        "browser_platform": "Win32",
        "browser_name": "Chrome",
        "browser_version": "139.0.0.0",
        "browser_online": "true",
        "engine_name": "Blink",
        "engine_version": "139.0.0.0",
        "os_name": "Windows",
        "os_version": "10",
        "cpu_core_num": "16",
        "device_memory": "8",
        "platform": "PC",
        "downlink": "10",
        "effective_type": "4g",
        "round_trip_time": "200",
        "support_h265": "1",
        "support_dash": "0",
        "uifid": "",
        "aweme_id": video_id,
    }
    return urlencode(params)


async def _register_ttwid(client: httpx.AsyncClient) -> str:
    """Create the anonymous visitor cookie required by Douyin's web API."""
    response = await client.post(
        TTWID_REGISTER_URL,
        headers=DOUYIN_WEB_HEADERS,
        json={
            "region": "cn",
            "aid": 1768,
            "needFid": False,
            "service": "www.douyin.com",
            "migrate_info": {"ticket": "", "source": "node"},
            "cbUrlProtocol": "https",
            "union": True,
        },
    )
    response.raise_for_status()
    ttwid = response.cookies.get("ttwid", "").strip()
    if not ttwid:
        raise ValueError("ttwid 注册响应缺少 Cookie")

    # The registration endpoint owns a bytedance.com cookie.  Copy the same
    # visitor value to Douyin so httpx sends it to the detail API.
    client.cookies.set("ttwid", ttwid, domain="www.douyin.com", path="/")
    return ttwid


async def _fetch_aweme_detail(
    client: httpx.AsyncClient,
    video_id: str,
) -> Optional[dict[str, Any]]:
    """Fetch the current signed web-detail payload with anonymous cookies."""
    for attempt in range(1, 3):
        try:
            await _register_ttwid(client)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "ttwid 注册失败: attempt=%s/2 error=%s",
                attempt,
                type(exc).__name__,
            )
            continue

        for aid in DETAIL_AID_CANDIDATES:
            query = _detail_query(video_id, aid)
            signer = ABogus(
                fp=BrowserFingerprintGenerator.generate_fingerprint("Chrome"),
                user_agent=DESKTOP_USER_AGENT,
            )
            signed_query, _, signed_ua, _ = signer.generate_abogus(query)
            try:
                response = await client.get(
                    f"{DOUYIN_DETAIL_URL}?{signed_query}",
                    headers={**DOUYIN_WEB_HEADERS, "User-Agent": signed_ua},
                )
                response.raise_for_status()
                if not response.content:
                    logger.warning(
                        "作品详情接口返回空响应: aid=%s attempt=%s/2",
                        aid,
                        attempt,
                    )
                    continue
                payload = response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                logger.warning(
                    "作品详情接口失败: aid=%s attempt=%s/2 error=%s",
                    aid,
                    attempt,
                    type(exc).__name__,
                )
                continue

            detail = payload.get("aweme_detail") if isinstance(payload, dict) else None
            if not isinstance(detail, dict):
                logger.warning(
                    "作品详情为空: aid=%s status_code=%s",
                    aid,
                    payload.get("status_code") if isinstance(payload, dict) else None,
                )
                continue
            if str(detail.get("aweme_id") or "") != video_id:
                logger.warning("作品详情 ID 不匹配: expected=%s", video_id)
                continue
            return detail
    return None


def _publish_time(item: dict[str, Any]) -> str:
    """ISO-8601 UTC publish time from Douyin's ``create_time`` (unix seconds)."""
    raw = item.get("create_time")
    if isinstance(raw, str) and raw.strip().isdigit():
        raw = int(raw.strip())
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return ""
    if raw > 1e12:  # defensively accept milliseconds
        raw = raw / 1000
    if not (946_684_800 <= raw <= 4_102_444_800):  # 2000-01-01 .. 2100-01-01
        return ""
    return datetime.fromtimestamp(raw, tz=timezone.utc).isoformat()


async def resolve_metadata(share_url: str) -> dict:
    """Resolve title, author, video id and publish time without downloading."""
    video_url, title, author, video_id, published_at = None, "未知标题", "未知作者", "", ""
    async with httpx.AsyncClient(headers=MOBILE_HEADERS, follow_redirects=True, timeout=30) as client:
        resp = await client.get(share_url)
        resp.raise_for_status()
        video_id = _extract_video_id(str(resp.url))
        if not video_id:
            raise ValueError("无法提取视频ID")
        item = _extract_router_item(resp.text)
        if item is None:
            item = await _fetch_aweme_detail(client, video_id)
        if item:
            title = item.get("desc") or title
            author = (item.get("author") or {}).get("nickname") or author
            published_at = _publish_time(item)
            video_url = _select_video_url(item)
    return {
        "video_id": video_id, "title": title, "author": author,
        "published_at": published_at, "video_url": video_url,
    }


def _job_directory(job_id: str) -> Path:
    """Return an isolated, validated temporary directory for one job."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", job_id):
        raise ValueError("非法 job_id")
    base = Path(TEMP_DIR) / "jobs"
    directory = base / job_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


async def resolve_and_download(share_url: str, job_id: Optional[str] = None) -> dict:
    """解析链接并下载视频"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    job_id = job_id or uuid.uuid4().hex
    work_dir = _job_directory(job_id)
    logger.info("解析抖音分享链接: host=%s", urlparse(share_url).hostname or "unknown")

    video_url, title, author, video_id, published_at = None, "未知标题", "未知作者", "", ""

    async with httpx.AsyncClient(headers=MOBILE_HEADERS, follow_redirects=True, timeout=30) as client:
        try:
            # 1. 获取 Video ID
            resp = await client.get(share_url)
            resp.raise_for_status()
            final_url = str(resp.url)
            video_id = _extract_video_id(final_url)
            if not video_id:
                raise ValueError("无法提取视频ID")

            # 2. 优先兼容旧分享页；页面数据降级为空时切换到当前签名接口。
            item = _extract_router_item(resp.text)
            if item is None:
                logger.info("分享页无作品数据，切换到签名详情接口")
                item = await _fetch_aweme_detail(client, video_id)

            if item:
                title = item.get("desc") or title
                author = (item.get("author") or {}).get("nickname") or author
                published_at = _publish_time(item)
                video_url = _select_video_url(item)
            else:
                logger.warning("抖音作品详情中未找到有效视频信息")
                
        except Exception as exc:
            logger.error("抖音页面解析失败: %s", type(exc).__name__)
            raise RuntimeError(f"抖音页面解析失败（{type(exc).__name__}）") from exc

    if not video_url: raise ValueError("无法获取视频地址")

    # 3. 下载视频
    video_path = await _download_video(video_url, video_id, job_id=job_id)

    return {
        "video_id": video_id, "title": title, "author": author,
        "published_at": published_at,
        "video_path": video_path, "video_url": video_url, "job_id": job_id,
        "work_dir": str(work_dir),
    }


async def _download_video(
    video_url: str,
    video_id: str,
    max_retries: int = 3,
    job_id: Optional[str] = None,
) -> str:
    """下载视频文件 (带重试)"""
    job_id = job_id or uuid.uuid4().hex
    work_dir = _job_directory(job_id)
    video_path = work_dir / "video.mp4"
    partial_path = work_dir / "video.mp4.part"
    max_bytes = 500 * 1024 * 1024

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "下载视频: attempt=%s/%s host=%s",
                attempt,
                max_retries,
                urlparse(video_url).hostname or "unknown",
            )
            async with httpx.AsyncClient(headers=DOUYIN_DOWNLOAD_HEADERS, follow_redirects=True, timeout=120) as client:
                async with client.stream("GET", video_url) as resp:
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "").lower()
                    if content_type and not any(t in content_type for t in ("video", "octet-stream")):
                        raise ValueError(f"下载内容类型异常: {content_type}")
                    content_length = resp.headers.get("content-length", "").strip()
                    if content_length.isdigit() and int(content_length) > max_bytes:
                        raise ValueError("视频文件超过 500MB 安全上限")
                    downloaded = 0
                    with open(partial_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=65536):
                            downloaded += len(chunk)
                            if downloaded > max_bytes:
                                raise ValueError("视频文件超过 500MB 安全上限")
                            f.write(chunk)

            if partial_path.stat().st_size < 1024 * 100:
                raise ValueError("下载文件过小，可能不是有效视频")

            os.replace(partial_path, video_path)
            return str(video_path)

        except asyncio.CancelledError:
            partial_path.unlink(missing_ok=True)
            raise
        except ValueError:
            if partial_path.exists():
                partial_path.unlink(missing_ok=True)
            raise
        except (
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.HTTPStatusError,
        ) as e:
            last_error = e
            logger.warning(f"下载失败 (attempt {attempt}/{max_retries}): {e}")
            # 清理不完整的文件
            if partial_path.exists():
                try:
                    partial_path.unlink()
                except OSError:
                    pass
            if attempt < max_retries:
                wait = 2 ** attempt  # 2s, 4s, 8s
                logger.info(f"等待 {wait}s 后重试...")
                await asyncio.sleep(wait)

    if last_error:
        raise RuntimeError(
            f"视频下载失败（{type(last_error).__name__}）"
        ) from last_error
    raise RuntimeError("视频下载失败")


async def extract_audio(video_path: str) -> str:
    """提取音频 (mp3, 16kHz, mono)"""
    audio_path = video_path.rsplit(".", 1)[0] + ".mp3"
    if os.path.exists(audio_path): return audio_path

    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-ab", "128k", "-ar", "16000", "-ac", "1", "-y", audio_path]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError("ffmpeg 提取音频超时")
    except asyncio.CancelledError:
        proc.kill()
        await proc.communicate()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {stderr.decode('utf-8', errors='replace')[:200]}")

    return audio_path


def cleanup_files(job_id: str):
    """仅清理当前任务的隔离目录。"""
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", job_id):
        logger.error("拒绝清理非法 job_id: %r", job_id)
        return
    directory = Path(TEMP_DIR) / "jobs" / job_id
    if directory.is_dir():
        shutil.rmtree(directory, ignore_errors=True)


def cleanup_stale_job_dirs() -> None:
    """Remove abandoned job directories older than the configured TTL."""
    base = Path(TEMP_DIR) / "jobs"
    if not base.is_dir():
        return
    cutoff = time.time() - TEMP_FILE_TTL_HOURS * 3600
    for directory in base.iterdir():
        try:
            if directory.is_dir() and directory.stat().st_mtime < cutoff:
                shutil.rmtree(directory, ignore_errors=True)
        except OSError:
            logger.warning("清理过期任务目录失败: %s", directory)
