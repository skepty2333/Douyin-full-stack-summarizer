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
from typing import Optional
from urllib.parse import urlparse
from app.config import TEMP_DIR, TEMP_FILE_TTL_HOURS

logger = logging.getLogger(__name__)

# 模拟移动端 UA
MOBILE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) EdgiOS/121.0.2277.107 Version/17.0 Mobile/15E148 Safari/604.1'
}


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

    video_url, title, author, video_id = None, "未知标题", "未知作者", ""

    async with httpx.AsyncClient(headers=MOBILE_HEADERS, follow_redirects=True, timeout=30) as client:
        try:
            # 1. 获取 Video ID
            resp = await client.get(share_url)
            resp.raise_for_status()
            final_url = str(resp.url)
            path = final_url.split('?')[0]
            video_id = path.split('/')[-1]
            if not video_id.isdigit():
                 ids = re.findall(r'\d{19}', path)
                 if ids: video_id = ids[0]
            
            if not video_id: raise ValueError("无法提取视频ID")
            
            # 2. 请求分享页获取 _ROUTER_DATA
            ies_url = f'https://www.iesdouyin.com/share/video/{video_id}'
            resp = await client.get(ies_url)
            resp.raise_for_status()
            html = resp.text
            
            pattern = re.compile(r"window\._ROUTER_DATA\s*=\s*(.*?)</script>", re.DOTALL)
            match = pattern.search(html)
            
            if match:
                data = json.loads(match.group(1).strip())
                loader_data = data.get("loaderData", {})
                video_info = loader_data.get("video_(id)/page", {}).get("videoInfoRes") or \
                             loader_data.get("note_(id)/page", {}).get("videoInfoRes")
                
                if video_info and "item_list" in video_info and video_info["item_list"]:
                    item = video_info["item_list"][0]
                    title = item.get("desc", title)
                    author = item.get("author", {}).get("nickname", author)
                    
                    if "video" in item and "play_addr" in item["video"]:
                        url_list = item["video"]["play_addr"]["url_list"]
                        if url_list:
                             video_url = url_list[0].replace("playwm", "play")
                             if video_url.startswith("//"): video_url = "https:" + video_url
                else:
                    logger.warning(f"JSON中未找到有效视频信息: videoInfoRes={bool(video_info)}")
            else:
                logger.warning("_ROUTER_DATA 未找到")
                
        except Exception as exc:
            logger.error("抖音页面解析失败: %s", type(exc).__name__)
            raise RuntimeError(f"抖音页面解析失败（{type(exc).__name__}）") from exc

    if not video_url: raise ValueError("无法获取视频地址")

    # 3. 下载视频
    video_path = await _download_video(video_url, video_id, job_id=job_id)

    return {
        "video_id": video_id, "title": title, "author": author,
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
            async with httpx.AsyncClient(headers=MOBILE_HEADERS, follow_redirects=True, timeout=120) as client:
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
