"""抖音视频解析模块 (API/Requests 版)"""
import asyncio
import os
import re
import json
import logging
import subprocess
import httpx
from typing import Optional
from app.config import TEMP_DIR

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
    remaining = text.replace(url, " ").strip()
    cleaners = [
        r'\d+\.?\d*\s+', r'复制.*?打开.*?抖音[，,]?\s*', r'看看[【\[]?[^】\]]*?的作品[】\]]?\s*',
        r'#\s*[^\s#]+\s*', r'@[^\s@]+\s*', r'"[^"]*\.\.\.\s*', 
        r'[A-Za-z]{2,4}:[/\\]\s*', r'[a-zA-Z]@[A-Za-z]\.[A-Z]{2}\s*', r'\d{1,2}/\d{1,2}\s*'
    ]
    for p in cleaners:
        remaining = re.sub(p, ' ', remaining)
    return re.sub(r'\s+', ' ', remaining).strip()


async def resolve_and_download(share_url: str) -> dict:
    """解析链接并下载视频"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    logger.info(f"解析URL: {share_url}")

    video_url, title, author, video_id = None, "未知标题", "未知作者", ""

    async with httpx.AsyncClient(headers=MOBILE_HEADERS, follow_redirects=True, timeout=30) as client:
        try:
            # 1. 获取 Video ID
            resp = await client.get(share_url)
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
            html = resp.text
            
            pattern = re.compile(r"window\._ROUTER_DATA\s*=\s*(.*?)</script>", re.DOTALL)
            match = pattern.search(html)
            
            if match:
                data = json.loads(match.group(1).strip())
                loader_data = data.get("loaderData", {})
                video_info = loader_data.get("video_(id)/page", {}).get("videoInfoRes") or \
                             loader_data.get("note_(id)/page", {}).get("videoInfoRes")
                
                if video_info:
                    item = video_info["item_list"][0]
                    title = item.get("desc", title)
                    author = item.get("author", {}).get("nickname", author)
                    
                    if "video" in item and "play_addr" in item["video"]:
                        url_list = item["video"]["play_addr"]["url_list"]
                        if url_list:
                             video_url = url_list[0].replace("playwm", "play")
                             if video_url.startswith("//"): video_url = "https:" + video_url
                else:
                    logger.warning("JSON中未找到 videoInfoRes")
            else:
                logger.warning("_ROUTER_DATA 未找到")
                
        except Exception as e:
            logger.error(f"解析过程出错: {e}")
            raise

    if not video_url: raise ValueError("无法获取视频地址")

    # 3. 下载视频
    video_path = await _download_video(video_url, video_id)

    return {
        "video_id": video_id, "title": title, "author": author,
        "video_path": video_path, "video_url": video_url,
    }


async def _download_video(video_url: str, video_id: str) -> str:
    """下载视频文件"""
    video_path = os.path.join(TEMP_DIR, f"{video_id}.mp4")
    if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
        return video_path

    logger.info(f"Downloading: {video_url[:60]}...")
    async with httpx.AsyncClient(headers=MOBILE_HEADERS, follow_redirects=True, timeout=120) as client:
        async with client.stream("GET", video_url) as resp:
            resp.raise_for_status()
            with open(video_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    f.write(chunk)

    if os.path.getsize(video_path) < 1024*500:
        logger.warning("下载文件过小，可能无效")
        
    return video_path


def extract_audio(video_path: str) -> str:
    """提取音频 (mp3, 16kHz, mono)"""
    audio_path = video_path.rsplit(".", 1)[0] + ".mp3"
    if os.path.exists(audio_path): return audio_path

    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-ab", "128k", "-ar", "16000", "-ac", "1", "-y", audio_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {proc.stderr[:200]}")

    return audio_path


def cleanup_files(video_id: str):
    """清理临时文件"""
    import glob
    for f in glob.glob(os.path.join(TEMP_DIR, f"{video_id}*")):
        try:
            if not os.path.isdir(f): os.remove(f)
        except: pass
